from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def scalarize_edge_objectives(
    edge_objectives: np.ndarray,
    preference: Sequence[float],
    latency_scale: float,
    fee_scale: float,
) -> np.ndarray:
    pref = np.asarray(preference, dtype=np.float64)
    scaled = np.asarray(edge_objectives, dtype=np.float64).copy()
    scaled[..., 0] /= max(1e-8, latency_scale)
    scaled[..., 1] /= max(1e-8, fee_scale)
    return (scaled * pref).sum(axis=-1)


def path_objectives(
    sample: Dict, path: Sequence[int], use_true: bool = True
) -> np.ndarray:
    key = "true_edge_objectives" if use_true else "obs_edge_objectives_proxy"
    feats = np.asarray(sample[key], dtype=np.float64)
    if not path:
        return np.zeros((feats.shape[-1],), dtype=np.float64)
    return feats[np.asarray(path, dtype=np.int64)].sum(axis=0)


def path_scalar_cost(
    sample: Dict,
    path: Sequence[int],
    preference: Sequence[float] | None = None,
    use_true: bool = True,
) -> float:
    if preference is None:
        preference = sample["request"]["preference"]
    obj = path_objectives(sample, path, use_true=use_true)
    scales = np.asarray(
        [
            sample["request"]["scale_latency"],
            sample["request"]["scale_fee"],
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    return float(np.dot(np.asarray(preference, dtype=np.float64), obj / scales))


def path_success_probability(sample: Dict, path: Sequence[int]) -> float:
    if not path:
        return 0.0
    penalties = np.asarray(sample["true_edge_objectives"], dtype=np.float64)[
        np.asarray(path, dtype=np.int64), 2
    ]
    return float(np.exp(-penalties.sum()))


def path_is_feasible(sample: Dict, path: Sequence[int]) -> bool:
    req = sample["request"]
    if not path or len(path) > req["max_hops"]:
        return False
    src = sample["src"]
    dst = sample["dst"]
    capacities = np.asarray(sample["capacity"], dtype=np.float64)
    cur = req["source"]
    visited = {cur}
    lat, fee = 0.0, 0.0
    forb_nodes = set(req["forbidden_nodes"])
    forb_edges = set(req["forbidden_edges"])
    for ei in path:
        if ei in forb_edges or src[ei] != cur or capacities[ei] + 1e-8 < req["demand"]:
            return False
        nxt = int(dst[ei])
        if nxt in forb_nodes or (nxt in visited and nxt != req["target"]):
            return False
        lat += float(sample["true_edge_objectives"][ei][0])
        fee += float(sample["true_edge_objectives"][ei][1])
        if lat > req["max_latency"] + 1e-8 or fee > req["max_fee"] + 1e-8:
            return False
        cur = nxt
        visited.add(cur)
    return cur == req["target"]


def _reverse_shortest_path_distances(
    num_nodes: int,
    reverse_adj: Sequence[Sequence[Tuple[int, int]]],
    edge_weights: Sequence[float] | np.ndarray,
    target: int,
) -> np.ndarray:
    dist = np.full((int(num_nodes),), np.inf, dtype=np.float64)
    dist[int(target)] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, int(target))]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist > float(dist[node]) + 1e-12:
            continue
        for prev_node, edge_id in reverse_adj[node]:
            next_dist = cur_dist + float(edge_weights[edge_id])
            if next_dist + 1e-12 < float(dist[prev_node]):
                dist[prev_node] = next_dist
                heapq.heappush(heap, (next_dist, int(prev_node)))
    return dist


def _build_usable_graph(
    sample: Dict,
    *,
    allowed_edges: Optional[Sequence[int]],
) -> Tuple[Dict[int, List[int]], List[List[Tuple[int, int]]], List[int]]:
    req = sample["request"]
    src = sample["src"]
    dst = sample["dst"]
    capacities = np.asarray(sample["capacity"], dtype=np.float64)
    forb_edges = set(req["forbidden_edges"])
    demand = float(req["demand"])
    num_nodes = int(sample["num_nodes"])
    adj = {i: [] for i in range(num_nodes)}
    reverse_adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    usable_edges: List[int] = []
    edge_iter = range(len(src)) if allowed_edges is None else allowed_edges
    for raw_edge_id in edge_iter:
        edge_id = int(raw_edge_id)
        if edge_id in forb_edges or capacities[edge_id] + 1e-8 < demand:
            continue
        prev_node = int(src[edge_id])
        next_node = int(dst[edge_id])
        adj[prev_node].append(edge_id)
        reverse_adj[next_node].append((prev_node, edge_id))
        usable_edges.append(edge_id)
    return adj, reverse_adj, usable_edges


def _reverse_unit_hop_bounds(
    num_nodes: int, reverse_adj: Sequence[Sequence[Tuple[int, int]]], target: int
) -> np.ndarray:
    dist = np.full((int(num_nodes),), np.inf, dtype=np.float64)
    dist[int(target)] = 0.0
    q: deque[int] = deque([int(target)])
    while q:
        node = int(q.popleft())
        next_hops = float(dist[node]) + 1.0
        for prev_node, _edge_id in reverse_adj[node]:
            if next_hops + 1e-12 < float(dist[prev_node]):
                dist[prev_node] = next_hops
                q.append(int(prev_node))
    return dist


def enumerate_feasible_paths(
    sample: Dict,
    edge_costs: Optional[Sequence[float]] = None,
    allowed_edges: Optional[Sequence[int]] = None,
    max_candidates: Optional[int] = 64,
) -> List[Dict]:
    req = sample["request"]
    source, target = req["source"], req["target"]
    dst = sample["dst"]
    n = sample["num_nodes"]
    limit = None if max_candidates is None else int(max_candidates)
    if limit is not None and limit <= 0:
        return []
    true_obj = np.asarray(sample["true_edge_objectives"], dtype=np.float64)
    if edge_costs is None:
        costs = scalarize_edge_objectives(
            true_obj,
            req["preference"],
            req["scale_latency"],
            req["scale_fee"],
        )
    else:
        costs = np.asarray(edge_costs, dtype=np.float64)
    adj, reverse_adj, _usable_edges = _build_usable_graph(
        sample, allowed_edges=allowed_edges
    )
    for node in adj:
        adj[node].sort(key=lambda ei: float(costs[ei]))
    forb_nodes = set(req["forbidden_nodes"])
    max_hops = int(req["max_hops"])
    max_latency = float(req["max_latency"])
    max_fee = float(req["max_fee"])
    out: List[Dict] = []

    non_negative = bool(np.min(costs) >= 0.0) if len(costs) > 0 else True
    score_lb = (
        _reverse_shortest_path_distances(n, reverse_adj, costs, target)
        if non_negative
        else None
    )
    hop_lb = _reverse_unit_hop_bounds(n, reverse_adj, target)
    lat_lb = _reverse_shortest_path_distances(n, reverse_adj, true_obj[:, 0], target)
    fee_lb = _reverse_shortest_path_distances(n, reverse_adj, true_obj[:, 1], target)
    if (
        not math.isfinite(float(hop_lb[source]))
        or float(hop_lb[source]) > max_hops + 1e-12
        or not math.isfinite(float(lat_lb[source]))
        or float(lat_lb[source]) > max_latency + 1e-8
        or not math.isfinite(float(fee_lb[source]))
        or float(fee_lb[source]) > max_fee + 1e-8
    ):
        return []
    if limit is not None and non_negative:
        heap: List[
            Tuple[
                float,
                float,
                int,
                int,
                int,
                float,
                float,
                Tuple[int, ...],
                frozenset[int],
            ]
        ] = []
        push_id = 0
        heapq.heappush(
            heap,
            (
                float(score_lb[source]),
                0.0,
                0,
                push_id,
                source,
                0.0,
                0.0,
                tuple(),
                frozenset({source}),
            ),
        )
        while heap and len(out) < limit:
            _priority, score, _depth, _push_id, node, lat, fee, path, visited = (
                heapq.heappop(heap)
            )
            if (
                len(path) + float(hop_lb[node]) > max_hops + 1e-12
                or lat + float(lat_lb[node]) > max_latency + 1e-8
                or fee + float(fee_lb[node]) > max_fee + 1e-8
            ):
                continue
            if node == target:
                if path and path_is_feasible(sample, path):
                    obj = path_objectives(sample, path, use_true=True)
                    out.append(
                        {
                            "path": list(path),
                            "objectives": obj.tolist(),
                            "score": float(score),
                        }
                    )
                continue
            if len(path) >= max_hops:
                continue
            for ei in adj[node]:
                nxt = int(dst[ei])
                if nxt in forb_nodes or (nxt in visited and nxt != target):
                    continue
                nlat = lat + float(true_obj[ei][0])
                nfee = fee + float(true_obj[ei][1])
                if nlat > max_latency + 1e-8 or nfee > max_fee + 1e-8:
                    continue
                npath = path + (int(ei),)
                if (
                    len(npath) + float(hop_lb[nxt]) > max_hops + 1e-12
                    or not math.isfinite(float(hop_lb[nxt]))
                    or nlat + float(lat_lb[nxt]) > max_latency + 1e-8
                    or not math.isfinite(float(lat_lb[nxt]))
                    or nfee + float(fee_lb[nxt]) > max_fee + 1e-8
                    or not math.isfinite(float(fee_lb[nxt]))
                    or not math.isfinite(float(score_lb[nxt]))
                ):
                    continue
                push_id += 1
                nscore = float(score + costs[ei])
                nvisited = visited if nxt == target else frozenset(set(visited) | {nxt})
                heapq.heappush(
                    heap,
                    (
                        float(nscore + score_lb[nxt]),
                        nscore,
                        len(npath),
                        push_id,
                        nxt,
                        nlat,
                        nfee,
                        npath,
                        nvisited,
                    ),
                )
        out.sort(key=lambda x: x["score"])
        return out

    def dfs(
        node: int, visited: set[int], path: List[int], lat: float, fee: float
    ) -> None:
        if limit is not None and len(out) >= limit:
            return
        if (
            len(path) + float(hop_lb[node]) > max_hops + 1e-12
            or not math.isfinite(float(hop_lb[node]))
            or lat + float(lat_lb[node]) > max_latency + 1e-8
            or not math.isfinite(float(lat_lb[node]))
            or fee + float(fee_lb[node]) > max_fee + 1e-8
            or not math.isfinite(float(fee_lb[node]))
        ):
            return
        if node == target:
            if path and path_is_feasible(sample, path):
                obj = path_objectives(sample, path, use_true=True)
                score = float(costs[np.asarray(path, dtype=np.int64)].sum())
                out.append(
                    {"path": list(path), "objectives": obj.tolist(), "score": score}
                )
            return
        if len(path) >= max_hops:
            return
        for ei in adj[node]:
            nxt = int(dst[ei])
            if nxt in forb_nodes or (nxt in visited and nxt != target):
                continue
            nlat = lat + float(true_obj[ei][0])
            nfee = fee + float(true_obj[ei][1])
            if nlat > max_latency + 1e-8 or nfee > max_fee + 1e-8:
                continue
            if (
                len(path) + 1 + float(hop_lb[nxt]) > max_hops + 1e-12
                or not math.isfinite(float(hop_lb[nxt]))
                or nlat + float(lat_lb[nxt]) > max_latency + 1e-8
                or not math.isfinite(float(lat_lb[nxt]))
                or nfee + float(fee_lb[nxt]) > max_fee + 1e-8
                or not math.isfinite(float(fee_lb[nxt]))
            ):
                continue
            path.append(ei)
            added = nxt not in visited
            if added:
                visited.add(nxt)
            dfs(nxt, visited, path, nlat, nfee)
            if added:
                visited.remove(nxt)
            path.pop()
            if limit is not None and len(out) >= limit:
                return

    dfs(source, {source}, [], 0.0, 0.0)
    out.sort(key=lambda x: x["score"])
    return out if limit is None else out[:limit]
