"""Regenerate optional zero-shot Sol predictions from the paired Spider prompts."""

import argparse
import fcntl
import gzip
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for per-question generations",
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    args = parser.parse_args()
    if not args.base_url.startswith("https://"):
        parser.error("Use an HTTPS endpoint")
    key = os.environ["OPENAI_API_KEY"]
    model = "gpt-6.2-sol"
    args.output.mkdir(parents=True, exist_ok=True)
    client = urllib.request.build_opener(NoRedirect())
    with (
        (args.output / ".generation.lock").open("a") as lock,
        gzip.open(ROOT / "data/spider/sol_prompts.jsonl.gz", "rt") as prompts,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another generation is using this output directory")
        for line in prompts:
            item = json.loads(line)
            target = args.output / f"{item['qid']}.json"
            if target.exists():
                record = json.loads(target.read_text())
                if (
                    record["model"] != model
                    or record["qid"] != item["qid"]
                    or record["endpoint"] != args.base_url
                ):
                    raise ValueError(f"Conflicting saved response for {target.name}")
                continue
            body = dict(
                model=model,
                input=item["prompt"],
                tools=[],
                reasoning={"effort": "medium"},
                max_output_tokens=4096,
                store=False,
            )
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/responses",
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
            )
            with client.open(request, timeout=120) as f:
                reply = json.load(f)
            reported = reply.get("model", "")
            if reported != model and not reported.startswith(model + "-"):
                raise ValueError("Unexpected model in response")
            if reply.get("status") not in ("completed", "incomplete"):
                raise ValueError("Unexpected completion status")
            parts = [
                c
                for m in reply.get("output", [])
                if m.get("type") == "message"
                for c in m.get("content", [])
            ]
            sql = "\n".join(c["text"] for c in parts if c.get("type") == "output_text")
            record = dict(
                qid=item["qid"],
                model=model,
                reported_model=reported,
                endpoint=args.base_url,
                status=reply["status"],
                output_text=sql,
                refused=any(c.get("type") == "refusal" for c in parts),
            )
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(record, ensure_ascii=False) + "\n")
            temp.replace(target)


if __name__ == "__main__":
    main()
