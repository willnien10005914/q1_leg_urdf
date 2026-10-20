#!/usr/bin/env python3
"""Upload a local mp4 to gofile.io and print the download page URL."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: upload_gofile.py <file> [out.json]")
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    # Get a server
    with urllib.request.urlopen("https://api.gofile.io/servers", timeout=60) as r:
        servers = json.load(r)
    server = servers["data"]["servers"][0]["name"]
    boundary = "----cursorunbox"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"https://{server}.gofile.io/uploadFile",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    if resp.get("status") != "ok":
        raise SystemExit(resp)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix(".gofile.json")
    out.write_text(json.dumps(resp, indent=2) + "\n")
    print(resp["data"]["downloadPage"])
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
