#!/usr/bin/env python3
import argparse
import re
import boto3
from pathlib import Path

S3_PREFIX = "raw"
LOCAL_DATA = Path(__file__).resolve().parent.parent / "data" / "movielens-1m"
FILES = ["ratings.dat", "movies.dat", "users.dat"]
DELIM_FROM = "::"
DELIM_TO = ";"

EXPECTED_FIELDS = {"ratings.dat": 4, "movies.dat": 3, "users.dat": 5}
_HTML_NUM_ENTITY = re.compile(r"&#(\d+);")
_HTML_NAMED_ENTITY = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&nbsp;": " ",
}


def _normalize(content: bytes) -> str:
    text = content.decode("latin-1")

    def replace_num(m: re.Match) -> str:
        cp = int(m.group(1))
        return chr(cp)

    for ent, val in _HTML_NAMED_ENTITY.items():
        text = text.replace(ent, val)
    return _HTML_NUM_ENTITY.sub(replace_num, text)


def _validate_content(text: str, expected_fields: int, name: str) -> None:
    for ln, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        if len(line.split(DELIM_TO)) != expected_fields:
            raise ValueError(
                f"Malformed row #{ln} in {name}: expected {expected_fields} "
                f"fields, got {len(line.split(DELIM_TO))}: {line[:120]!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload raw MovieLens data to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    for name in FILES:
        path = LOCAL_DATA / name
        subdir = Path(name).stem
        key = f"{S3_PREFIX}/{subdir}/{name}"
        text = _normalize(path.read_bytes()).replace(DELIM_FROM, DELIM_TO)
        _validate_content(text, EXPECTED_FIELDS[name], name)
        content = text.encode("utf-8")
        s3.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=content,
            ContentType="text/plain",
        )
        print(f"Uploaded {path} -> s3://{args.bucket}/{key} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
