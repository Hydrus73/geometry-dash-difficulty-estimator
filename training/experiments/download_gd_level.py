"""Fetch one public Geometry Dash level by ID and decode its object string."""
import argparse
import base64
import json
import os
import tempfile
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "training" / "level_data"
ENDPOINT = "https://www.boomlings.com/database/downloadGJLevel22.php"
COMMON_SECRET = "Wmfd2893gb7"


def split_level_response(response):
    segments = response.strip().split("#")
    level_segment = segments[0]
    pieces = level_segment.split(":")
    if len(pieces) % 2 != 0:
        raise ValueError("Malformed key:value level response")
    header = {}
    for index in range(0, len(pieces) - 1, 2):
        header[pieces[index]] = pieces[index + 1]
    if "4" not in header or not header["4"]:
        raise ValueError(f"Level data missing from response: {response[:200]!r}")
    return header, segments[1:]


def iter_level_objects(encoded, compressed_chunk_size=64 * 1024, decoded_chunk_size=1024 * 1024):
    """Yield parsed positioned objects without materializing the full level string."""
    compressed = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    decompressor = zlib.decompressobj(15 | 32)
    buffer = b""

    def parse_object(raw):
        values = raw.split(b",")
        if len(values) < 6 or len(values) % 2:
            return None
        object_id = x = y = None
        rotated = scaled = False
        for index in range(0, len(values) - 1, 2):
            key, value = values[index], values[index + 1]
            if key == b"1":
                object_id = value
            elif key == b"2":
                x = value
            elif key == b"3":
                y = value
            elif key == b"6":
                rotated = True
            elif key == b"32":
                scaled = True
        if object_id is None or x is None or y is None:
            return None
        try:
            return int(object_id), float(x), float(y), rotated, scaled
        except ValueError:
            return None

    for start in range(0, len(compressed), compressed_chunk_size):
        pending = compressed[start:start + compressed_chunk_size]
        while pending:
            decoded = decompressor.decompress(pending, decoded_chunk_size)
            pending = decompressor.unconsumed_tail
            buffer += decoded
            while True:
                separator = buffer.find(b";")
                if separator < 0:
                    break
                item = parse_object(buffer[:separator])
                if item is not None:
                    yield item
                buffer = buffer[separator + 1:]
    buffer += decompressor.flush()
    for raw in buffer.split(b";"):
        item = parse_object(raw)
        if item is not None:
            yield item


def download(level_id, cache_dir, timeout=30):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{level_id}.json"
    if cache_path.exists():
        return cache_path
    payload = urllib.parse.urlencode({"levelID": str(level_id), "secret": COMMON_SECRET}).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"User-Agent": "", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    if not text or text.strip() == "-1":
        raise RuntimeError(f"GD server did not return level {level_id}: {text!r}")
    header, _integrity = split_level_response(text)
    compressed = base64.urlsafe_b64decode(header["4"] + "=" * (-len(header["4"]) % 4))
    decoder = zlib.decompressobj(15 | 32)
    if not decoder.decompress(compressed, 4096):
        raise ValueError(f"level {level_id} has no decodable object data")
    # Preserve only the encoded level content. Server ratings, votes and other
    # header fields can leak label information and are not candidate inputs.
    record = {"level_id": int(level_id), "header": {"4": header["4"]}}
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=cache_dir, prefix=f".{level_id}.",
        suffix=".partial", delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        json.dump(record, temporary, separators=(",", ":"))
    try:
        os.replace(temp_path, cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return cache_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level_id", type=int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    path = download(args.level_id, args.cache_dir)
    record = json.loads(path.read_text(encoding="utf-8"))
    print(f"saved {path}")
    print(f"level_id={record['level_id']} compressed_chars={len(record['header']['4'])} "
          f"cache_bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
