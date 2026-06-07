"""Tests for the MJPEG Huffman-table fixup (server/mjpeg.py).

Dependency-free: run directly (``python3 server/test_mjpeg.py``) or under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mjpeg import _STD_DHT_SEGMENT, ensure_huffman_tables  # noqa: E402

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
SOS = b"\xff\xda\x00\x0c..."  # start-of-scan marker + a little header-ish filler
SCAN = b"scan-entropy-data-bytes"

# A minimal MJPEG-style frame with NO Huffman table: SOI, APP0/AVI1, DQT, SOF0,
# then SOS + entropy data + EOI — mirrors what DHT-less USB webcams emit.
_NO_DHT = SOI + b"\xff\xe0\x00\x10AVI1\x00" + b"\xff\xdb\x00\x04qz" + \
    b"\xff\xc0\x00\x05sof01" + SOS + SCAN + EOI


def test_dht_segment_is_wellformed():
    # FFC4 marker + 2-byte length that matches the segment body.
    assert _STD_DHT_SEGMENT[:2] == b"\xff\xc4"
    declared = (_STD_DHT_SEGMENT[2] << 8) | _STD_DHT_SEGMENT[3]
    assert declared == len(_STD_DHT_SEGMENT) - 2 == 418


def test_inserts_tables_when_missing():
    out = ensure_huffman_tables(_NO_DHT)
    sos = out.find(b"\xff\xda")
    assert out.find(b"\xff\xc4", 0, sos) != -1, "DHT must be inserted before SOS"
    assert out[:2] == SOI and out[-2:] == EOI
    assert out.endswith(SOS + SCAN + EOI), "scan data must be left intact"
    assert len(out) == len(_NO_DHT) + len(_STD_DHT_SEGMENT)


def test_noop_when_tables_present():
    already = SOI + b"\xff\xdb\x00\x04qz" + _STD_DHT_SEGMENT + b"\xff\xc0\x00\x05sof01" + SOS + SCAN + EOI
    assert ensure_huffman_tables(already) is already


def test_noop_when_not_a_jpeg():
    junk = b"not a jpeg, no start-of-scan here"
    assert ensure_huffman_tables(junk) is junk


def test_idempotent():
    once = ensure_huffman_tables(_NO_DHT)
    twice = ensure_huffman_tables(once)
    assert once == twice


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
