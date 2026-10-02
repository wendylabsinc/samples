#!/bin/sh
set -eu
# Run from package root with clang/libjpeg development headers installed.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
clang -g -O1 -fsanitize=address -fno-omit-frame-pointer -I Sources/CameraC/include \
  Sources/CameraC/CameraC.c Tests/Native/thumbnail_failure.c -ljpeg -o "$work/test"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 "$work/test" Tests/CameraWireTests/Fixtures/source.jpg
