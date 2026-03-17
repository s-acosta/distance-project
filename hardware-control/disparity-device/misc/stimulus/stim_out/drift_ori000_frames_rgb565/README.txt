Each file is one raw RGB565 frame, row-major LITTLE-endian, size 240x240.
File name encodes frame index. Intended for Teensy: send header "IMG W H\n" then the bytes.
