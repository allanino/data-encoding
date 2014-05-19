#!/usr/bin/env sh

python make_names_list.py

rm -r train.leveldb
rm -r test.leveldb
rm mean.binaryproto

convert_imageset.bin images/ listnames_train.txt train.leveldb 1
convert_imageset.bin images/ listnames_test.txt test.leveldb 1
compute_image_mean.bin train.leveldb mean.binaryproto