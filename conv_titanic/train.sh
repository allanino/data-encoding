#!/usr/bin/env sh

TOOLS=/home/allan/Git/caffe/build/tools

GLOG_logtostderr=1 $TOOLS/train_net.bin caffe_solver.prototxt