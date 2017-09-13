#!/usr/bin/env bash
mkdir snapshot
mkdir log
rm ./log/3D_DenseSeg.log
/home/toanhoi/caffe/build/tools/caffe train --solver=solver.prototxt -gpu 0 2>&1 | tee -a ./log/3D_DenseSeg.log

