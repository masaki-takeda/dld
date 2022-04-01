#!/bin/sh
sudo docker run --rm --shm-size=32G --gpus all --name dld -v ${PWD}:/opt/dld -v /data1:/data1  -v /data2:/data2 -p 8888:8888 -p 6006:6006 -it kut/dld /bin/bash
