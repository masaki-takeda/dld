#!/bin/sh
sudo docker run --rm --shm-size=16G --gpus all --name simul -v ${PWD}:/opt/simul -v /data1:/data1 -it kut/simul /bin/bash
