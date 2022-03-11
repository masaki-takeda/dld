#!/bin/sh

if [ $# -ne 1 ]; then
  echo "Please specify log directory" 1>&2
  exit 1
fi

tensorboard --host 0.0.0.0 --logdir $1
