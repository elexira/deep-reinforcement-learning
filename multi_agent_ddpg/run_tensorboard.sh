#!/bin/bash
tensorboard --logdir=./log/ --host localhost --port 8088 &> /dev/null &
echo "Wait around 10 seconds and open this link at your browser (ignore other outputs):"
echo "http://localhost:8088"
