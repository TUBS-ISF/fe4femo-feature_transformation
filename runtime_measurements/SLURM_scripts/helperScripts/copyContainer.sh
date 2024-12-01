#!/bin/bash

rm -rf $HOME/feature_metrics/performance/enroot/
rm -rf $HOME/.local/share/enroot/pyxis_*

scp -i ~/.ssh/bwCloud_private -r ubuntu@134.60.155.236:/home/ubuntu/MA_docker/enroot $HOME/feature_metrics/performance/
