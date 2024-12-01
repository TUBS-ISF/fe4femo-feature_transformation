#!/bin/bash

rm -rf $HOME/fe4femo/runtime_measurements/container/enroot/
rm -rf $HOME/.local/share/enroot/pyxis_*

scp -i ~/.ssh/bwCloud_private -r ubuntu@134.60.155.236:/home/ubuntu/MA/MA_docker/enroot $HOME/fe4femo/runtime_measurements/container/
