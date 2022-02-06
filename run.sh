#! /bin/bash

set -e
set -x

python -u dataset.py -j 8 |& tee dataset.log

