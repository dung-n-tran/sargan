#!/bin/bash

python radarconf19_generate_snr_table.py --save -e 191
cd /data/dung/sargan/radarconf19_paper/22074730sjjkfjywrsmx
git pull
git add .
git commit -m "Update snr table"
git push
cd /home/dung/Development/sargan_v4
