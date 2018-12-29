#!/bin/bash

python radarconf19_sargan_vs_omp_recovery.py --save -d 20 -s clustered -e 181

cd BoomSARSimulation_light

matlab -r "run_generate_sar_images"

cd ..

python radarconf19_visualize_sargan_vs_omp_recovery.py --save -s clustered

cd /data/dung/sargan/radarconf19_paper/22074730sjjkfjywrsmx
git pull
git add .
git commit -m "Added clustered scene recovery figures"
git push
cd /home/dung/Development/sargan_v4
