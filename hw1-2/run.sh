#!/bin/bash
cd src
wget 'https://www.dropbox.com/s/ip1f9gyboylxipa/t2-all.zip?dl=1' -O 't2-all.zip'
unzip t2-all.zip
python tf-idf.py
python gragh_feature.py
python clf.py
python pred-txt-to-csv.py pred.txt
