#!/bin/bash
cd src
wget 'https://www.dropbox.com/s/iowhfprgzjus27a/t3-all.zip?dl=1' -O 't3-all.zip'
unzip t3-all.zip
python tf-idf.py
python time.py
python gragh_feature.py
python clf.py
python ensemble.py
python pred-txt-to-csv.py pred.txt
