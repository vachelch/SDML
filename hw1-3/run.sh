#!/bin/bash
cd src
python tf-idf.py
python time.py
python gragh_feature.py
python clf.py
python ensemble.py
python pred-txt-to-csv.py pred.txt
