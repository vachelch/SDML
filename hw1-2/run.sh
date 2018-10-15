#!/bin/bash
cd src
python tf-idf.py
python gragh_feature.py
python clf.py
python pred-txt-to-csv.py pred.txt
