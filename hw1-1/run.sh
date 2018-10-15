#!/bin/bash
cd src
wget 'https://www.dropbox.com/s/5q4674tj1t7ucdu/deepwalk-emb?dl=1' -O 'embeddings'
python get_input.py
python predict.py
python pred-txt-to-csv.py pred.txt
