#!/bin/bash
cd code

pip install -r requirements.txt
python setup.py develop

echo 'download model 1 with attention...'
wget 'https://www.dropbox.com/s/62flq5b0uek3o8q/2018_11_28_00_12_17.tar.gz?dl=1' -O 2018_11_28_00_12_17.tar.gz 

echo 'download model 2 without attention...'
wget 'https://www.dropbox.com/s/9g1zoyh6hmx3ep5/2018_12_03_23_32_58.tar.gz?dl=1' -O 2018_12_03_23_32_58.tar.gz

tar -zxvf 2018_11_28_00_12_17.tar.gz 
tar -zxvf 2018_12_03_23_32_58.tar.gz 

mv -f 2018_11_28_00_12_17 experiment/checkpoints/2018_11_28_00_12_17
mv -f 2018_12_03_23_32_58 experiment/checkpoints/2018_12_03_23_32_58

TRAIN_PATH="$1"
DEV_PATH="$1"
OUTPUT_PATH="$2"

#########   unquote this part to get results with attention   #########
echo 'predict results 1 with attention'
MODEL_PATH=2018_11_28_00_12_17
python examples/test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --output_path=$OUTPUT_PATH --load_checkpoint=$MODEL_PATH --atten

########   unquote this part to get results without attention   #######
# echo 'predict results 2 without attention'
# MODEL_PATH=2018_12_03_23_32_58
# python examples/test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --output_path=$OUTPUT_PATH --load_checkpoint=$MODEL_PATH

cd ..
