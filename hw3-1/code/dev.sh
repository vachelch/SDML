TRAIN_PATH=data/lyric/train/data.txt
DEV_PATH=data/lyric/test/data.txt
OUTPUT_PATH=data/lyric/test/no-atten-output.txt
MODEL_PATH=2018_12_03_23_32_58

# Start training
python examples/test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --output_path=$OUTPUT_PATH --load_checkpoint=$MODEL_PATH
