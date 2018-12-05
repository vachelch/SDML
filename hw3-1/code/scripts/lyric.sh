#! /usr/bin/env bash
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${BASE_DIR}/data/lyric"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"


# Create Vocabulary
python generate_vocab.py \
  --max_vocab_size 35000 \
  --outdir ${OUTPUT_DIR_TRAIN} \
  < ${OUTPUT_DIR_TRAIN}/data.txt 
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.source"
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.target"

cp "${OUTPUT_DIR_TRAIN}/vocab.source" $OUTPUT_DIR_DEV
cp "${OUTPUT_DIR_TRAIN}/vocab.target" $OUTPUT_DIR_DEV
cp "${OUTPUT_DIR_TRAIN}/vocab.source" $OUTPUT_DIR_TEST
cp "${OUTPUT_DIR_TRAIN}/vocab.target" $OUTPUT_DIR_TEST
