#! /usr/bin/env python
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

#pylint: disable=invalid-name
"""
Generate vocabulary for a tokenized text file.
"""

import sys, os
import argparse
import collections
import logging

parser = argparse.ArgumentParser(
    description="Generate vocabulary for a tokenized text file.")
parser.add_argument(
    "--min_frequency",
    dest="min_frequency",
    type=int,
    default=0,
    help="Minimum frequency of a word to be included in the vocabulary.")
parser.add_argument(
    "--max_vocab_size",
    dest="max_vocab_size",
    type=int,
    help="Maximum number of tokens in the vocabulary")
parser.add_argument(
    "--downcase",
    dest="downcase",
    type=bool,
    help="If set to true, downcase all text before processing.",
    default=False)
parser.add_argument(
    "infile",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="Input tokenized text file to be processed.")
parser.add_argument(
    "--outdir",
    type=str,
    help="vocabulary output directory")
parser.add_argument(
    "--delimiter",
    dest="delimiter",
    type=str,
    default=" ",
    help="Delimiter character for tokenizing. Use \" \" and \"\" for word and char level respectively."
)
args = parser.parse_args()


def cnt_updata(line, cnt):
  if args.downcase:
    line = line.lower()
  if args.delimiter == "":
    tokens = list(line.strip())
  else:
    tokens = line.strip().split(args.delimiter)
  cnt.update(tokens)

def write_vocab(file_name, vocab):
  with open(file_name, 'w', encoding='utf8') as f:
    for word, count in vocab:
      f.write("{}\n".format(word))
  

def filter_vocab(cnt):
  logging.info("Found %d unique tokens in the vocabulary.", len(cnt))
  # Filter tokens below the frequency threshold
  if args.min_frequency > 0:
      filtered_tokens = [(w, c) for w, c in cnt.most_common()
                          if c > args.min_frequency]
      cnt = collections.Counter(dict(filtered_tokens))

  logging.info("Found %d unique tokens with frequency > %d.",
                  len(cnt), args.min_frequency)

  # Sort tokens by 1. frequency 2. lexically to break ties
  word_with_counts = cnt.most_common()
  word_with_counts = sorted(
      word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

  # Take only max-vocab
  if args.max_vocab_size is not None:
      word_with_counts = word_with_counts[:args.max_vocab_size]
  
  return word_with_counts

def main():
  # Counter for all tokens in the vocabulary
  cnt_source = collections.Counter()
  cnt_target = collections.Counter()

  for line in args.infile:
    source, target = line.split('\t')
    cnt_updata(source, cnt_source)
    cnt_updata(target, cnt_target)
  
  print('finish counting')

  cnt_source = filter_vocab(cnt_source)
  cnt_target = filter_vocab(cnt_target)

  write_vocab(os.path.join(args.outdir, 'vocab.source'), cnt_source)
  write_vocab(os.path.join(args.outdir, 'vocab.target'), cnt_target)
  
if __name__ == "__main__":
    main()




