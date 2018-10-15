#!/usr/bin/env python3

'''
    Required: pred.txt

    To produce the corresponding pred.txt.csv.
'''

import sys


def main(pred_file):
    with open(pred_file, 'r') as f, open(pred_file + '.csv', 'w') as g:
        print('query_id,prediction', file=g)
        for idx, line in enumerate(f):
            print('%d,%d' % (1 + idx, int(line)), file=g)


if __name__ == '__main__':
    if len(sys.argv) != 1 + 1:
        print('Input error. (Usage: python3 %s <pred.txt>)' % (sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
