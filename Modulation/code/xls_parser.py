#!/usr/bin/env python
# coding=utf-8

import xlrd
import sys

def parser(filepath, savepath, direction='row'):
    sheets = xlrd.open_workbook(filepath)
    sheet = sheets.sheet_by_index(0)
    with open(savepath, 'wb') as fw:
        if direction == 'row':
            total_len = sheet.nrows
            for i, row in enumerate(sheet.get_rows()):
                sys.stdout.write('\r\rDone {} / {}'.format(i + 1, total_len))
                fw.write('\t'.join(map(lambda kk: str(kk.value), row)) + '\n')
                sys.stdout.flush()
        elif direction == 'col':
            total_len = sheet.ncols
            for i in xrange(total_len):
                col = sheet.col_values(i)
                sys.stdout.write('\r\rDone {} / {}'.format(i + 1, total_len))
                fw.write('\t'.join(map(lambda kk: str(kk), col)) + '\n')
                sys.stdout.flush()
        else:
            raise ValueError("`direction` must be in `['row', 'col']`")



if __name__ == '__main__':
    parser('../data/5dB/ask.xlsx', '../data/5dB/parse_mod_4ASK.txt', 'col')
    parser('../data/5dB/fsk.xlsx', '../data/5dB/parse_mod_2FSK.txt', 'col')
    parser('../data/5dB/qpsk.xlsx', '../data/5dB/parse_mod_QPSK.txt', 'col')
