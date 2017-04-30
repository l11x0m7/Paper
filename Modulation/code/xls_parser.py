#!/usr/bin/env python
# coding=utf-8

import xlrd
import sys

def parser(filepath, savepath):
    sheets = xlrd.open_workbook(filepath)
    sheet = sheets.sheet_by_index(0)
    with open(savepath, 'wb') as fw:
        total_len = sum(1 for _ in sheet.get_rows())
        for i, row in enumerate(sheet.get_rows()):
            sys.stdout.write('\r\rDone {} / {}'.format(i + 1, total_len))
            fw.write('\t'.join(map(lambda kk: str(kk.value), row)) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    parser('../data/ask.xlsx', '../data/parse_mod_4ASK.txt')
    parser('../data/fsk.xlsx', '../data/parse_mod_2FSK.txt')
    parser('../data/qpsk.xlsx', '../data/parse_mod_QPSK.txt')
