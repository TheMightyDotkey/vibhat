#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import vibhat
import zipfile

def main():
    """
    runs vibhat every minute, compresses vibhat data

    """
    FDT = vibhat.vibhat()
    print(FDT)
    print('\n')
    zipfile.ZipFile('FDT', 'w', zipfile.ZIP_DEFLATED).write(FDT, arcname=FDT)

if __name__ == '__main__':
    main()