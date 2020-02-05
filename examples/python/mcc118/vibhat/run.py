#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import vibhat
import zipfile
import os

def main():
    """
    runs vibhat every minute, compresses vibhat data

    """
    FDT = vibhat.vibhat()
    print(FDT)
    print('\n')
    zipname = FDT[:-4]
    filenameonly = FDT.split(os.sep)[-1]
    zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED).write(FDT, filenameonly)

if __name__ == '__main__':
    main()