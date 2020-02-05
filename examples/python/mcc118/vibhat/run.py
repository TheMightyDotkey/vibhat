#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import vibhat
import zipfile
import os
from time import sleep

def main():
    """
    runs vibhat every minute, compresses vibhat data

    """
    while True:

        FDT = vibhat.vibhat()
        print(FDT)
        print('\n')
        zipname = FDT[:-4]
        filenameonly = FDT.split(os.sep)[-1]
        zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED).write(FDT, filenameonly)
        os.remove(FDT)
        sleep(60)
        

if __name__ == '__main__':
    main()