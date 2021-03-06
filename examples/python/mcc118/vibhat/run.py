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
        sleep(5)
        os.system('rclone --config="/home/pi/.config/rclone/rclone.conf" copy /home/pi/New seniordesign:')
        os.system('rclone --config="/home/pi/.config/rclone/rclone.conf" copy /home/pi/Documents/Measurement_Computing/Scanning_log_files seniordesign:')
        sleep(60)
        

if __name__ == '__main__':
    main()