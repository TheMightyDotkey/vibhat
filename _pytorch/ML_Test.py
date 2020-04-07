
print('test')

from io import StringIO
from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
import pandas as pd

def datasetmaker(offset, matfilename):
    """offset is multiple of 256 matfile name is output name"""

    #Computes FFT of dataset
    Fs = 50000
    T = 1/Fs
    L = 50000
    t = np.arange(0, L) * T
    a = 0 + 50000 * offset
    b = 49999 + 50000 * offset
    matfilenamedat = "{}{}.csv" .format(matfilename,offset) #Makes dataset filename
    matfilenamedat = pjoin(data_dir + '\\Sets', matfilenamedat)
    #vectorname = "fftdata{}" .format(matfilename)
    S = data[a:b]

    f = Fs * np.arange(0, L//2 + 1) / L

    Y = np.fft.fft(S, axis = 0)
    P2 = abs(Y/L)
    P1 = P2[0:L//2+1]
    P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]
    #P1 = P1/max(P1)  Standardizes the data.  Was removed to use the scaler function later on
    P1[0] = 0
    np.savetxt(matfilenamedat, P1, delimiter=',')
    #sp.savemat(matfilenamedat, {vectorname:P1})
    #sp.savemat('timefile.mat', {'time':f})

    plt.plot(f,P1)
    plt.show()

matname = 'Good'

data_dir = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\NickRealData'
mat_fname = pjoin(data_dir, matname + '.mat')



df = pd.read_csv('C:\\Users\\Nick\\source\\repos\\vibhat\\vibhat\\_pytorch\\2020_March_03_065029.csv')


bearingdata = np.array(df)
bearingdata = bearingdata[:,1]
#sorted(bearingdata.keys())
print(bearingdata)

data = bearingdata
print(data)

dataname = matname + 'set'

datasetmaker(0, dataname)
datasetmaker(1, dataname)
datasetmaker(2, dataname)
datasetmaker(3, dataname)




#N = 1000
#T = 1 / 800
#x = np.linspace(0, N*T, N-1)
#y = 2*np.sin(20*x*2*np.pi) + 6*np.sin(150*x*2*np.pi)
#plt.plot(y)
#ffty = np.fft.fft(y)
#Spec = abs(ffty/N)
#Dat = Spec[1:N//2+1]
#Dat[2:len(Dat)-1] = 2 * Dat[2:len(Dat)-1]
#f = 800 * np.linspace(0, 1, N//2) / N
#plt.plot(f, Dat)


#Fs = 1000
#T = 1/Fs
#L = 1500
#t = np.arange(0, L) * T

#S = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

#f = Fs * np.arange(0, L//2 + 1) / L

#Y = np.fft.fft(S)
#P2 = abs(Y/L)
#P1 = P2[0:L//2+1]
#P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]

#plt.plot(f,P1)


#plt.plot(fftx, np.abs(ffty[0:N//2]) * 2 / len(ffty))

#plt.show()

#Fs = 12000
#T = 1/Fs
#L = 256
#t = np.arange(0, L) * T

#S = data[0:256]

#f = Fs * np.arange(0, L//2 + 1) / L

#Y = np.fft.fft(S, axis = 0)
#P2 = abs(Y/L)
#P1 = P2[0:L//2+1]
#P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]

#sp.savemat('normaldata1.mat', {'fftdata':P1})


#plt.plot(f, P1)
#plt.show()



#plt.plot(data[0:100])

#plt.show()

#fftdata = np.fft.fft(data)

#plt.plot(fftdata[0:100])

#plt.show()
