

print('test2')

from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def testarrymake(setname):
    """makes the test array input of setname output concated test array"""
    
    datadirect = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets'
    datadirect = pjoin(datadirect, setname)
    datakey = "fftdata{}set" .format(setname)
    

    for i in range(0, 8):
        exten = "set{}.mat" .format(i)
        a = sp.loadmat(datadirect + exten, squeeze_me = True)
        b = a.get(datakey)
        testset.append(b)

    

#normaldata1 = sp.loadmat('normalset1.mat')
#normaldataset1 = np.squeeze(normaldata1['fftdatanormalset1'])
#normaldata2 = sp.loadmat('normalset2.mat')
#normaldataset2 = np.squeeze(normaldata2['fftdatanormalset2'])
#normaldata3 = sp.loadmat('normalset3.mat')
#normaldataset3 = np.squeeze(normaldata3['fftdatanormalset3'])
#innerrace1 = sp.loadmat('12kDEinnerrace7dia0hpset1.mat')
#innerraceset1 = np.squeeze(innerrace1['fftdata12kDEinnerrace7dia0hpset1'])
#innerrace2 = sp.loadmat('12kDEinnerrace7dia0hpset2.mat')
#innerraceset2 = np.squeeze(innerrace2['fftdata12kDEinnerrace7dia0hpset2'])
#innerrace3 = sp.loadmat('12kDEinnerrace7dia0hpset3.mat')
#innerraceset3 = np.squeeze(innerrace3['fftdata12kDEinnerrace7dia0hpset3'])
#innerraceset3 = np.reshape(innerraceset3, (1,-1))
#normaldataset3 = np.reshape(normaldataset3, (1,-1))

set1 = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets\\7InnerRace0HPset9.mat'
InnerRace0HP = sp.loadmat(set1, squeeze_me = True)
InnerRace0HP = InnerRace0HP.get('fftdata7InnerRace0HPset')
InnerRace0HP = np.reshape(InnerRace0HP, (1,-1))

set2 = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets\\7Ball0HPset9.mat'
Ball0HP = sp.loadmat(set2, squeeze_me = True)
Ball0HP = Ball0HP.get('fftdata7Ball0HPset')
Ball0HP = np.reshape(Ball0HP, (1,-1))

set3 = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets\\7OuterRace0HP6set9.mat'
OuterRace0HP6 = sp.loadmat(set3, squeeze_me = True)
OuterRace0HP6 = OuterRace0HP6.get('fftdata7OuterRace0HP6set')
OuterRace0HP6 = np.reshape(OuterRace0HP6, (1,-1))

set4 = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets\\Normal0HPset9.mat'
Normal0HP = sp.loadmat(set4, squeeze_me = True)
Normal0HP = Normal0HP.get('fftdataNormal0HPset')
Normal0HP = np.reshape(Normal0HP, (1,-1))

set5 = 'C:\\Users\\Nick\\Desktop\\SeDesgn-master\\SeDesgn-master\\ML_Test\\Data\\Sets\\21OuterRace0HPset9.mat'
OuterRace210HP = sp.loadmat(set5, squeeze_me = True)
OuterRace210HP = OuterRace210HP.get('fftdata21OuterRace0HPset')
OuterRace210HP = np.reshape(OuterRace210HP, (1,-1))




testset = []

testarrymake('7InnerRace0HP')
testarrymake('7Ball0HP')
testarrymake('7OuterRace0HP6')
testarrymake('7InnerRace1HP')
testarrymake('7Ball1HP')
testarrymake('7OuterRace1HP6')
testarrymake('21InnerRace0HP')
testarrymake('21Ball0HP')
testarrymake('21OuterRace0HP')
testarrymake('21InnerRace1HP')
testarrymake('21Ball1HP')
testarrymake('21OuterRace1HP')
testarrymake('Normal0HP')
testarrymake('Normal1HP')


flags = []

for x in range (0, 14):
    for y in range(0,8):
        flags.append(x)

#changes lists into numpy ndarray format
testset = np.asarray(testset)
flags = np.asarray(flags)

scaler = StandardScaler()
#sets up neural network. solver is Limited Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm
#alpha adjusts the L2 penalty for regularization, hidden layers are neurons
clf = MLPClassifier(solver='lbfgs', alpha = 1e-5, hidden_layer_sizes=(100,90,80,70,60,50,40), random_state=1)
#scales the training and test sets.  Note the training set is actually labeled testset
scaler.fit(testset)
testset = scaler.transform(testset)
InnerRace0HP = scaler.transform(InnerRace0HP) 
Ball0HP = scaler.transform(Ball0HP)
OuterRace0HP6 = scaler.transform(OuterRace0HP6)
Normal0HP = scaler.transform(Normal0HP)
OuterRace210HP = scaler.transform(OuterRace210HP)

validset = [InnerRace0HP, Ball0HP, OuterRace0HP6, Normal0HP]
validset = np.asarray(validset)
validset = np.squeeze(validset)
validflags = [0, 1, 2, 12]
validflags = np.asarray(validflags)
#fits ANN to testset with flags
clf.fit(testset, flags)
#predicts specific data sets using the trained ANN
A = clf.predict(InnerRace0HP)
B = clf.predict(Ball0HP)
C = clf.predict(OuterRace0HP6)
D = clf.predict(Normal0HP)
F = clf.predict(OuterRace210HP)
E = clf.score(validset, validflags)
#output
print('Test A: Inner Race 0 HP 0.007 in. Defect')
print('Expected Result: [0] Actual Result:',A)
print('')
print('Test B: Ball 0 HP 0.007 in. Defect')
print('Expected Result: [1] Actual Result:',B)
print('')
print('Test C: Outer Race 0 HP 0.007 in. Defect')
print('Expected Result: [2] Actual Result:',C)
print('')
print('Test D: Normal 0 HP')
print('Expected Result: [12] Actual Result:',D)
print('')
print('Test F: Outer Race 0 HP 0.021 in. Defect')
print('Expected Result: [8] Actual Result:',F)
print('')


print('ok')