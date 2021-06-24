import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
import scalogram

def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")
            

## -- 1 -- ##
def plot3Classes(data1,name1,data2,name2):
    plt.subplot(2,1,1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(2,1,2)
    plt.plot(data2)
    plt.title(name2)
  
    
    plt.show()
    waitforEnter()
    
## -- 2 -- ##
def breakTrainTest(data,oWnd=50,trainPerc=0.5):
    nSamp,nCols=data.shape
    nObs=int(nSamp/oWnd)
    data_obs=data[:nObs*oWnd,:].reshape((nObs,oWnd,nCols))
    
    # order=np.random.permutation(nObs)
    order=np.arange(nObs)    #Comment out to random split
    
    nTrain=int(nObs*trainPerc)
    
    data_train=data_obs[order[:nTrain],:,:]
    data_test=data_obs[order[nTrain:],:,:]
    
    return(data_train,data_test)

## -- 3 -- ##
def extractFeatures(data,Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        M1=np.mean(data[i,:,:],axis=0)
        #Md1=np.median(data[i,:,:],axis=0)
        Std1=np.std(data[i,:,:],axis=0)
        #S1=stats.skew(data[i,:,:])
        #K1=stats.kurtosis(data[i,:,:])
        p=[75,90,95]
        Pr1=np.array(np.percentile(data[i,:,:],p,axis=0)).T.flatten()
        
        #faux=np.hstack((M1,Md1,Std1,S1,K1,Pr1))
        faux=np.hstack((M1,Std1,Pr1))
        features.append(faux)
        
    return(np.array(features),oClass)

## -- 4 -- ##
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()
    
def logplotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r']
    for i in range(nObs):
        plt.loglog(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()
    waitforEnter()

## -- 5 -- ##
def extratctSilence(data,threshold=256):
    if(data[0]<=threshold):
        s=[1]
    else:
        s=[]
    for i in range(1,len(data)):
        if(data[i-1]>threshold and data[i]<=threshold):
            s.append(1)
        elif (data[i-1]<=threshold and data[i]<=threshold):
            s[-1]+=1
    
    return(s)
    
def extractFeaturesSilence(data,Class=0,threshold=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        silence_features=np.array([])
        for c in range(nCols):
            silence=extratctSilence(data[i,:,c],threshold)
            if len(silence)>0:
                silence_features=np.append(silence_features,[np.mean(silence),np.var(silence)])
            else:
                silence_features=np.append(silence_features,[0,0])
            
            
        features.append(silence_features)
        
    return(np.array(features),oClass)

## -- 7 -- ##

def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        scalo_features=np.array([])
        for c in range(nCols):
            #fixed scales->fscales
            scalo,fscales=scalogram.scalogramCWT(data[i,:,c],scales)
            scalo_features=np.append(scalo_features,scalo)
            
        features.append(scalo_features)
        
    return(np.array(features),oClass)
    
## -- 11 -- ##
def distance(c,p):
    return(np.sqrt(np.sum(np.square(p-c))))




## initial data
Classes={0:'Normal',1:'Infected'}

plt.ion()
normal=np.loadtxt('normal.dat')
infected = np.loadtxt('infectedMode4Big.dat')

bytesNormal = np.array(normal[:,[2,4]])
bytesInfected = np.array(infected[:,[2,4]])

packetNormal = np.array(normal[:,[1,3]])
packetInfected = np.array(infected[:,[1,3]])

plt.figure(1)
plot3Classes(bytesNormal,'bytesNormal',bytesInfected,'bytesInfected')
plt.figure(2)
plot3Classes(packetNormal,'packetNormal',packetInfected,'packetInfected')



## Divide dataset in obs windows

bytesNormal_train,bytesNormal_test=breakTrainTest(bytesNormal)
packetNormal_train,packetNormal_test=breakTrainTest(packetNormal)

bytesInfected_train,bytesInfected_test=breakTrainTest(bytesInfected)
packetInfected_train,packetInfected_test=breakTrainTest(packetInfected)

plt.figure(3)
plt.subplot(2,1,1)
print(len(bytesNormal_train))
for i in range(int(len(bytesNormal_train)/2)):
    plt.plot(bytesNormal_train[i,:,0],'b')
    plt.plot(bytesNormal_train[i,:,1],'g')
plt.title('Normal')
plt.ylabel('Bytes/sec')
plt.subplot(2,1,2)
for i in range(int(len(bytesInfected_train)/2)):
    plt.plot(bytesInfected_train[i,:,0],'b')
    plt.plot(bytesInfected_train[i,:,1],'g')
plt.title('Infected')
plt.ylabel('Bytes/sec')

plt.figure(4)
plt.subplot(2,1,1)
for i in range(int(len(packetNormal_train)/2)):
    plt.plot(packetNormal_train[i,:,0],'b')
    plt.plot(packetNormal_train[i,:,1],'g')
plt.title('Normal')
plt.ylabel('Packet/sec')
plt.subplot(2,1,2)
for i in range(int(len(packetInfected_train)/2)):
    plt.plot(packetInfected_train[i,:,0],'b')
    plt.plot(packetInfected_train[i,:,1],'g')
plt.title('Infected')
plt.ylabel('Packet/sec')

plt.show()
waitforEnter()

## extract features
################################################################################
#Train features Bytes #
train_features_b,oClass_bytes = extractFeatures(bytesNormal_train,Class=0)

#Train features Packets #
train_features_p,oClass_packet = extractFeatures(packetNormal_train,Class=0)

plt.figure(5)
plotFeatures(train_features_b,oClass_bytes,1,3)#0,8


plt.figure(6)
plotFeatures(train_features_p,oClass_packet,0,2)#0,8

# Test features Bytes #
test_features_b_n, oClass_bytesNormal = extractFeatures(bytesNormal_test,Class=0)
test_features_b_i, oClass_bytesInfected = extractFeatures(bytesInfected_test,Class=1)

test_features_b = np.vstack((test_features_b_n, test_features_b_i))

# Test features Packets #
test_features_p_n, oClass_bytesNormal = extractFeatures(packetNormal_test,Class=0)
test_features_p_i, oClass_bytesInfected = extractFeatures(packetInfected_test,Class=1)

test_features_p = np.vstack((test_features_p_n, test_features_p_i))

print('Train Bytes Features Size:',train_features_b.shape)
print('Train Packets Features Size:',train_features_p.shape)

print('Test Bytes Features Size:',test_features_b.shape)
print('Test Packets Features Size:',test_features_p.shape)


#####################################################################################
## Train time features Bytes ##
train_features_bS,oClass_bytesNormal = extractFeaturesSilence(bytesNormal_train,Class=0, threshold=128)

## Train time features Packets ##
train_features_pS,oClass_packetNormal = extractFeaturesSilence(packetNormal_train,Class=0,threshold = 4)

## Test time features Bytes ##
test_features_b_nS,oClass_bytesInfected = extractFeaturesSilence(bytesNormal_test,Class=0, threshold=128)
test_features_b_iS,oClass_bytesInfected = extractFeaturesSilence(bytesInfected_test,Class=1, threshold=128)

test_features_bS = np.vstack((test_features_b_nS, test_features_b_iS))

## Test time features Packets ##
test_features_p_nS,oClass_bytesInfected = extractFeaturesSilence(packetNormal_test,Class=0, threshold = 4)
test_features_p_iS,oClass_bytesInfected = extractFeaturesSilence(packetInfected_test,Class=1, threshold = 4)

test_features_pS = np.vstack((test_features_p_nS, test_features_p_iS))

print('Train Silence Bytes Features Size:',train_features_bS.shape)
print('Train Silence Packets Features Size:',train_features_pS.shape)

print('Test Silence Bytes Features Size:',test_features_bS.shape)
print('Test Silence Packets Features Size:',test_features_pS.shape)

####################################################################################
##  Wavelet features ##
scales=[2,4,8,16,32,64,128,256]

# Train features wavelet Bytes#
train_features_bW,oClass_b=extractFeaturesWavelet(bytesNormal_train,scales,Class=0)

# Train features wavelet Packets#
train_features_pW,oClass_p=extractFeaturesWavelet(packetNormal_train,scales,Class=1)

## Test time features Bytes ##
test_features_b_nW,oClass_b = extractFeaturesWavelet(bytesNormal_test,scales,Class=0)
test_features_b_iW,oClass_bi = extractFeaturesWavelet(bytesInfected_test,scales,Class=1)

test_features_bW = np.vstack((test_features_b_nW, test_features_b_iW))

# Test features wavelet Packets#
test_features_p_nW,oClass_p=extractFeaturesWavelet(packetNormal_test,scales,Class=0)
test_features_p_iW,oClass_pi=extractFeaturesWavelet(packetInfected_test,scales,Class=1)

test_features_pW = np.vstack((test_features_p_nW, test_features_p_iW))

print('Train Wavelet Bytes Features Size:',train_features_bW.shape)
print('Train Wavelet Packets Features Size:',train_features_pW.shape)

print('Test Wavelet Bytes Features Size:',test_features_bW.shape)
print('Test Wavelet Packets Features Size:',test_features_pW.shape)

plt.figure(9)
plotFeatures(train_features_bW,oClass_b,3,10)

o3testClassB=np.vstack((oClass_b, oClass_bi))
o3testClassP=np.vstack((oClass_p, oClass_pi))
#plt.figure(10)
#plotFeatures(train_features_pW,oClass_p,3,10)

#######################################################################################
## Creating trainSet with normal traffic ##

trainFeaturesBytes=np.hstack((train_features_b,train_features_bS,train_features_bW))

trainFeaturesPackets=np.hstack((train_features_p,train_features_pS,train_features_pW))

## Creating testSet ##
testFeaturesBytes=np.hstack((test_features_b,test_features_bS,test_features_bW))

testFeaturesPackets=np.hstack((test_features_p,test_features_pS,test_features_pW))

#######################################################################################
##  Normalizing features ##
from sklearn.preprocessing import MaxAbsScaler

## Normalizing train Bytes ##
trainScalerBytes = MaxAbsScaler().fit(trainFeaturesBytes)
trainFeaturesBytesN = trainScalerBytes.transform(trainFeaturesBytes)

## Normalizing train Packets ##
trainScalerPackets = MaxAbsScaler().fit(trainFeaturesPackets)
trainFeaturesPacketsN = trainScalerPackets.transform(trainFeaturesPackets)

## Normalizing test Bytes ##
testScalerBytes = MaxAbsScaler().fit(testFeaturesBytes)
testFeaturesBytesN = testScalerBytes.transform(testFeaturesBytes)

## Normalizing test Packets ##
testScalerPackets = MaxAbsScaler().fit(testFeaturesPackets)
testFeaturesPacketsN = testScalerPackets.transform(testFeaturesPackets)



print('Test Normalized Packets Features Size:',trainFeaturesBytesN.shape)


####################################################################################
## PCA Feature Reduction ##
from sklearn.decomposition import PCA

pca = PCA(n_components=3, svd_solver='full')

## PCA Train Bytes##
trainBPCA=pca.fit(trainFeaturesBytesN)
trainFeaturesBytesNPCA = trainBPCA.transform(trainFeaturesBytesN)

## PCA Train Packets ##

trainBPCA=pca.fit(trainFeaturesPacketsN)
trainFeaturesPacketsNPCA = trainBPCA.transform(trainFeaturesPacketsN)

## PCA Test Bytes##
testBPCA=pca.fit(testFeaturesBytesN)
testFeaturesBytesNPCA = testBPCA.transform(testFeaturesBytesN)

## PCA Test Packets##
testBPCA=pca.fit(testFeaturesPacketsN)
testFeaturesPacketsNPCA = testBPCA.transform(testFeaturesPacketsN)

print('Train Reduction Packets Features Size:',trainFeaturesBytesNPCA.shape)
plt.figure(11)
plotFeatures(trainFeaturesPacketsNPCA,oClass_p,0,1)

plt.figure(12)
plotFeatures(trainFeaturesBytesNPCA,oClass_b,0,2)

# -- 14 -- ##
from sklearn import svm

##WITH PCA
# MACHINE LEARNING FOR BYTES
print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) -- Bytes Analyze')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesBytesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesBytesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesBytesNPCA)  

L1=ocsvm.predict(testFeaturesBytesNPCA)
L2=rbf_ocsvm.predict(testFeaturesBytesNPCA)
L3=poly_ocsvm.predict(testFeaturesBytesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=testFeaturesBytesNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClassB[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

B1 =L1
B2= L2
B3=L3

## MACHINE LEARNING FOR PACKETS
print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) -- Bytes Analyze')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(trainFeaturesPacketsNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(trainFeaturesPacketsNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(trainFeaturesPacketsNPCA)  

L1=ocsvm.predict(testFeaturesPacketsNPCA)
L2=rbf_ocsvm.predict(testFeaturesPacketsNPCA)
L3=poly_ocsvm.predict(testFeaturesPacketsNPCA)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=testFeaturesPacketsNPCA.shape
for i in range(nObsTest):
    print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClassP[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))

print('\n-- Ensemble -- ')
print('\n-- with poly -- ')
for i in range(nObsTest):
    R1=B1[i]+L1[i]+B2[i]+L2[i]+B3[i]+L3[i]
    
    if(R1 == -6 or R1 == -4 or R1 == -2):
        R1=-1
    else:
        R1=1
        
    print('Obs: {:2} ({:<8}): Result->{:<10}'.format(i,Classes[o3testClassP[i][0]],AnomResults[R1]))

print('\n-- no poly -- ')
for i in range(nObsTest):
    R1=B1[i]+L1[i]+B2[i]+L2[i]
    
    if(R1 == -4 or R1 == -2):
        R1=-1
    else:
        R1=1
        
    print('Obs: {:2} ({:<8}): Result->{:<10}'.format(i,Classes[o3testClassP[i][0]],AnomResults[R1]))








plt.show(block=True)