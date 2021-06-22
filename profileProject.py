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
def breakTrainTest(data,oWnd=30,trainPerc=0.5):
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
normal=np.loadtxt('dataSetRefinedTabuas.dat')
infected = np.loadtxt('dataSetRefinedInfected.dat')

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

features_bytesNormal,oClass_bytesNormal = extractFeatures(bytesNormal_train,Class=0)
features_bytesInfected,oClass_bytesInfected = extractFeatures(bytesInfected_train,Class=1)


featuresBytes=np.vstack((features_bytesNormal,features_bytesInfected))
oClassBytes=np.vstack((oClass_bytesNormal,oClass_bytesInfected))

print('Train Stats FeaturesBytes Size:',featuresBytes.shape)


features_packetNormal,oClass_packetNormal = extractFeatures(packetNormal_train,Class=0)
features_packetInfected,oClass_packetInfected = extractFeatures(packetInfected_train,Class=1)


featuresPacket=np.vstack((features_packetNormal,features_packetInfected))
oClassPacket=np.vstack((oClass_packetNormal,oClass_packetInfected))

print('Train Stats FeaturesPacket Size:',featuresPacket.shape)


plt.figure(5)
plotFeatures(featuresBytes,oClassBytes,6,7)#0,8


plt.figure(6)
plotFeatures(featuresPacket,oClassPacket,8,9)#0,8


## time features



features_bytesNormalS,oClass_bytesNormal = extractFeaturesSilence(bytesNormal_train,Class=0)
features_bytesInfectedS,oClass_bytesInfected = extractFeaturesSilence(bytesInfected_train,Class=1)


featuresBytesS=np.vstack((features_bytesNormalS,features_bytesInfectedS))
oClassBytes=np.vstack((oClass_bytesNormal,oClass_bytesInfected))

print('Train Silence FeaturesBytes Size:',featuresBytesS.shape)


features_packetNormalS,oClass_packetNormal = extractFeaturesSilence(packetNormal_train,Class=0,threshold = 4)
features_packetInfectedS,oClass_packetInfected = extractFeaturesSilence(packetInfected_train,Class=1)


featuresPacketS=np.vstack((features_packetNormalS,features_packetInfectedS))
oClassPacket=np.vstack((oClass_packetNormal,oClass_packetInfected))

print('Train Silence FeaturesPacket Size:',featuresPacketS.shape)


plt.figure(7)
plotFeatures(featuresBytesS,oClassBytes,0,1)#0,8


plt.figure(8)
plotFeatures(featuresPacketS,oClassPacket,1,2)#0,8

















plt.show(block=True)