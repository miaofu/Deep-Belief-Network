# -*- coding: cp936 -*-
import numpy as np
from DBN import *
def makedata():
    import scipy.io as sio      
    f=sio.loadmat('mnist_all.mat')
    batchdata = np.random.rand(numcases,numdims ,numbatches)*np.nan
    targetbatchdata = np.random.rand(numcases,1 ,numbatches)*np.nan
    flag= range(0,784,4)
    for batch in range(numbatches):
        t= np.random.randint(numcases-1)+1
        batchdata[0:t,:,batch]= (f['train0'][batch*numcases:batch*numcases+t,flag]>255/2)
        targetbatchdata[0:t,:,batch]= np.zeros((t, 1))
        batchdata[t:,:,batch]= (f['train1'][batch*numcases+t:batch*numcases+numcases,flag]>255/2)
        targetbatchdata[t:,:,batch]= np.ones((numcases-t,1))
    return batchdata,targetbatchdata

def maketestdata(start,numcases):
    import scipy.io as sio      
    f=sio.loadmat('mnist_all.mat')
    batchdata = np.random.rand(2*numcases,numdims )*np.nan
    targetbatchdata = np.random.rand(2*numcases,1)*np.nan
    flag= range(0,784,4)
    batchdata[0:numcases,:]= (f['test0'][start:start+numcases,flag]>255/2)
    targetbatchdata[0:numcases,:]= np.zeros((numcases, 1))
    batchdata[numcases:,:]= (f['train1'][start:start+numcases,flag]>255/2)
    targetbatchdata[numcases:,:]= np.ones((numcases,1))
    return batchdata,targetbatchdata

maxepoch=10;
alpha=5.0;
numcases,numdims ,numbatches =30,784/4,100
batchdata,targetbatchdata =makedata()
(numcases,numdims ,numbatches)=batchdata.shape
N=numcases;
data_test,target_test=maketestdata(0,900)

l1=50
l2 =30
l3=20
l4=10

USE_RBM=True
if USE_RBM:
    W1,c1,b1 = trainRBM(numdims,l1,1,batchdata,30)
    batchdata1= useRBM(batchdata,W1,c1,b1)
    W2,c2,b2 = trainRBM(l1,l2,1,batchdata1,50)
    batchdata2= useRBM(batchdata1,W2,c2,b2)
    W3,c3,b3 = trainRBM(l2,l3,1,batchdata2,50)
    batchdata3= useRBM(batchdata2,W3,c3,b3)
    W4,c4,b4 = trainRBM(l3,l4,1,batchdata3,50)
    w1=W1.T;c1=c1.T;w2=W2.T;c2=c2.T;
    w3=W3.T;c3=c3.T;
    w4=W4.T;c4=c4.T;
else:
    w1 = 0.*np.random.randn(numdims,l1)
    c1= 0*np.random.rand(1,l1)
    w2 = 0.*np.random.rand(l1,l2)
    c2= 0*np.random.rand(1,l2)
    w3 = 0.*np.random.randn(l2,l3)
    c3= 0*np.random.rand(1,l3)
    w4 = 0.*np.random.randn(l3,l4)
    c4= 0*np.random.rand(1,l4)
l5=1
w_class = 0.01*np.random.randn(l4,l5)
c_class = 0*np.random.rand(1,l5)


#BP
test_err=np.empty(maxepoch);
train_err=np.empty(maxepoch);

for epoch in range(maxepoch):
    # test error
    err=0;
    data = data_test
    target = target_test
    M= data.shape[0]
    h1 = 1/(1 + np.exp(-np.dot(data,w1)- np.tile(c1,(M,1)) )); 
    h2 = 1/(1 + np.exp(-np.dot(h1,w2)-np.tile(c2,(M,1)) ));   
    h3 = 1/(1 + np.exp(-np.dot(h2,w3)-np.tile(c3,(M,1)) ));  
    h4 = 1/(1+np.exp(-np.dot(h3,w4)-np.tile(c4,(M,1))));   
    dataout = 1/(1 + np.exp(-np.dot(h4,w_class)-np.tile(c_class,(M,1))));
    err= err +   ((target-dataout)**2 ).sum();
    test_err[epoch]=err
    #train error
    err=0;
    for batch in range(numbatches):
        data = batchdata[:,:,batch];
        target = targetbatchdata[:,:,batch];
        h1 = 1/(1 + np.exp(-np.dot(data,w1)- np.tile(c1,(N,1)) )); 
        h2 = 1/(1 + np.exp(-np.dot(h1,w2)-np.tile(c2,(N,1)) ));   
        h3 = 1/(1 + np.exp(-np.dot(h2,w3)-np.tile(c3,(N,1)) ));  
        h4 = 1/(1+np.exp(-np.dot(h3,w4)-np.tile(c4,(N,1))));   
        dataout = 1/(1 + np.exp(-np.dot(h4,w_class)-np.tile(c_class,(N,1))));
        err= err +   ((target-dataout)**2 ).sum();
    train_err[epoch]=err
    print 'Before epoch %s Train squared error: %6.3s Test squared error:%6.3s\n'%(epoch,train_err[epoch],test_err[epoch])
    
    if epoch>50:
       alpha=0.05 
    for batch in range(numbatches):
        data = batchdata[:,:,batch];
        target = targetbatchdata[:,:,batch];
        #正向传播
        h1 = 1/(1 + np.exp(-np.dot(data,w1)- np.tile(c1,(N,1)) )); 
        dh1= h1*(1-h1); 
        h2 = 1/(1 + np.exp(-np.dot(h1,w2)-np.tile(c2,(N,1)) ));   
        dh2= h2*(1-h2); 
        h3 = 1/(1 + np.exp(-np.dot(h2,w3)-np.tile(c3,(N,1)) ));  
        dh3= h3*(1-h3); 
        h4 = 1/(1+np.exp(-np.dot(h3,w4)-np.tile(c4,(N,1))));   
        dh4= h4*(1-h4); 
        dataout = 1/(1 + np.exp(-np.dot(h4,w_class)-np.tile(c_class,(N,1))));
        Ddataout=dataout*(1-dataout);      
        err= target-dataout; 
        #负向传播
        ne_dataout=err*Ddataout;
        ne_h4=np.dot(ne_dataout,w_class.T)*dh4;
        ne_h3=np.dot(ne_h4,w4.T)*dh3;
        ne_h2=np.dot(ne_h3,w3.T)*dh2;
        ne_h1=np.dot(ne_h2,w2.T)*dh1;
        #更新权值
        w_class=w_class+alpha/N*np.dot(h4.T,ne_dataout) ;
        c_class=c_class+alpha/N*np.dot(np.ones((1,N)),ne_dataout);
        w4=w4+alpha/N*np.dot(h3.T,ne_h4) ;
        c4=c4+alpha/N*np.dot(np.ones((1,N)),ne_h4);
        w3=w3+alpha/N*np.dot(h2.T,ne_h3) ;
        c3=c3+alpha/N*np.dot(np.ones((1,N)),ne_h3);
        w2=w2+alpha/N*np.dot(h1.T,ne_h2) ;
        c2=c2+alpha/N*np.dot(np.ones((1,N)),ne_h2);
        w1=w1+alpha/N*np.dot(data.T,ne_h1) ;
        c1=c1+alpha/N*np.dot(np.ones((1,N)),ne_h1);
