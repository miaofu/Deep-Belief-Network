# -*- coding: cp936 -*-
import datetime
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def useRBM(batchdata,W,c,b):
    (numcase,numdims,numbatches)=batchdata.shape
    y= np.random.rand(numcase,len(c),numbatches)
    for batch in range(numbatches):
        v_init=batchdata[:,:,batch].T
        h_now=  sigmoid(np.dot(W,v_init)+np.tile(c,(1,numcase)) )  #%sample(i,v(:,t));
        y[:,:,batch]=h_now.T
    return y
def trainRBM(vis_dim,hid_dim,k,batchdata,interation):
    '''
    %author:miaofu, in 2015/7/21.
    %training binaryRBM using k-step contrastive divergence
    %Input RBM(V1,,,Vm and H1,,,Hn) training batch S,S(:,s)to denote the sth sample 
    %Output gradient approximation Wij bj ci for i=1,2,,,n j=1,2,..m
    rewritten by Fmiao,at 2015/12/6 by Python2.7
    '''
    print datetime.datetime.now()
    print ('training binaryRBM using k-step contrastive divergence');
    W=0.1*np.random.randn(hid_dim,vis_dim);
    b=-4*np.zeros((vis_dim,1));
    c=-4*np.zeros((hid_dim,1));
    alpha0=0.1;
    weightcost  = 0.0002;
    MaxIteration=interation; 
    (numcase,numdims,numbatches)=batchdata.shape
    
    for iteration in range(MaxIteration):
      errsum=0
      for batch in range(numbatches):
          v_init=batchdata[:,:,batch].T     #表示初始的样本  size:numdims,numcase,
          v_now=v_init
          '''
          h_now=[]
          h_new=[]
          v_new=[]
          '''
          for t in range(k):
              h_now= np.random.rand(hid_dim,numcase) < sigmoid(np.dot(W,v_now)+np.tile(c,(1,numcase)) )  #%sample(i,v(:,t));
              if t==k: 
                  v_new= sigmoid( np.dot(W.T,h_now)+np.tile(b,(1,numcase)) ) #;%sample(j,h(:,t));%;
              else:
                  v_new= np.random.rand(vis_dim,numcase)<sigmoid( np.dot(W.T,h_now)+np.tile(b,(1,numcase)) ) #%sample(j,h(:,t));%;
              v_now=v_new
          err=((v_now-v_init)**2).sum().sum()
          errsum=errsum+err
          alpha=alpha0 #%*exp(-iteration*s/sigma);
          hid_pro0=sigmoid( np.dot(W,v_init)+np.tile(c,(1,numcase)) ) #;%size:hid_dim*numcase
          hid_proNow=sigmoid( np.dot(W,v_now)+np.tile(c,(1,numcase)) )
          W_new=W+alpha*(   ( np.dot(hid_pro0,v_init.T)-np.dot(hid_proNow,v_now.T) )/numcase- weightcost*W  )#;%
          c_new=c+alpha*(  hid_pro0-hid_proNow).mean()  
          b_new=b+alpha*( v_init-v_now).mean()
          W=W_new;
          c=c_new;
          b=b_new;
      print 'epoch %4s error %6.1f  \n'%(iteration, errsum)
    print 'training over'
    print datetime.datetime.now()
    #sio.savemat(weightname,{'W':W,'c':c,'b':b})
    return W,c,b  

