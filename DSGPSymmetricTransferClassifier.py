#
# Source code for the STL-DGP model.
#
# Paper: M. Kandemir, Asymmetric Transfer Learning with Deep Gaussian Processes, ICML, 2015
#
# Contact: melihkandemir@gmail.com
#
# All rights reserved.
#
from MultiClassPrediction import MultiClassPrediction
from LinearKernel import LinearKernel
from RBFKernel import RBFKernel
import numpy as np
from scipy.stats import norm
import sklearn.cluster
from copy import deepcopy
import scipy

class DSGPSymmetricTransferClassifier:
    
    learned_model=0
    kernels=0
    inducing_kernel=0
    
    # model params
    R=-1      # num latent dimensions
    beta=0.5   # observation noise precision 
    P=20      # Number of inducing points
    
    max_iter=1
    Xind=0    # Observed inducing points
    
    # variational params
    Z=0      # inducing points, size = P x R
    m=0      # mean of q(u), size = P x 1
    S=0      # covariance of q(u), size = P x P

    A=0     # mean of q(X), size = N x R
    B=0     # variance of q(X), size = N x R
            
    def __init__(self,inducing_kernel,kernels_source, kernels_target, common_dims, num_inducing, max_iter, \
                 learnhyper=1, \
                 learn_sub_gps=1, \
                 learning_rate_start=1.):
        
        self.R=len(kernels_source)
       
        self.kernels=list ()
        
        self.kernels.append(kernels_target)
        
        self.kernels.append(kernels_source)
        
        self.common_dims = common_dims
        
        self.P = num_inducing
                
        self.inducing_kernel= list()
        
        self.inducing_kernel.append(deepcopy(inducing_kernel))
        self.inducing_kernel.append(deepcopy(inducing_kernel))
        
        self.max_iter = max_iter
        
        self.learnhyper = learnhyper
        
        self.learn_sub_gps=learn_sub_gps
        
        self.learning_rate_start=learning_rate_start
        
        
    def train(self,Xall,yall,sourceTargetInfo):

        #dist = scipy.spatial.distance.cdist(X,X,'euclidean')        

        X = list()
        y = list()

        X.append(Xall[sourceTargetInfo.ravel()==1,:])
        y.append(yall[sourceTargetInfo.ravel()==1])
                        
        X.append(Xall[sourceTargetInfo.ravel()==0,:])
        y.append(yall[sourceTargetInfo.ravel()==0])
                
        self.eps=0.0001 #0.1
 
        t = list() 
        for ww in range(2):
            Ntr = X[ww].shape[0]
            
            dist = scipy.spatial.distance.cdist(X[ww],X[ww],'euclidean')        
            
            for rr in range(len(self.kernels[ww])):
                
                self.kernels[ww][rr].length_scale = np.mean(dist)   
                
            #t = np.float64(y.ravel())
            self.T = np.unique(y[ww]).shape[0]
            
            t.append(np.zeros([Ntr, self.T]))
            
            for tt in range(self.T):
                t[ww][:,tt] = y[ww].ravel() == tt
                
       
        m = list(); S = list(); A = list(); B = list(); C = list()
        Vzz = list();  Vzz_inv = list(); EVzxVzxT_list = list();
        EVzxVzxT_this = list(); EVzx_this = list(); 
        Z = list(); self.Kzz_inv = list(); self.Xind=list(); self.Kzx = list();
        self.kxx = list(); self.kernel_list = list(); self.kpred = list()
        
        Lthis = np.zeros([2,1]).ravel()
        
        for ww in range(2):
            params=self.initialize(X[ww], y[ww], t[ww], self.kernels[ww],self.inducing_kernel[ww])
            
            m.append(params[0])
            S.append(params[1])
            A.append(params[2])
            B.append(params[3])
            C.append(params[4])
            Vzz.append(params[5])
            Vzz_inv.append(params[6])
            EVzxVzxT_list.append(params[7])
            EVzxVzxT_this.append(params[8])
            EVzx_this.append(params[9])
            Z.append(params[10])   
            self.Kzz_inv.append(params[11])
            self.Xind.append(params[12])
            self.Kzx.append(params[13])
            self.kxx.append(params[14])
            self.kernel_list.append(params[15])
            self.kpred.append(params[16])
            
            Lthis[ww]=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz[ww], \
                                      Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list[ww], \
                                      EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww])
              

        kpred_common = list()        
        Acommon = np.zeros([X[0].shape[0],self.common_dims*self.T])
        for rr in range(self.common_dims*self.T):
           
            kpred_common.append(self.kernel_list[1][rr].compute(self.Xind[1][rr],X[0]).T.dot(self.Kzz_inv[1][rr]))
            Acommon[:,rr]=kpred_common[rr].dot(C[1][:,rr])
            
        Zcommon = np.zeros([self.P,self.common_dims*self.T,self.T])    
        for tt in range(self.T):
            km=sklearn.cluster.KMeans(n_clusters=self.P,max_iter=400)        
            km.fit(Acommon)
            Zcommon[:,:,tt]=km.cluster_centers_            
                    
        Z[0] = np.concatenate( (Zcommon, Z[0] ), axis=1 )
        A[0] = np.concatenate( (Acommon, A[0] ), axis=1 )
        self.kpred[0] = kpred_common + self.kpred[0]
        self.kernel_list[0] = self.kernel_list[1][0:self.common_dims*self.T] + self.kernel_list[0]
        
        C[0] = np.concatenate( (C[1][:,0:self.common_dims*self.T], C[0]), axis=1 ) 
        self.Kzz_inv[0] = self.Kzz_inv[1][0:self.common_dims*self.T] + self.Kzz_inv[0]
        self.Xind[0] = self.Xind[1][0:self.common_dims*self.T] + self.Xind[0]
        
        B[0] = np.concatenate( (B[0],B[0][:,0:self.common_dims*self.T]), axis=1 ) 
        
        for tt in range(self.T):
            Vzz[0][:,:,tt] = self.inducing_kernel[0].compute(Z[0][:,:,tt],Z[0][:,:,tt])+np.identity(self.P)*self.eps
            for ii in range(20):
                Vzz[0][:,:,tt] = self.inducing_kernel[0].compute(Z[0][:,:,tt],Z[0][:,:,tt])+np.identity(self.P)*self.eps
                  
            
            Vzz_inv[0][:,:,tt] = np.linalg.inv(Vzz[0][:,:,tt])      
            
            EVzxVzxT_list[0][tt]=self.inducing_kernel[0].EVzxVzxT(Z[0][:,:,tt],A[0],B[0] )
            EVzxVzxT_this[0][tt] = np.sum(EVzxVzxT_list[0][tt],axis=0)
            EVzx_this[0][tt] = self.inducing_kernel[0].EVzx(Z[0][:,:,tt],A[0],B[0])        
        
        
        Tsign = np.zeros([X[0].shape[0], self.T])
        
        for tt in range(self.T):
            Tsign[:,tt]=-1+2*t[0][:,tt]*2
            
        for tt in range(self.T):
            Q=EVzx_this[0][tt].T.dot(Vzz_inv[0][:,:,tt])
            m_tt=np.linalg.lstsq(Q,Tsign[:,tt])       
            m[0][:,tt]=m_tt[0]    
            
        self.iscommon = list()
        self.iscommon.append(np.zeros([C[0].shape[1],1]).ravel())
        self.iscommon.append(np.zeros([C[1].shape[1],1]).ravel())        
        self.iscommon[0][0:self.common_dims*self.T] =1
        self.iscommon[1][0:self.common_dims*self.T] =1
                    
        learningrate_m=np.ones([2,1])*self.learning_rate_start
        learningrate_Z=np.ones([2,1])*self.learning_rate_start
        learningrate_hyper=np.ones([2,1])*self.learning_rate_start
        learningrate_C=np.ones([2,1])*self.learning_rate_start

          
        # variational iterations
        for ii in range(self.max_iter):
                      
           for ww in range(2):
               # Update q(u)
               mtry = m[ww].copy()
               for tt in range(self.T):
                   dM = self.gradLowerBound_by_m(X[ww],t[ww][:,tt],m[ww][:,tt],S[ww],Z[ww][:,:,tt],Vzz[ww][:,:,tt], \
                                                 Vzz_inv[ww][:,:,tt],A[ww],B[ww],EVzxVzxT_list[ww][tt], \
                                                 EVzxVzxT_this[ww][tt],EVzx_this[ww][tt])
        
                   mtry[:,tt]=m[ww][:,tt]+learningrate_m[ww]*dM
                                  
               L=self.lowerBound(X[ww],t[ww],mtry,S[ww],Z[ww],Vzz[ww],Vzz_inv[ww], \
                                 A[ww],B[ww],C[ww],EVzxVzxT_list[ww],EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww])
                   
               if L>Lthis[ww]:
                   m[ww]=mtry.copy()
               else:
                   learningrate_m[ww]=learningrate_m[ww]*0.9  
                       
              
               Lthis[ww]=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz[ww], \
                                          Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list[ww], \
                                          EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww])    
               print  "Iter: " + str(ii) + " ELBO after m:" + np.str(Lthis[ww])     
 
           
           for ww in range(2):
                   
               Ztry = np.zeros([self.P, Z[ww].shape[1], self.T])   
               Vzz_try = np.zeros([self.P, self.P, self.T])
               Vzz_inv_try = np.zeros([self.P, self.P, self.T])
               EVzxVzxT_list_try = deepcopy(EVzxVzxT_list[ww])
               EVzxVzxT_this_try = deepcopy(EVzxVzxT_this[ww])
               EVzx_this_try = deepcopy(EVzx_this[ww])
               
               for tt in range(self.T):
                   myFunct =  self.gradLowerBound_by_Z_closure(X[ww],t[ww][:,tt],m[ww][:,tt],S[ww],Z[ww][:,:,tt],Vzz[ww][:,:,tt],Vzz_inv[ww][:,:,tt], \
                                                               A[ww],B[ww],EVzxVzxT_list[ww][tt],EVzxVzxT_this[ww][tt],EVzx_this[ww][tt],self.inducing_kernel[ww])
                                                               
                   mapped = np.array(map(myFunct, range(self.P*Z[ww].shape[1])))  
                   dZ = np.reshape(mapped,[self.P,Z[ww].shape[1]]) 
                 
                   Ztry[:,:,tt]=Z[ww][:,:,tt]+learningrate_Z[ww]*dZ
               
                   EVzxVzxT_list_try[tt]=self.inducing_kernel[ww].EVzxVzxT(Ztry[:,:,tt],A[ww],B[ww])        
                   EVzxVzxT_this_try[tt]=np.sum(EVzxVzxT_list_try[tt],axis=0)
                   EVzx_this_try[tt]=self.inducing_kernel[ww].EVzx(Ztry[:,:,tt],A[ww],B[ww])   
                   Vzz_try[:,:,tt] = self.inducing_kernel[ww].compute(Ztry[:,:,tt],Ztry[:,:,tt])+np.identity(self.P)*0.0001
                   Vzz_inv_try[:,:,tt] = np.linalg.inv(Vzz_try[:,:,tt])                 
               
               L =self.lowerBound(X[ww],t[ww],m[ww],S[ww],Ztry,Vzz[ww], \
                                  Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list_try, \
                                  EVzxVzxT_this_try,EVzx_this_try,self.Kzz_inv[ww])
                   
               if L>Lthis[ww]:
                   Z[ww]=Ztry.copy()
                   EVzxVzxT_list[ww]=deepcopy(EVzxVzxT_list_try)
                   EVzxVzxT_this[ww]=deepcopy(EVzxVzxT_this_try)
                   EVzx_this[ww] = deepcopy(EVzx_this_try)
                   Vzz[ww]=Vzz_try.copy()
                   Vzz_inv[ww]=Vzz_inv_try.copy()
               else:
                   learningrate_Z[ww]=learningrate_Z[ww]*0.9     
                              
               Lthis[ww]=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz[ww], \
                                          Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list[ww], \
                                          EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww])   
                          
               print  "Iter: " + str(ii) + " ELBO after Z:" + np.str(Lthis[ww])  
           
           # Update the GP inducing outputs a_r! -----------------------------
           Ctry = deepcopy(C)
           Atry = deepcopy(A)
           EVzxVzxT_list_try = deepcopy(EVzxVzxT_list)
           EVzxVzxT_this_try = deepcopy(EVzxVzxT_this)
           EVzx_this_try = deepcopy(EVzx_this)                   
           
           for ww in range(2):
               
               myFunct2 =  self.gradLowerBound_by_C_closure(X,t,m,S, \
                                                            Z,Vzz,Vzz_inv, \
                                                            A,B,C,EVzxVzxT_list, \
                                                            EVzxVzxT_this,EVzx_this,self.Kzz_inv, self.kpred,self.inducing_kernel,ww)
                                                            
               mapped2 = np.array(map(myFunct2, range(self.P*Z[ww].shape[1])))  
               gC = np.reshape(mapped2,[self.P,Z[ww].shape[1]])                
                                        
#               for aa in range(5):
               Ctry[ww] = C[ww] + learningrate_C[ww]*gC
                       
              
               for rr in range(A[ww].shape[1]):
                   Atry[ww][:,rr]=self.kpred[ww][rr].dot(Ctry[ww][:,rr])
                   
               if ww == 1:
                   Ctry[0][:,0:self.common_dims*self.T] = Ctry[1][:,0:self.common_dims*self.T]
               
                   for rr in range(self.common_dims*self.T):
                       Atry[0][:,rr]=self.kpred[0][rr].dot(Ctry[0][:,rr])
                   
                   
               for tt in range(self.T):                   
                   EVzxVzxT_list_try[ww][tt]=self.inducing_kernel[ww].EVzxVzxT(Z[ww][:,:,tt],Atry[ww],B[ww])        
                   EVzxVzxT_this_try[ww][tt]=np.sum(EVzxVzxT_list_try[ww][tt],axis=0)
                   EVzx_this_try[ww][tt]=self.inducing_kernel[ww].EVzx(Z[ww][:,:,tt],Atry[ww],B[ww])            
                                  
        
           L =self.lowerBound(X[0],t[0],m[0],S[0],Z[0],Vzz[0], \
                              Vzz_inv[0],A[0],B[0],Ctry[0],EVzxVzxT_list_try[0], \
                              EVzxVzxT_this_try[0],EVzx_this_try[0],self.Kzz_inv[0])
                              
           L +=self.lowerBound(X[1],t[1],m[1],S[1],Z[1],Vzz[1], \
                              Vzz_inv[1],A[1],B[1],Ctry[1],EVzxVzxT_list_try[1], \
                              EVzxVzxT_this_try[1],EVzx_this_try[1],self.Kzz_inv[1])         
                              
           if L>Lthis[0]+Lthis[1]:
               A=deepcopy(Atry)
               C=deepcopy(Ctry)
               EVzxVzxT_list=deepcopy(EVzxVzxT_list_try)
               EVzxVzxT_this=deepcopy(EVzxVzxT_this_try)
               EVzx_this=deepcopy(EVzx_this_try)
               #break
           else:
               learningrate_C[0]=learningrate_C[0]*0.9
               learningrate_C[1]=learningrate_C[1]*0.9
       
           for ww in range(2):
               Lthis[ww]=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz[ww], \
                                          Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list[ww], \
                                          EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww]) 
               print "Iter: " + str(ii) + " ELBO after C:" + np.str(Lthis[ww])   
                   
           # Update the length scale!        
           if self.learnhyper==1 & self.inducing_kernel[0].num_hyperparams>0:
           
               for ww in range(2):    
                   gSigma = np.zeros([self.inducing_kernel[ww].num_hyperparams,1]).ravel()
                   for hh in range(self.inducing_kernel[ww].num_hyperparams):
                       gSigma[hh]=self.gradLowerBound_by_hyper(X[ww],t[ww],m[ww],S[ww], \
                                                               Z[ww],Vzz[ww],Vzz_inv[ww], \
                                                               A[ww],B[ww],EVzxVzxT_list[ww],
                                                               EVzxVzxT_this[ww],EVzx_this[ww],self.inducing_kernel[ww],hh)                                                        
                   
                   trykernel=self.inducing_kernel[ww].clone()
                   
                   #for aa in range(5):
                   trykernel.length_scale = self.inducing_kernel[ww].length_scale+learningrate_hyper[ww]*gSigma
                   
                   for tt in range(self.T):                   
                       EVzxVzxT_list_try[ww][tt]=trykernel.EVzxVzxT(Z[ww][:,:,tt],A[ww],B[ww])        
                       EVzxVzxT_this_try[ww][tt]=np.sum(EVzxVzxT_list_try[ww][tt],axis=0)
                       EVzx_this_try[ww][tt]=trykernel.EVzx(Z[ww][:,:,tt],A[ww],B[ww])   
                       Vzz_try[:,:,tt] = trykernel.selfCompute(Z[ww][:,:,tt])+np.identity(self.P)*self.eps
                       Vzz_inv_try[:,:,tt] = np.linalg.inv(Vzz_try[:,:,tt])                 
                       
                   L=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz_try, \
                                     Vzz_inv_try,A[ww],B[ww],C[ww],EVzxVzxT_list_try[ww], \
                                     EVzxVzxT_this_try[ww],EVzx_this_try[ww],self.Kzz_inv[ww])                    
                   if L>Lthis[ww]:
                       self.inducing_kernel[ww].length_scale=trykernel.length_scale
                       EVzxVzxT_list[ww]=deepcopy(EVzxVzxT_list_try[ww])     
                       EVzxVzxT_this[ww]=deepcopy(EVzxVzxT_this_try[ww])
                       EVzx_this[ww]=deepcopy(EVzx_this_try[ww])                             
                       Vzz[ww] = Vzz_try.copy()
                       Vzz_inv[ww] = Vzz_inv_try.copy()
                       #break
                   else:
                       learningrate_hyper[ww]=learningrate_hyper[ww]*0.9       
                   
                   Lthis[ww]=self.lowerBound(X[ww],t[ww],m[ww],S[ww],Z[ww],Vzz[ww], \
                                              Vzz_inv[ww],A[ww],B[ww],C[ww],EVzxVzxT_list[ww], \
                                              EVzxVzxT_this[ww],EVzx_this[ww],self.Kzz_inv[ww]) 
                              
                   print "Iter: " + str(ii) + " ELBO after H:" + np.str(Lthis)  + " Length scale : " + np.str(self.inducing_kernel[ww].length_scale)
                                
        #Vzz = self.inducing_kernel.compute(Z,Z)+np.identity(self.P)*eps
        #Vzz_inv = np.linalg.inv(Vzz)
        self.m = m
        self.S = S
        self.Z = Z
        self.A = A
        self.B = B     
        self.C = C
        self.Vzz_inv=Vzz_inv
        self.Vzz=Vzz
        
    def gradLowerBound_by_Z_closure(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,inducing_kernel):               
        
        def funct(idx):
            return self.gradLowerBound_by_Z(X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,inducing_kernel,idx)
            
        return funct 
        

    def gradLowerBound_by_C_closure(self,X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,Kzz_inv,kpred,inducing_kernel,ww):               
        
        def funct(idx):
            return self.gradLowerBound_by_c(X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,Kzz_inv,kpred,inducing_kernel,ww,idx)
            
        return funct          
          
    def lowerBound(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,Kzz_inv):
        val = 0
        Ntr = np.float64(X.shape[0])
        
        for tt in range(self.T):
            
            t_this=t[:,tt]*1
            m_this=m[:,tt]*1
            
            sqBetaPlusOne=np.sqrt(1/self.beta+1)
            
            mmtS = np.outer(m_this,m_this)+S
            Minterm = Vzz_inv[:,:,tt].dot(EVzxVzxT_this[tt]).dot(Vzz_inv[:,:,tt])
            
            phi = EVzx_this[tt].T.dot(Vzz_inv[:,:,tt]).dot(m_this) / sqBetaPlusOne
            
            M2=m_this.dot(Vzz_inv[:,:,tt]).dot(EVzx_this[tt])
            
            normCdfPhi=norm.cdf(phi)
            
            smallElements=normCdfPhi < 0.000001
            
            normCdfPhi[smallElements]=1.0
            
            logNormCdfPhi=np.log(normCdfPhi)    
            
            likTerm1 =  0.5*self.beta*np.power(M2,2) + logNormCdfPhi 
            likTerm0 = np.log(np.sqrt(2*np.pi/self.beta))- logNormCdfPhi
            
            tinv=np.abs(t_this-1)
                
            (temp,logdetS)=np.linalg.slogdet(S)
            (temp,logdetVzz)=np.linalg.slogdet(Vzz[:,:,tt])
            
            val += 0.5*Ntr*np.log(self.beta) \
                  -0.5*logdetVzz \
                  +0.5*logdetS \
                  -0.5*m_this.dot(Vzz_inv[:,:,tt]).dot(m_this) \
                  -0.5*Vzz_inv[:,:,tt].dot(S).trace() \
                  +0.5*self.beta*Vzz_inv[:,:,tt].dot(EVzxVzxT_this[tt]).trace() \
                  -0.5*self.beta*Minterm.dot(mmtS).trace()\
                  + (likTerm1*t_this).sum() + (likTerm0*tinv).sum()
                  
        for rr in range(C.shape[1]):
            val += -0.5*C[:,rr].dot(Kzz_inv[rr]).dot(C[:,rr])
            
        return val
        
    def gradLowerBound_by_m(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):

        P=Z.shape[0]
        val = 0
        
        for tt in range(self.T):
            val += -self.beta*Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv).dot(m)-Vzz_inv.dot(m)
            
            ttiled=np.tile(t,[P,1])
            tinv_tiled=np.tile(np.abs(t-1),[P,1]) 
            sqBetaPlusOne=np.sqrt(1/self.beta+1)  
            
            Mm2= EVzx_this.T.dot(Vzz_inv).dot(m)  
            Mm2tiled =  np.tile(Mm2,[P,1])
            phiTiled= norm.cdf(Mm2tiled/sqBetaPlusOne)
            dLogPhi= norm.pdf(Mm2tiled/sqBetaPlusOne,0,1)*(1/phiTiled)*Vzz_inv.dot(EVzx_this)/sqBetaPlusOne 
            
            gradLik1 = self.beta*Mm2tiled*Vzz_inv.dot(EVzx_this) + dLogPhi
            gradLik0 = -dLogPhi
            
            val += (gradLik1*ttiled).sum(axis=1) + (gradLik0*tinv_tiled).sum(axis=1)
       
        return val  
        
    def gradLowerBound_by_S(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this):

        Sinv=np.linalg.inv(S)+np.identity(Z.shape[0])*0.0001
        val = -0.5*Vzz_inv + 0.5*Sinv.T - 0.5*self.beta*Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
               
        return val
        
    def gradLowerBound_by_Z(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,inducing_kernel,idx):
 
        xx=idx/Z.shape[1]       
        yy=np.mod(idx,Z.shape[1])
        
        tinv=np.abs(t-1)        
        
        # global terms  
        grad_Vzz=inducing_kernel.grad_K_by_Z(Vzz,Z,xx,yy)        
        grad_Vzz_inv = -Vzz_inv.dot(grad_Vzz).dot(Vzz_inv)
        grad_EVzxVzxT_by_Z_this=inducing_kernel.grad_EVzxVzxT_by_Z(EVzxVzxT_list,Z,A,B,xx,yy)        
        grad_EVzx=inducing_kernel.grad_EVzx_by_Z(EVzx_this,Z,A,B,xx,yy)
        
        mmtS = np.outer(m,m)+S  

        sqBetaPlusOne=np.sqrt(1/self.beta+1)              
                
        Mm1= grad_EVzx.T.dot(Vzz_inv).dot(m)+EVzx_this.T.dot(grad_Vzz_inv).dot(m)
        Mm2= EVzx_this.T.dot(Vzz_inv).dot(m)   
        dLogPhi= norm.pdf(Mm2/sqBetaPlusOne)*(1/norm.cdf(Mm2/sqBetaPlusOne))*Mm1/sqBetaPlusOne
        
        gradLik1=self.beta*Mm2*Mm1+dLogPhi
        gradLik0=-dLogPhi
        
        T1 = grad_Vzz_inv.dot(EVzxVzxT_this).dot(Vzz_inv)
        T2 = Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).dot(Vzz_inv)
        T3 = Vzz_inv.dot(EVzxVzxT_this).dot(grad_Vzz_inv)
        Ttot = T1+T2+T3 
        
        val = -0.5*Vzz_inv.dot(grad_Vzz).trace() \
               -0.5*m.dot(grad_Vzz_inv).dot(m) \
               -0.5*grad_Vzz_inv.dot(S).trace() \
               +0.5*self.beta*(grad_Vzz_inv.dot(EVzxVzxT_this).trace() \
               +Vzz_inv.dot(grad_EVzxVzxT_by_Z_this).trace()) \
               -0.5*self.beta*Ttot.dot(mmtS).trace() \
               +(gradLik1*t).sum() + (gradLik0*tinv).sum()
 
        return val  
        
    # Actually the same as the above function!    
    def gradLowerBound_by_hyper(self,X,t,m,S,Z,Vzz,Vzz_inv,A,B,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,inducing_kernel,hyperno):
        val=0 


        for tt in range(self.T):
            grad_Vzz=inducing_kernel.grad_K_by_hyper(Vzz[:,:,tt],Z[:,:,tt],hyperno)       
            grad_Vzz_inv = -Vzz_inv[:,:,tt].dot(grad_Vzz).dot(Vzz_inv[:,:,tt]) 
            grad_EVzxVzxT_by_Z_this=inducing_kernel.grad_EVzxVzxT_by_hyper(EVzxVzxT_list[tt],Z[:,:,tt],A,B,hyperno)          
            grad_EVzx=inducing_kernel.grad_EVzx_by_hyper(EVzx_this[tt],Z[:,:,tt],A,B,hyperno) 
            
            tinv=np.abs(t[:,tt]-1)        
            
            mmtS = np.outer(m[:,tt],m[:,tt])+S  
    
            sqBetaPlusOne=np.sqrt(1/self.beta+1)              
                    
            Mm1= grad_EVzx.T.dot(Vzz_inv[:,:,tt]).dot(m[:,tt])+EVzx_this[tt].T.dot(grad_Vzz_inv).dot(m[:,tt])
            Mm2= EVzx_this[tt].T.dot(Vzz_inv[:,:,tt]).dot(m[:,tt])   
            dLogPhi= norm.pdf(Mm2/sqBetaPlusOne)*(1/norm.cdf(Mm2/sqBetaPlusOne))*Mm1/sqBetaPlusOne
            
            gradLik1=self.beta*Mm2*Mm1+dLogPhi
            gradLik0=-dLogPhi
            
            T1 = grad_Vzz_inv.dot(EVzxVzxT_this[tt]).dot(Vzz_inv[:,:,tt])
            T2 = Vzz_inv[:,:,tt].dot(grad_EVzxVzxT_by_Z_this).dot(Vzz_inv[:,:,tt])
            T3 = Vzz_inv[:,:,tt].dot(EVzxVzxT_this[tt]).dot(grad_Vzz_inv)
            Ttot = T1+T2+T3 
            
            val += -0.5*Vzz_inv[:,:,tt].dot(grad_Vzz).trace()
            val += -0.5*m[:,tt].dot(grad_Vzz_inv).dot(m[:,tt])
            val += -0.5*grad_Vzz_inv.dot(S).trace()
            val += 0.5*self.beta*(grad_Vzz_inv.dot(EVzxVzxT_this[tt]).trace() + Vzz_inv[:,:,tt].dot(grad_EVzxVzxT_by_Z_this).trace())
            val += -0.5*self.beta*Ttot.dot(mmtS).trace()
            val += (gradLik1*t[:,tt]).sum() + (gradLik0*tinv).sum()
 
        return val 
        
    def gradLowerBound_by_c(self,X,y,m,S,Z,Vzz,Vzz_inv,A,B,C,EVzxVzxT_list,EVzxVzxT_this,EVzx_this,Kzz_inv,kpred,inducing_kernel,ww,idx):
        xx=idx/C[ww].shape[1]       
        yy=np.mod(idx,C[ww].shape[1])
        
        if ww == 0 & self.iscommon[ww][yy].astype(int) == 1:
            val = 0
            return val


                
        val = (-Kzz_inv[ww][yy].dot(C[ww][:,yy]))[xx]
        
        for tt in range(self.T):
            
            mmtS = np.outer(m[ww][:,tt],m[ww][:,tt])+S[ww]              
    
            grad_EVzx=inducing_kernel[ww].grad_EVzx_by_c(EVzx_this[ww][tt],Z[ww][:,:,tt],A[ww],B[ww],C[ww],kpred[ww],xx,yy)
               
            grad_EVzxVzxT=inducing_kernel[ww].grad_EVzxVzxT_by_c(EVzxVzxT_list[ww][tt],Z[ww][:,:,tt],A[ww],B[ww],C[ww],kpred[ww],xx,yy)
            
            val += self.beta*y[ww][:,tt].dot(grad_EVzx.T).dot(Vzz_inv[ww][:,:,tt]).dot(m[ww][:,tt]) \
                  -0.5*self.beta*Vzz_inv[ww][:,:,tt].dot(grad_EVzxVzxT).dot(Vzz_inv[ww][:,:,tt]).dot(mmtS).trace()\
                  +0.5*self.beta*Vzz_inv[ww][:,:,tt].dot(grad_EVzxVzxT).trace()\
                  -0.5*self.beta*inducing_kernel[ww].grad_EVxx_by_c(kpred[ww],A[ww],B[ww],C[ww],xx,yy)
                
                
        if ww == 1 & self.iscommon[ww][yy].astype(int) == 1:
            mmtS = np.outer(m[0][:,tt],m[0][:,tt])+S[0]              
    
            grad_EVzx=inducing_kernel[0].grad_EVzx_by_c(EVzx_this[0][tt],Z[0][:,:,tt],A[0],B[0],C[0],kpred[0],xx,yy)
               
            grad_EVzxVzxT=inducing_kernel[0].grad_EVzxVzxT_by_c(EVzxVzxT_list[0][tt],Z[0][:,:,tt],A[0],B[0],C[0],kpred[0],xx,yy)
            
            val += self.beta*y[0][:,tt].dot(grad_EVzx.T).dot(Vzz_inv[0][:,:,tt]).dot(m[0][:,tt]) \
                  -0.5*self.beta*Vzz_inv[0][:,:,tt].dot(grad_EVzxVzxT).dot(Vzz_inv[0][:,:,tt]).dot(mmtS).trace()\
                  +0.5*self.beta*Vzz_inv[0][:,:,tt].dot(grad_EVzxVzxT).trace()\
                  -0.5*self.beta*inducing_kernel[0].grad_EVxx_by_c(kpred[0],A[0],B[0],C[0],xx,yy)
                  
        return val            
        
      
    def initialize(self,X,y,t,kernels,inducing_kernel):

        Ntr = X.shape[0]

        Tsign = np.zeros([Ntr, self.T])
        
        for tt in range(self.T):
            Tsign[:,tt]=-1+2*t[:,tt] * 2
        
        
            
        Xind = list()  # choose this more carefully (e.g. k-means)
        yind = list() 
        Kzz_inv=list()
        Kzx=list()
        kxx=list()

        Tsubset = np.zeros([self.P, self.T])        

        kernel_list=list()            
          
        cnt = 0

        for rr in range(len(kernels)):        
            for tt in range(self.T):
        
                chosen = np.zeros([self.P, 1]).ravel()
                
                idx = np.where(y==tt)                
                permInd = np.random.permutation(len(idx[0]))                
                chosen[0:3] = idx[0][permInd[0:3]] 
                
                idx = np.where(y!=tt)
                permInd = np.random.permutation(len(idx[0]))              
                chosen[3:self.P] = idx[0][permInd[0:(self.P-3)]]
                
                chosen = chosen.astype(int)
                
                Tsubset[:,tt] = Tsign[chosen,tt]
             
                Xind.append(X[chosen,:])
                Kzz=kernels[rr].compute(Xind[cnt],Xind[cnt]) 
                Kzz_inv.append(np.linalg.inv(Kzz+np.identity(self.P)*0.001))    
                Kzx.append(kernels[rr].compute(Xind[cnt],X))            
                kxx.append(kernels[rr].computeSelfDistance(X))
                Q=Kzx[cnt].T.dot(Kzz_inv[cnt])
                yind_rr=np.linalg.lstsq(Q,Tsign[:,tt])
                yind_rr=yind_rr[0]
                yind.append(yind_rr)
                kernel_list.append(kernels[rr])
                
                cnt += 1

        C=np.zeros([self.P,len(kernel_list)])        
        Z=np.zeros([self.P,len(kernel_list),self.T])
        A=np.zeros([Ntr,len(kernel_list)])
        
        kpred=list()
        for rr in range(len(kernel_list)):
            kpred.append(kernel_list[rr].compute(Xind[rr],X).T.dot(Kzz_inv[rr]))
            A[:,rr]=kpred[rr].dot(yind[rr])
            C[:,rr]=yind[rr]                   

        for tt in range(self.T):
            km=sklearn.cluster.KMeans(n_clusters=self.P,max_iter=400)        
            km.fit(A)
            Z[:,:,tt]=km.cluster_centers_


        S = np.identity(self.P)*0.0000001   
        
        B=np.ones([Ntr, len(kernel_list)])*0.01  
               
        Vzz = np.zeros([self.P, self.P, self.T])
        Vzz_inv = np.zeros([self.P, self.P, self.T])
        EVzxVzxT_list=list()      
        EVzxVzxT_this=list()
        EVzx_this=list()
            
        for tt in range(self.T):
            Vzz[:,:,tt] = inducing_kernel.compute(Z[:,:,tt],Z[:,:,tt])+np.identity(self.P)*self.eps
            if  isinstance(inducing_kernel,LinearKernel) == 1:
                inducing_kernel.scale= np.max(Vzz[:,:,tt])            
                Vzz[:,:,tt] /= inducing_kernel.scale      
            elif  isinstance(inducing_kernel,RBFKernel) == 1: 
                for ii in range(20):
                  if np.mean(Vzz[:,:,tt]) > 0.8:
                     inducing_kernel.length_scale *= 0.5
                     Vzz[:,:,tt] = inducing_kernel.compute(Z[:,:,tt],Z[:,:,tt])+np.identity(self.P)*self.eps
                     print "Length scale updated to: " + str(inducing_kernel.length_scale) 
                  elif np.mean(Vzz[:,:,tt]) < 0.2:
                     inducing_kernel.length_scale *= 2.
                     Vzz[:,:,tt] = inducing_kernel.compute(Z[:,:,tt],Z[:,:,tt])+np.identity(self.P)*self.eps                 
                     print "Length scale updated to: " + str(inducing_kernel.length_scale) 
                  else:
                     break
            
            Vzz_inv[:,:,tt] = np.linalg.inv(Vzz[:,:,tt])      
            
            EVzxVzxT_list.append(inducing_kernel.EVzxVzxT(Z[:,:,tt],A,B) )
            EVzxVzxT_this.append(np.sum(EVzxVzxT_list[tt],axis=0)  )
            EVzx_this.append(inducing_kernel.EVzx(Z[:,:,tt],A,B) )        
        
        m = np.zeros([self.P, self.T])
        for tt in range(self.T):
            Q=EVzx_this[tt].T.dot(Vzz_inv[:,:,tt])
            m_tt=np.linalg.lstsq(Q,Tsign[:,tt])       
            m[:,tt]=m_tt[0] 
            
        params = list()
        params.append(m)
        params.append(S)
        params.append(A)
        params.append(B)        
        params.append(C)
        params.append(Vzz)
        params.append(Vzz_inv)
        params.append(EVzxVzxT_list)
        params.append(EVzxVzxT_this)
        params.append(EVzx_this)
        params.append(Z)        
        params.append(Kzz_inv)
        params.append(Xind)
        params.append(Kzx)
        params.append(kxx)
        params.append(kernel_list)
        params.append(kpred)
        
        return params
        
    def predict(self,Xts):
      
      Xtest = np.zeros([Xts.shape[0],self.C[0].shape[1]])
      
      for rr in range(len(self.kernel_list[0])):
          Kzx=self.kernel_list[0][rr].compute(self.Xind[0][rr],Xts)
          Xtest[:,rr] = Kzx.T.dot(self.Kzz_inv[0][rr]).dot(self.C[0][:,rr])
                
                
      probabilities = np.zeros([Xtest.shape[0], self.T])
      
      for tt in range(self.T):
          Vzx = self.inducing_kernel[0].compute(self.Z[0][:,:,tt],Xtest)
          
          probabilities[:,tt] = norm.cdf(Vzx.T.dot(self.Vzz_inv[0][:,:,tt]).dot(self.m[0][:,tt]))
          #probabilities = norm.cdf(Xtest[:,2])
          #classes[:,tt] = (probabilities[:,tt]>0.5)*1
          
      classes=np.argmax(probabilities,axis=1)
        
      class_prediction=MultiClassPrediction(classes,probabilities)        
      
      return class_prediction       
