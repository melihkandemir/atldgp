#
# Radial basis function (RBF) kernel class.
#
# Contact: melihkandemir@gmail.com
#
# All rights reserved.
#
from Kernel import Kernel
import numpy as np
import copy
import scipy

#
# exp(-0.5 ||x1-x2||_2^2 / (2*sqrt(length_scale)))
#
class RBFKernel(Kernel):
    
    length_scale=1.0
    featuremap=None
    
    def __init__(self,length_scale,featuremap=None):        
        
        self.length_scale=np.float64(length_scale)
        self.num_hyperparams=1;
        self.featuremap=featuremap
        
    def clone(self):
        
        newinstance=copy.deepcopy(self)
        
        return newinstance
        
    def compute(self,X,Y):
        
        #if self.featuremap is not None:
        #    X=X[:,self.featuremap[0]:self.featuremap[1]]
        #    Y=Y[:,self.featuremap[0]:self.featuremap[1]]        

        sqdist=scipy.spatial.distance.cdist(X,Y,'euclidean')
        sqdist=sqdist*sqdist
        
        return np.exp(sqdist*(-1/(2*self.length_scale*self.length_scale))) 
        
    def selfCompute(self,X):
        
        #if self.featuremap is not None:
        #    X=X[:,self.featuremap[0]:self.featuremap[1]]
        
        return self.compute(X,X)
        
    def computeSelfDistance(self,X):
        
       # if self.featuremap is not None:
       #     X=X[:,self.featuremap[0]:self.featuremap[1]]
        
        return np.ones([X.shape[0],1]).ravel()
                
    
    def grad_EVzx_by_Z(self,EVzx_this,Z,A,B,p,r):
        
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
        
        dZ=np.zeros([P,R])
        dZ[p,r]=1;        
               
        alpha=self.length_scale*self.length_scale
        S=B[0,0]*B[0,0]
        
        res=np.zeros([P,P])
        
        Sinv = 1/(1/alpha+1/S)
        
        ZZt=Z.dot(dZ.T) + dZ.dot(Z.T)
        
        E1=(-0.5*(1/alpha)+0.5*(1/alpha)*(1/alpha)*(Sinv))*np.tile(ZZt.diagonal(),[N,1]).T
        E3=(1/alpha)*(1/S)*(Sinv)*dZ.dot(A.T)
                     
        res=EVzx_this*(E1+E3)
                                    
        return res     
        
    def grad_EVzx_by_hyper(self,EVzx_this,Z,A,B,hyperno):
        
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
        
        if hyperno != 0 :
            return EVzx_this*0
        
        alpha=self.length_scale*self.length_scale
        
        I=np.identity(R)
        S=np.diag(B[0,:]*B[0,:])
        Sinv=np.diag(1./(B[0,:]*B[0,:]))
        C=I*alpha
        Cinv=I*(1./alpha)
        CinvSinv=Cinv+Sinv       
        CinvSinv_inv=np.diag(1./CinvSinv.diagonal())
        
        dC=2*self.length_scale*I
        dCinv=-Cinv.dot(dC).dot(Cinv)
        dCinvSinv=dCinv
        dCinvSinv_inv=-CinvSinv_inv.dot(dCinvSinv).dot(CinvSinv_inv)
        
        S1=dCinv-dCinv.dot(CinvSinv_inv).dot(Cinv)-Cinv.dot(dCinvSinv_inv).dot(Cinv)-Cinv.dot(CinvSinv_inv).dot(dCinv)
        S2=-Sinv.dot(dCinvSinv_inv).dot(Sinv)
        S3=Sinv.dot(dCinvSinv_inv).dot(Cinv)+Sinv.dot(CinvSinv_inv).dot(dCinv)
        
        T1=np.tile(Z.dot(S1).dot(Z.T).diagonal(),[N,1]).T
        T2=np.tile(A.dot(S2).dot(A.T).diagonal(),[P,1])
        T3=A.dot(S3).dot(Z.T).T
        
        SCinvI=Cinv*S+I
        SCinvI_inv=np.diag(1./SCinvI.diagonal())
        (temp,logDetSCinvI)=np.linalg.slogdet(SCinvI)
        detSCinvI=np.exp(logDetSCinvI)
        dDetSCinvI=-0.5*np.power(detSCinvI,-0.5)*SCinvI_inv.dot(dCinv).dot(S).trace()
        
        expTerm=EVzx_this/np.power(detSCinvI,-0.5)
        
        res=EVzx_this*(-0.5*T1-0.5*T2+T3)+dDetSCinvI*expTerm
                                    
        return res     
        
    def grad_EVzxVzxT_by_hyper(self,EVzxVzxT_list_this,Z,A,B,hyperno):
                 
        EVzxVzxT_list=EVzxVzxT_list_this
        
        newkernel=self.clone()
        newkernel.length_scale += 0.00000001
        
        EVzxVzxT_list_diff=newkernel.EVzxVzxT(Z,A,B)
                                
        EVzxVzxT_this=np.sum(EVzxVzxT_list,axis=0)
        EVzxVzxT_this_diff=np.sum(EVzxVzxT_list_diff,axis=0)
                                    
        return (EVzxVzxT_this_diff-EVzxVzxT_this)/0.00000001
#
    def grad_EVzxVzxT_by_hyper_exact(self,EVzxVzxT_list_this,Z,A,B,hyperno):
        
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
        
        if hyperno != 0 :
            return EVzxVzxT_list_this*0
        
        alpha=self.length_scale*self.length_scale
        
        I=np.identity(R)
        S=np.diag(B[0,:]*B[0,:])
        Sinv=np.diag(1/B[0,:]*B[0,:])
        C=I*alpha
        Cinv=I*(1/alpha)
        CinvSinv=2*Cinv+Sinv       
        CinvSinv_inv=np.diag(1/CinvSinv.diagonal())
        
        dC=self.length_scale*I
        dCinv=-Cinv.dot(dC).dot(Cinv)
        dCinvSinv=2*dCinv
        dCinvSinv_inv=-CinvSinv_inv.dot(dCinvSinv).dot(CinvSinv_inv)
        
        S1=dCinv-dCinv.dot(CinvSinv_inv).dot(Cinv)-Cinv.dot(dCinvSinv_inv).dot(Cinv)-Cinv.dot(CinvSinv_inv).dot(dCinv)
        S2=-Sinv.dot(dCinvSinv_inv).dot(Sinv)
        S3=Sinv.dot(dCinvSinv_inv).dot(Cinv)+Sinv.dot(CinvSinv_inv).dot(dCinv)
        S4=dCinv.dot(CinvSinv_inv).dot(Cinv)+Cinv.dot(dCinvSinv_inv).dot(Cinv)+Cinv.dot(CinvSinv_inv).dot(dCinv)
        
        T1s=np.tile(Z.dot(S1).dot(Z.T).diagonal(),[P,1])
        T1=np.tile(T1s,[N,1,1])
        T2s=T1s.T
        T2=np.tile(T2s,[N,1,1])        
        T3=np.tile(Z.dot(S4).dot(Z.T),[N,1,1])
        T4=np.tile(A.dot(S2).dot(A.T).diagonal(),[P,1]).T
        T4=np.expand_dims(T4,axis=2)
        T4=np.repeat(T4,P,axis=2)
        T5=A.dot(S3).dot(Z.T)
        T5=np.expand_dims(T5,axis=2)
        T5=np.repeat(T5,P,axis=2)
        T6=np.swapaxes(T5,1,2)
        
        SCinvI=2*Cinv.dot(S)+I
        SCinvI_inv=np.diag(1/SCinvI.diagonal())
        (temp,logDetSCinvI)=np.linalg.slogdet(SCinvI)
        detSCinvI=np.exp(logDetSCinvI)
        dDetSCinvI=-0.5*np.power(detSCinvI,-0.5)*SCinvI_inv.dot(2*dCinv).dot(S).trace()
        
        expTerm=EVzxVzxT_list_this/np.power(detSCinvI,-0.5)
        
        res=EVzxVzxT_list_this*(-0.5*T1-0.5*T2+T3-0.5*T4+T5+T6)+dDetSCinvI*expTerm
        
        res=np.sum(res,axis=0)
                                    
        return res     
         
    def grad_EVxx_by_Z(self,Z,A,p,r):
        
        return 0.0
        
    def grad_EVzxVzxT_by_Z(self,EVzxVzxT_list_this,Z,A,B,p,r):
        
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
        
        ainv=1/(self.length_scale*self.length_scale)
        siginv=1/(B[0,0]*B[0,0])        
      
        dZthis=np.zeros([1,R])
        
        dZthis[0,r]=1        
        
        res1 = -0.5*(dZthis.dot(Z[p,:])+Z[p,:].dot(dZthis.T))*(ainv-ainv*(1/(siginv+2*ainv))*ainv)
        
        res2 = np.tile(dZthis.dot(A.T)*(ainv*(1/(siginv+2*ainv))*siginv),[P,1])
       
        res3 = np.tile(dZthis.dot(Z.T)*(ainv*(1/(siginv+2*ainv))*ainv),[N,1])
        
        dZ=np.zeros([N,P,P])
        
        dZ[:,p,:] += np.float64(res1)+res2.T+res3
        dZ[:,:,p] += np.float64(res1)+res2.T+res3
        
        # set the diagonal
        #dZ[:,p,p] = dZ[:,p,p]/2.
        
        res=np.sum(EVzxVzxT_list_this*dZ,axis=0)
        
        return res
        
    def grad_K_by_hyper(self,K,Z,hyperno):
        
        if hyperno != 0 :
            return K*0                  
                
        gamma=pow(self.length_scale,-3)
        
        XXdist = np.tile(Z.dot(Z.T).diagonal(),[Z.shape[0],1])
        XYdist = Z.dot(Z.T)               
        
        sqdist = XXdist.T - 2 * XYdist +XXdist
        
        res = K*sqdist*gamma        
        
        return res   
                
    def grad_K_by_Z(self,K,Z,p,r):
        P=Z.shape[0]

        
        gamma=1/(self.length_scale*self.length_scale)
        
        res=np.zeros([P,P])
        
        entry=-gamma*Z[p,r] + gamma*Z[:,r]
        
        res[:,p]=entry
        res[p,:]=entry
        res[p,p]=0
        
        res=res*K
        
        return res
        
    def grad_K_inv_by_Z(self,K,K_inv,Z,p,r):
        gVzz = self.grad_K_by_Z(K,Z,p,r)
    
        res = -K_inv.dot(gVzz).dot(K_inv)
        
        return res

 
    def grad_Kzx_by_Z(self,Kzx,Z,X,p,r):
        
        gamma=1/(self.length_scale*self.length_scale)
        
        res=np.zeros([Z.shape[0],X.shape[0]])
        
        res[p,:] = gamma* Kzx[p,:] * (X[:,r] - Z[p,r])        
        
        return res
                  

    # -------------------------------------------------------------------------
    # Expectation wrt a normal distribution
    # -----------------------------------------------------------------------             
    # Returns a Px1 vector    
    def EVzx(self,Z,A,B):
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
               
        alpha=self.length_scale*self.length_scale
        S=B[0,0]*B[0,0]
        
        logdetM=pow(S/alpha+1.,-R/2.)
      
        res=np.zeros([P,P])
        
        Sinv = 1/(1/alpha+1/S)
        
        ZZt=Z.dot(Z.T)
        
        E1=(-0.5*(1/alpha)+0.5*(1/alpha)*(1/alpha)*(Sinv))*np.tile(ZZt.diagonal(),[N,1]).T
        E2=(-0.5*(1/S)+0.5*(1/S)*(1/S)*(Sinv))*np.tile(A.dot(A.T).diagonal(),[P,1])
        E3=(1/alpha)*(1/S)*(Sinv)*Z.dot(A.T)
                     
        res=logdetM*np.exp(E1+E2+E3)
    
        return res       
        
    def EVzxVzxT(self,Z,A,B):
        
        N=A.shape[0]
                        
        myFunct = self.EVzxVzxT_single_closure(Z,A,B)
        mapped = np.array(map(myFunct, range(N)))        
        
        return mapped
        
    def EVzxVzxT_single(self,Z,A,B,i):   
        
        P=Z.shape[0]
        R=A.shape[1]
        
        A=np.reshape(A[i,:],[1,R])
        #B=np.reshape(B[i,:],[1,R])        
       
        alpha=self.length_scale*self.length_scale
        S=B[0,0]*B[0,0]
        
        logdetM=pow(2*S/alpha+1.,-R/2.)
      
        res=np.zeros([P,P])
        
        Sinv = 1/(2/alpha+1/S)
        
        ZZt=Z.dot(Z.T)
        
        E1=(-0.5*(1/alpha)+0.5*(1/alpha)*(1/alpha)*(Sinv))*np.tile(ZZt.diagonal(),[P,1]).T
        E2=(-0.5*(1/S)+0.5*(1/S)*(1/S)*(Sinv))*A.dot(A.T)
        E3=(1/alpha)*(1/S)*(Sinv)*Z.dot(A.T) + 0.5*np.tile(E2,[P,1])
        E4=(1/alpha)*(1/alpha)*(Sinv)*ZZt
                     
        E3e=np.tile(E3,[1,P])+np.tile(E3,[1,P]).T
        res=logdetM*np.exp(E1+E1.T+E4+E3e)
    
        return res 

    def EVzxVzxT_single_closure(self,Z,A,B):               
        
        def funct(idx):
            return self.EVzxVzxT_single(Z,A,B,idx)
            
        return funct
        
    def EVxx(self,A,B):
        
        return 0.0       
        
        
    def grad_Kxx_by_hyper(self,Kxx,hyperno):
        
        N = Kxx.shape[0]
        
        return np.zeros([N,1]).ravel()
        
    # derivatives wrt GP input prior mean
    def grad_EVzx_by_c(self,EVzx_this,Z,A,B,C,Kpred,p,r):
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]  
        
        dA=np.zeros([N,R])
        dA[:,r]=Kpred[r][:,p]
               
        alpha=self.length_scale*self.length_scale
        S=B[0,0]*B[0,0]
               
        Sinv = 1/(1/alpha+1/S)
        
        AAt=2*A[:,r]*dA[:,r]        
        
        E1=(-0.5*(1/S)+0.5*(1/S)*(1/S)*(Sinv))*np.tile(AAt,[P,1])
        E3=(1/alpha)*(1/S)*(Sinv)*Z.dot(dA.T)
                     
        res=EVzx_this*(E1+E3)        
                                    
        return res
        
    def grad_EVzxVzxT_by_c(self,EVzxVzxT_list_this,Z,A,B,C,Kpred,p,r):
        
        P=Z.shape[0]
        R=Z.shape[1]
        N=A.shape[0]
        
        ainv=1/(self.length_scale*self.length_scale)
        siginv=1/(B[0,0]*B[0,0])        
      
        dA=np.zeros([N,R])
        dA[:,r]=Kpred[r][:,p]  
        
        AAt=2*A[:,r]*dA[:,r]
        
        res1 = -0.5*np.tile(AAt,[P,1]).T*(siginv-siginv*(1/(siginv+2*ainv))*siginv)
        res1=np.expand_dims(res1,axis=2)
        res1=np.repeat(res1,P,axis=2)        
        
        res2 = dA.dot(Z.T)*(ainv*(1/(siginv+2*ainv))*siginv)
        res2=np.expand_dims(res2,axis=2)
        res2=np.repeat(res2,P,axis=2) 
        
        res3 =np.swapaxes(res2,1,2)
        
        res=EVzxVzxT_list_this*(res1+res2+res3) 
        
        res=np.sum(res,axis=0)
        
        return res  
        
    def grad_EVxx_by_c(self,Kpred,A,B,C,p,r):
                
        return 0.0          
