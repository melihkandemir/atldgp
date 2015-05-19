from Kernel import Kernel
import numpy as np
import scipy
#
# exp(-0.5 ||x1-x2||_2^2 / (2*sqrt(length_scale)))
#
class LinearKernel(Kernel):
    
    scale=1.
    num_hyperparams=0
    
    def __init__(self):
        self.scale=1.
        self.num_hyperparams=0
        
    def compute(self,X,Y):
       
        
        res=X.dot(Y.T)/self.scale
        
        
        return res
        
    def selfCompute(self,X):

        res =  self.compute(X,X)
        
        #self.scale = np.max(X)
        
        res = res/self.scale
        
        return res
        
    def computeSelfDistance(self,X):
        
        N = X.shape[0]
        
        res = np.zeros([N,1])
        
        for nn in range(0,N):
            res[nn] = X[nn,:].dot(X[nn,:])
            
        return res    
        
    # Derivatives of the kernel matrix wrt a visible entry
    def grad_EVzx_by_Z(self,EVzx_this,Z,A,B,p,r):
                 
        dZ=np.zeros([Z.shape[0], Z.shape[1]])
        
        dZ[p,r]=1
                
        return dZ.dot(A.T)
         
    def grad_EVxx_by_Z(self,Z,A,p,r):
        
        return 0.0
        
    def grad_EVzxVzxT_by_Z(self,EvzxVzxT_list,Z,A,B,p,r):
      
        dZ=np.zeros([Z.shape[0], Z.shape[1]])
        
        dZ[p,r]=1
        
        res=dZ.dot(A.T.dot(A)+A.shape[0]*np.identity(Z.shape[1])).dot(Z.T)+Z.dot(A.T.dot(A)+A.shape[0]*np.identity(Z.shape[1])).dot(dZ.T)
            
        return res  
        
    def grad_K_by_Z_fast(self,Z,p,r):
        dZ=np.zeros([Z.shape[0], Z.shape[1]])
        
        dZ[p,r]=1
                
        return dZ.dot(Z.T)+Z.dot(dZ.T)     
        
    def grad_K_by_Z_slow(self,Z,p,r):
        dZ=np.zeros([Z.shape[0], Z.shape[1]])
        
        dZ[p,r]=1
                
        return dZ.dot(Z.T)+Z.dot(dZ.T)    
        
    def grad_K_by_hyper(self,K,Z,hyperno):
        return -1     
        
    def grad_K_by_Z(self,K,Z,p,r):
        dZ=np.zeros([Z.shape[0], Z.shape[1]])
        
        dZ[p,r]=1
                
        return dZ.dot(Z.T)+Z.dot(dZ.T)
        
    def grad_K_inv_by_Z(self,K,K_inv,Z,p,r):
        gVzz = self.grad_K_by_Z(K,Z,p,r)
    
        res = -K_inv.dot(gVzz).dot(K_inv)
        
        return res        
        
        
    # -------------------------------------------------------------------------
    # Expectation wrt a normal distribution
    # -----------------------------------------------------------------------
        
    def EVzxVzxT(self,Z,A,B):
        
        res=Z.dot(A.T.dot(A)+A.shape[0]*np.identity(Z.shape[1])).dot(Z.T)                

        res=np.tile(res/A.shape[0],[A.shape[0],1,1])        
        
        return res        
            
    # Returns a Px1 vector    
    def EVzx(self,Z,A,B):

        return Z.dot(A.T)        
        
    def EVxx(self,A,B):
        
        res=A.dot(A.T).diagonal()+np.sum(np.power(B,2),axis=1)
        res=sum(res)
        
        return res
        
    def grad_EVzx_by_c(self,EVzx_this,Z,A,B,C,Kpred,p,r):
        
        return np.outer(Kpred[r][:,p],Z[:,r]).T
        
        
    def grad_EVzxVzxT_by_c(self,EVzxVzxT_list_this,Z,A,B,C,Kpred,p,r):
        P=Z.shape[0]
        R=Z.shape[1]
        
        M=np.zeros([R,R])
        
        dC=np.zeros([P,1]).ravel()
        dC[p]=1.
        
        for rr in range(R):
            M[r,rr] = C[:,rr].dot(Kpred[rr].T).dot(Kpred[r]).dot(dC)
            M[rr,r] = M[r,rr] 
            
        M[r,r] += M[r,r]
        
        res = Z.dot(M).dot(Z.T)
        
        return res
        
    def grad_EVxx_by_c(self,Kpred,A,B,C,p,r):
        P=Kpred[0].shape[1]        
        dC=np.zeros([P,1]).ravel()
        dC[p]=1.
                
        return sum(2*Kpred[r].dot(np.outer(C[:,r],dC)).dot(Kpred[r].T).diagonal())
