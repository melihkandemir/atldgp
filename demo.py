#
# Demo application for the ATL-DGP model.
#
# Paper: M. Kandemir, Asymmetric Transfer Learning with Deep Gaussian Processes, ICML, 2015
#
# Contact: melihkandemir@gmail.com
#
# All rights reserved.
#
from DSGPSymmetricTransferClassifier import DSGPSymmetricTransferClassifier
from DSGPAsymmetricTransferClassifier import DSGPAsymmetricTransferClassifier
from RBFKernel import RBFKernel
import numpy as np
import scipy.io

if __name__ == "__main__":

    num_inducing_points=10    # Number of inducing points
    num_hidden_units_source=2 # Number of hidden units per class for the source task
    num_hidden_units_target=2 # Number of hidden units per class for the target task    
    max_iteration_count=1     # Maximum number of iterations allowed
    learning_rate_start=0.001 # Starting learning rate
    inducing_kernel=RBFKernel(np.sqrt(num_hidden_units_source))
    
    # Load the data set
    mat = scipy.io.loadmat('data/webcam_to_dslr_rep_1.mat')

    Xtrain_source = mat["XtrSource"]
    ytrain_source = mat["ytrSource"]-1  

    Nsrc = Xtrain_source.shape[0]              

    Xtrain_target = mat["XtrTarget"]
    ytrain_target = mat["ytrTarget"]-1              
    
    # Concatenate source and target data sets
    Data =np.concatenate((Xtrain_source,Xtrain_target))
    labels = np.concatenate((ytrain_source.T,ytrain_target.T))    
    
    Xtest_target = mat["XtsTarget"]
    ytest_target = mat["ytsTarget"]-1                  
    
    # Construct the source-target info map. 
    # 0: data point is on the source task
    # 1: data point is on the target task
    source_target_info = np.ones([Data.shape[0],1])
    source_target_info[0:Nsrc] = 0    
    
    # Construct kernel lists
    kernels_source=list()  
    for rr in range(num_hidden_units_source):
       length_scale=Data.shape[1]
       kernel=RBFKernel(length_scale)  
       kernels_source.append(kernel)
         
    kernels_target=list()  
    for rr in range(num_hidden_units_target):
       length_scale=Data.shape[1]
       kernel=RBFKernel(length_scale)  
       kernels_target.append(kernel)     

#  Comment in the lines below to try out the symmetric classifier
#    common_dimensions=2
#    model=DSGPSymmetricTransferClassifier(inducing_kernel,kernels_source, kernels_target, 2, \
#                                             num_inducing=num_inducing_points, \
#                                             max_iter=iter_cnt, \
#                                             learning_rate_start=learning_rate_start)                         
                         

    # Create the class object for the asymmetric classifier                         
    model=DSGPAsymmetricTransferClassifier(inducing_kernel,kernels_source, kernels_target, num_inducing_points, \
                                            max_iteration_count, learning_rate_start=learning_rate_start)                         
                        
    # Train the model                        
    model.train(Data,labels,source_target_info)
    
    # Predict on test data and report accuracy
    predictions=model.predict(Xtest_target)
    buf = "Accuracy: %.1f %% vs Random: 10.0 %%\n" % ( np.mean(predictions.predictions==ytest_target)*100)
    print buf