SAVE_PARAMS_EVERY=5000

import pickle
import glob
import random
import numpy as np
import os.path as op

def load_saved_params():
    """
    A helper functions that loads previously saved parameters and
    resets iteration start
    :return:
    """
    st=0
    for f in glob.glob("save_params_*.npy"):
        iter =int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter>st):
            st=iter
    if st>0:
        params_file="saved_params_%d.npy" %st
        state_file="saved_state_%d.pickle" %st
        params=np.load(params_file)
        with open(state_file,"rb") as f:
            state=pickle.load(f)
        return st,params,state
    else:
        return st,None,None

def save_params(iter,params):
    params_file="saved_params_%d.npy"%iter
    np.save(params_file,params)
    #with open("saved_state_%d.pickle" %iter ,"wb") as f
    #       pickle.dump(random.getstate(),f)

def sgd(f,x0,step,iterations,postprocessing=None,useSaved=False,PRINT_EVERY=10):
    """
    Stochastic gradient descent

    :param f: the function to optimize it could take a single argument and yeild two outputs
                a loss and the gradients with respect to the argument
    :param x0: the initial point to start SGD from
    :param step: the step size fro SGD
    :param iterations: total iteration to run SGD for
    :param postprocessing: postprocessing function for parameters if necessary  in the case of t
                           the word2vec we will need to normalize the word vector to have unit length
    :param useSaved:
    :param PRINT_EVERY: 指定输出loss的迭代次数
    :return:

    x --- 完成SGD后的参数值
    """
    # Anneal learning  rate every several iterations
    ANNEAL_EVERY=20000

    if useSaved:
        start_iter,oldx,state=load_saved_params()
        if start_iter>0:
            x0=oldx
            step*=0.5**(start_iter/ANNEAL_EVERY)
        if state:
            random.setstate(state)
    else:
        start_iter=0
    x=x0

    if not postprocessing:
        postprocessing=lambda x:x

    for iter in  range(start_iter+1,iterations+1):
        # you might want to print the process every few iteation
        # loss=None
        loss,grad=f(x)
        x-=step*grad

        x=postprocessing(x)
        exploss=None

        if iter % PRINT_EVERY==0:
            if not exploss:
                exploss=loss
            else:
                exploss= .95 * exploss+.05*loss
            print("iter %d: %f "%(iter,exploss))
        if iter % SAVE_PARAMS_EVERY==0 and useSaved:
            save_params(iter,x)
        if iter% ANNEAL_EVERY==0:
            step*=0.5
    return x
def sanity_check():
    quad= lambda x : (np.sum(x**2),x*2)

    print("Running sanity check !!!!")
    t1=sgd(quad,0.5,0.01,1000,PRINT_EVERY=100,useSaved=True)
    print("test t1 result:",t1)
    assert abs(t1)<=1e-6

    t2=sgd(quad,0.0,0.01,1000,PRINT_EVERY=100)
    print("test t2 result:",t2)
    assert abs(t2)<=1e-6

    t3=sgd(quad,-1.5,0.01,1000,PRINT_EVERY=100)
    print("test t3 result:",t3)
    assert abs(t3)<=1e-6

    print('-'*40)
    print("All test pass!!!")
    print('-'*40)


if __name__ == '__main__':
    sanity_check()







