import random
import numpy as np
from utils.treebank import StanfordSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from word2vec import *
from sgd import *

#check python version
import sys
assert sys.version_info[0]==3
assert sys.version_info[1]>=5

#Reset the random seed to make sure that everyone gets the same results
random.seed(314)
datasets=StanfordSentiment()
tokens=datasets.tokens()
nWords=len(tokens)

#We are going to train 10-dimensional vectors for this assignment
dimVectors=10

#Context size
C=5

#Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors=np.concatenate(((np.random.rand(nWords,dimVectors)-0.5)/dimVectors,np.zeros((nWords,dimVectors))),axis=0)

wordVectors=sgd(lambda vec:word2vec_sgd_wrapper(skipgram,tokens,vec,datasets,C,negSamplingLossAndGradient),wordVectors,0.3,40000,None,True,PRINT_EVERY=10)

#Note that normalization is not called here .This not bug
#Normalization during training loses the notion of length

print("sanity check: cost at convergence should be around or below 10 ")
print("traing took %d second !" %(time.time()-startTime))

#concatenate input vectors and output vectors
wordVectors=np.concatenate((wordVectors[:nWords,:],wordVectors[nWords:,:]),axis=0)


visualizeWords = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
    "hail", "coffee", "tea"]
visualizeIdx=[tokens[word] for word in visualizeWords]
visualizeVes=wordVectors[visualizeIdx,:]
tmp=(visualizeVes-np.mean(visualizeVes,axis=0))# 这里主要是为计算协方差矩阵才这样安排
# 词向量矩阵行代表样本 列代表特征 而协方差矩阵是比较两个特征之间的线性关系
# 所以此刻他axis=0 就是得到所有样本的均值
covariance=1/len(visualizeIdx)*tmp.T.dot(tmp) #协方差矩阵计算公式
# U,S,V=np.linalg.svd(covariance)#主成分分析就是把协方差矩阵进行奇异值分解，求出最大奇异值的特征方向
#coord=tmp.dot(U[:,0:2])

# for i in range(len(visualizeIdx)):
#     plt.text(coord[i,0],coord[i,1],visualizeWords[i],bbox=dict(facecolor='green',alpha=0.1))
# plt.xlim((np.min(coord[:,0]),np.max(coord[:,0])))
# plt.ylim((np.min(coord[:,1]),np.max(coord[:,1])))
# plt.savefig('word_vectors.png')
#
#



