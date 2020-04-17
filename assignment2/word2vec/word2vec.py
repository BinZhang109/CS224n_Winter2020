import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows,softmax

def sigmoid(x):
    """
    compute the sigmoid function for the input here
    :param x: A scalar or numpy array
    :return: sigmoid(x)
    """
    s=1/1+np.exp(-x)
    return s
def naiveSoftmaxLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset
):
    """
    Naive Softmax Loss & Gradient function for word2vec model
    implement the naive softmax loss and gradients between a center's word
    embedding and an outside word's embedding .This will be the building bolck for
    our word2vec models
    :param centerWordVec:
    numpy ndarray center word's embedding
    :param outsideWordIdx:
     integer the index of outside word's embedding
    :param outsideVectors:
    outside vector (rows of matrix) for all words in vocab
    :param dataset:
    needed for negative samping unused here
    :return:
    loss ---- navie softmax loss
    gradCenterVec --- the gradient with respect to the center word 就是对V_c的导数
    gradOutsideVecs ---the graident with respect to all the outside word vector 就是对U的导数
    """
    y_hat=softmax(centerWordVec@outsideVectors.T)
    loss=-np.log(y_hat[outsideWordIdx])# 损失函数 交叉熵损失函数 对特定outsideID的损失
    y=np.zeros_like(y_hat)
    y[outsideWordIdx]=1 # 就只有所选定的那个上下文元素对应的y的索引为1
    gradCenterVec=(y_hat-y)@outsideVectors # 这是对中心向量求导后的值
    gradOutsideVecs=(y_hat-y).reshape(-1,1)@centerWordVec.reshape(1,-1)# 这是对上下文向量求导后的结果
    return loss,gradCenterVec,gradOutsideVecs

def getNegativeSamples(outsideWordId,datasets,K):
    """
    从数据集负采样K个不等于outsideWordId的样本
    :param outsideWordId: 上下文向量的ID
    :param datasets:数据集
    :param K: 负采样的值
    :return: 含有负采样的列表
    """
    negSampleWordIndices=[None]*K
    for k in range(K):
        newidx=datasets.sampleTokenIdx()
        while (newidx==outsideWordId).all():
            newidx=datasets.sampleTokenIdx()
        negSampleWordIndices[k]=newidx
    return negSampleWordIndices

def negSamplingLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset,
        K=10
):
    """
    Word2vec模型的负采样的损失函数
    Implement the negative sampling loss and gradients for a centerWordVector
    and a outsideWordIdx word vector as a building block for word2vec models
    K is the number of negative samples to take
    Note The same of words may be negatively sampled multipy times
    for example if an outside words is sample twice ,you shall have to
    double count the gradients with respect to this word
     :param centerWordVec:
    numpy ndarray center word's embedding
    :param outsideWordIdx:
     integer the index of outside word's embedding
    :param outsideVectors:
    outside vector (rows of matrix) for all words in vocab
    :param dataset:
    needed for negative samping unused here
    :return:
    loss ---- navie softmax loss
    gradCenterVec --- the gradient with respect to the center word 就是对V_c的导数
    gradOutsideVecs ---the graident with respect to all the outside word vector 就是对U的导数
    """
    #Negative sampling of words is done for you
    negSampleWordIndices=getNegativeSamples(outsideVectors,dataset,K)

    uv_oc=sigmoid((outsideVectors[outsideWordIdx,:]*centerWordVec).sum())# (1,1) uv_oc=U_o.T@V_c 因为此时U_o和V_c均为行
                                                                         #向量所以可以直接让他们对应元素两两相乘再求和 这里就是这么解决的
    uv_kc=sigmoid(-centerWordVec@outsideVectors[negSampleWordIndices,:].T)# (1,k)

    loss=-np.log(uv_oc)-np.log(np.sum(uv_kc))

    gradCenterVec=(1-uv_kc)@outsideVectors[negSampleWordIndices,:]-(1-uv_oc)*outsideVectors[outsideWordIdx]

    gradOutsideVecs=np.zeros_like(outsideVectors)
    gradOutsideVecs[outsideWordIdx]=(uv_oc-1)*centerWordVec

    #由于负采样可能会吧一些单词重复采出，所以梯度需要累加，而下面这种方式值保留最后一次梯度所以必须以循环的方式计算
    #gradOutsideVecs[negSampleWordIndices]=(1-uv_kc).reshape(-1,1)@centerWordVec.reshape(1,-1)
    gradOutsideKsampling=(1-uv_kc).reshape(-1,1)@centerWordVec.reshape(1,-1)
    for i,idx in enumerate(negSampleWordIndices):
        gradOutsideVecs[idx]+=gradOutsideKsampling[i]
    return loss,gradCenterVec,gradOutsideVecs

def skipgram(
        currentCenterWord,
        windowSize,
        outsideWords,
        word2Ind,
        CenterWordVectors,
        outsideVectors,
        dataset,
        word2vecLossAndGradients=naiveSoftmaxLossAndGradient
):
    """
    word2vec的skip_gram模型
    :param currentCenterWord:当前中心字的字符串
    :param windowsize: integer 窗口大小
    :param outsideWords: list 大小不超过2倍的窗口大小
    :param word2Ind: dict 将单词映射成索引的字典
    :param CenterWordVectors: vector 在vocab中的所有单词中心词向量
    :param outsideVectors: vector 在vocab中的所有单词的外部向量
    :param dataset:
    :param word2vecLossAndGradients: 在给定outsideWordId的情况下 计算此时的loss和gradients
    :return:
    loss ---- navie softmax loss
    gradCenterVec --- the gradient with respect to the center word 就是对V_c的导数
    gradOutsideVecs ---the graident with respect to all the outside word vector 就是对U的导数
    """
    loss=0.0
    gradCenterVec=np.zeros(CenterWordVectors.shape)
    gradOutsideVecs=np.zeros(outsideVectors.shape)

    ###注意一个窗口内得到的梯度相加得到一次梯度，并不做平均

    currentCenterWordInd=word2Ind[currentCenterWord]
    centerwordVector=CenterWordVectors[currentCenterWordInd]
    for outsideWord in outsideWords:
        outsideWordInd=word2Ind[outsideWord]
        #outsideVector=outsideVectors(outsideWordInd)
        tmp_loss,tmp_gradCenterVec,tmp_gradOutsideVecs=word2vecLossAndGradients(centerwordVector,outsideWordInd,outsideVectors,dataset)
        loss+=tmp_loss
        gradCenterVec[currentCenterWordInd,:]+=tmp_gradCenterVec
        gradOutsideVecs+=tmp_gradOutsideVecs
    return loss,gradCenterVec,gradOutsideVecs

def word2vec_sgd_wrapper(
        word2vecModel,
        word2Ind,
        wordVectors,
        dataset,
        windowSize,
        word2vecLossAndGradients=naiveSoftmaxLossAndGradient
):
    """
    :param word2vecModel:
    :param word2Ind:
    :param wordVectors:
    :param dataset:
    :param windowSize:
    :param word2vecLossAndGradients:
    :return:
    """
    batchsize=50
    loss=0.0
    grad=np.zeros(wordVectors.shape)
    N=wordVectors.shape[0]
    centerWordVectors=wordVectors[:int(N/2),:]
    outsideVectors=wordVectors[int(N/2):,:]
    for i  in range(batchsize):
        windowsize=random.randint(1,windowSize)
        centerWord,context=dataset.getRandomContext(windowsize)
        c,gin,gout=word2vecModel(centerWord,windowsize,context,word2Ind,centerWordVectors,outsideVectors,dataset,word2vecLossAndGradients)
        loss+=c/batchsize
        grad[:int(N/2),:]+=gin/batchsize
        grad[int(N/2):,:]+=gout/batchsize
    return loss,grad

def test_word2vec():
    """Test the two word2vec implements """
    dataset=type('dummy',(),{})()
    def dummySampleTokenIdx():
        return random.randint(0,4)
    def getRandomContext(C):
        token=['a','b','c','d','e']
        return token[random.randint(0,4)],[token[random.randint(0,4)] for i in range(2*C)]
    dataset.sampleTokenIdx=dummySampleTokenIdx
    dataset.getRandomContext=getRandomContext

    random.seed(31415)
    np.random.seed(9265)

    dummy_vectors=normalizeRows(np.random.randn(10,3))
    dummy_tokens=dict([("a",0),("b",1),("c",2),("d",3),("e",4)])

    print("=====Gradient check for skip-gram with naiveSoftmaxLossAndGradients==========")
    gradcheck_naive(lambda vec:word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5,naiveSoftmaxLossAndGradient),dummy_vectors,"naiveSoftmaxGradientsand loss")

    print("=====Gradient check for skip-gram with negSamplingGradientsAndLoss==========")
    gradcheck_naive(lambda vec:word2vec_sgd_wrapper(skipgram,dummy_tokens,vec,dataset,5,negSamplingLossAndGradient),dummy_vectors,"negSamplingLossAndGradient loss")

    print("\n====Result========")
    print("Skip-gram with naiveSoftmaxLossAndGradient")

    print("Your result!!")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                  dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    )
    )
    print("Expected result: Value should approximate thess:")

    print("""Loss: 11.16610900153398
    Gradient wrt Center Vectors (dJ/dV):
     [[ 0.          0.          0.        ]
     [ 0.          0.          0.        ]
     [-1.26947339 -1.36873189  2.45158957]
     [ 0.          0.          0.        ]
     [ 0.          0.          0.        ]]
    Gradient wrt Outside Vectors (dJ/dU):
     [[-0.41045956  0.18834851  1.43272264]
     [ 0.38202831 -0.17530219 -1.33348241]
     [ 0.07009355 -0.03216399 -0.24466386]
     [ 0.09472154 -0.04346509 -0.33062865]
     [-0.13638384  0.06258276  0.47605228]]
        """)

    print("Skip-Gram with negSamplingLossAndGradient")
    print("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :],
                  dummy_vectors[5:, :], dataset, negSamplingLossAndGradient)
    )
    )
    print("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
    Gradient wrt Center Vectors (dJ/dV):
     [[ 0.          0.          0.        ]
     [ 0.          0.          0.        ]
     [-4.54650789 -1.85942252  0.76397441]
     [ 0.          0.          0.        ]
     [ 0.          0.          0.        ]]
     Gradient wrt Outside Vectors (dJ/dU):
     [[-0.69148188  0.31730185  2.41364029]
     [-0.22716495  0.10423969  0.79292674]
     [-0.45528438  0.20891737  1.58918512]
     [-0.31602611  0.14501561  1.10309954]
     [-0.80620296  0.36994417  2.81407799]]
        """)

if __name__ == '__main__':
    test_word2vec()

