"""
单层隐藏神经网络实例
"""
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

"""
np.random.seed(1)
X,Y=load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c=np.squeeze(Y),s=40,cmap=plt.cm.Spectral)
plt.show()
"""
def layer_size(X,Y):
    """
        X 输入数据集,维度(输入数量,训练/测试的数量)
        Y 标签,维度(输出数量,训练/测试的数量)
     返回:
        n_x: 输入层的数量
        n_h: 隐藏层的数量
        n_y: 输出层的数量
   """
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]

    return (n_x,n_h,n_y)

#测试layer_sizes函数
"""
X_asses,Y_asses=layer_sizes_test_case()
(n_x,n_h,n_y)=layer_sizes(X_asses,Y_asses)
print("n_x=%d\tn_h=%d\tn_y=%d\n"%(n_x,n_h,n_y))
"""
def initialize_parameters(n_x,n_h,n_y):
    """
    返回值:
        W1 权重矩阵,维度(n_h,n_x)
        b1 偏向量,维度(n_h,1)
        W2 权重矩阵,维度(n_y,n_h)
        b2 偏向量,维度(n_y,1)
    """
    np.random.seed(2)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))

    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))

    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}

    return parameters

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return(s)
#前向传播
def forward_propagation(X,parameters):
    """
    参数:
        X:维度为(n_x,m)的输入数据
        parameters:初始化函数(initialize_parameters)的输出
    返回:
        A2:使用sigmoid函数计算的第二次激活后的数值
        cache:包含"Z1","A1","Z2","A2"的字典型变量
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)

    assert(A2.shape==(1,X.shape[1]))
    cache={"Z1":Z1,
           "A1":A1,
           "Z2":Z2,
           "A2":A2}
    return (A2,cache)

#测试forward_propagation
"""
X_assess,parameters=forward_propagation_test_case()
A2,cache=forward_propagation(X_assess,parameters)
print(np.mean(cache["Z1"]),np.mean(cache["A1"]),np.mean(cache["Z2"]),np.mean(cache["A2"]))
"""

#计算损失函数
def compute_cost(A2,Y,parameters):
    """
    参数:
        A2:使用sigmoid函数计算的第二次激活后的数值
        Y:标签向量,维度(1,样本数量)
    返回:
        损失
    """
    m=Y.shape[1]
    
    logprobs=np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost=-np.sum(logprobs)/m
    cost=float(np.squeeze(cost))
    assert(isinstance(cost,float))
    return cost
"""
A2,Y_assess,parameters=compute_cost_test_case()
print("cost="+str(compute_cost(A2,Y_assess,parameters)))
"""
def backward_propagation(parameters,cache,X,Y):
    """
        实现反向传播函数
        参数:parameters 包含参数的字典型变量
             cache 包含Z1,A1,Z2,A2的字典型变量
             X 输入数据,维度(2,数量)
             Y 标签数据,维度(1,数量)
        返回:
            grads 包含W和b的字典型变量
    """
    m=X.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    A1=cache["A1"]
    A2=cache["A2"]
    
    dZ2=A2-Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
    
    grads={"dW1":dW1,
                "db1":db1,
                "dW2":dW2,
                "db2":db2}
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    """
    使用梯度下降发,更新参数
    参数:
    parameters:包含参数的字典型变量
    grads:包含数值的字典型变量
    learning_rate:学习速率
    返回:
    parameters:更新参数的字典型变量
    """
    W1,W2=parameters["W1"],parameters["W2"]
    b1,b2=parameters["b1"],parameters["b2"]
    
    dW1,dW2=grads["dW1"],grads["dW2"]
    db1,db2=grads["db1"],grads["db2"]


    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2    
    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters
def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    """
    参数:
        X 数据集,维度(2,示例数量)
        Y 标签,维度(1,示例数量)
        n_h 隐藏层的数量
        num_iterations 梯度下降循环中的迭代次数
        print_cost 如果为True,每1000次迭代打印一次损失值
    返回:
        parameters 模型学习的参数,可以用来进行预测
    """
    np.random.seed(2)
    n_x=layer_size(X,Y)[0]
    n_y=layer_size(X,Y)[2]

    parameters=initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.5)

        if print_cost:
            if i%1000==0:
                print("Number",i,"\tcost:"+str(cost))
    


    return parameters
def predict(parameters,X):
    A2,cache=forward_propagation(X,parameters)
    predictions=np.round(A2)

    return predictions

X,Y=load_planar_dataset()
parameters=nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)
#plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
predictions=predict(parameters,X)
