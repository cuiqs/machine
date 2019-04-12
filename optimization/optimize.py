# -*-coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCase

plt.rcParams["figure.figsize"]=(7.0,4.0)
plt.rcParams['image.interpolation']="nearest"
plt.rcParams['image.cmap']='gray'

def update_parameters_with_gd(parameters,grads,learning_rate):
    """
        使用梯度下降更新参数
    参数:
        parameters -字典,包含来wl,bl
        grads -字典,包含来每个梯度值用以更新参数
    返回:
        parameters -更新后的参数
    """
    L=len(parameters)//2

    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters

# Test update_parameters_with_gd function
"""
parameters,grads,learning_rate=testCase.update_parameters_with_gd_test_case()
parameters=update_parameters_with_gd(parameters,grads,learning_rate)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))
"""

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    """
        从(X,Y)中创建一个随机的mini_batch列表
    参数:
        X -输入的数据,维度为(输入节点数量,样本数量)
        Y -对应的X的标签,维度(1,样本数量)
        mini_batch_size -每一个mini_tatch的样本数量
    返回:
        mini_tatches -一个同步列表,维度(mini_batch_X,mini_batch_Y)
    """
    np.random.seed(seed)
    m=X.shape[1]
    mini_batches=[]

    permutation=list(np.random.permutation(m))
    shuffled_X=X[:,permutation]
    shuffled_Y=Y[:,permutation].reshape(1,m)

    num_complete_minibatches=math.floor(m/mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X=shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y=shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    if m%mini_batch_size!=0:
        mini_batch_X=shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y=shuffled_Y[:,mini_batch_size*num_complete_minibatches:]

        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
"""
#Test random_mini_batches function
X_assess,Y_assess,mini_batch_size=testCase.random_mini_batches_test_case()
mini_batches=random_mini_batches(X_assess,Y_assess,mini_batch_size)
print("first mini_batch_X's dimension:",mini_batches[0][0].shape)
print("first mini_batch_Y's dimension:",mini_batches[0][1].shape)
print("second mini_batch_X's dimension:",mini_batches[1][0].shape)
print("second mini_batch_X's dimension:",mini_batches[1][1].shape)
print("third mini_batch_X's dimension:",mini_batches[2][0].shape)
print("third mini_batch_X's dimension:",mini_batches[2][1].shape)
"""
def initialize_velocity(parameters):
    """
    init speed,velociy is a dictionary:
        -keys :'dW1','db1'......
        -values: 与相应的梯度/参数维度相同的值为0的矩阵
    参数:
        parameters -一个字典,包含来以下参数:
        parameters["W"+str(l)=Wl
        parameters["b"+str(l)=bl
    返回:
        v -一个字典变量,包含了以下参数
        v["dW"+str(l)]=dWl的速度
        v["db"+str(l)]=dbl的速度
    """
    L=len(parameters)//2
    v={}
    
    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

    return v

def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    """
        使用动量更新参数
    参数:
        parameters -字典型变量,包含了以下字段
            parameters["Wl"]=Wl
            parameters["bl"]=bl
        grads -字典型变量,包含一下字段
            grads["dWl"]=dWl
            grads["dbl"]=dbl
        v -包含当前速度的字典型变量,有以下字段
            v["dWl"]=...
            v["dbl"]=...
        beta -超参数,动量,实数
        learing_rate -学习率
    返回:
        parameters -更新后的字典
        v -更新后的速度变量
    """
    L=len(parameters)//2
    for l in range(L):
        v["dW"+str(l+1)]=beta*v["dW"+str(l+1)]+(1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta*v["db"+str(l+1)]+(1-beta)*grads["db"+str(l+1)]
        
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*v["db"+str(l+1)]
    
    return parameters,v

"""
#Test update_parameters_with_momentun function
parameters,grads,v=testCase.update_parameters_with_momentum_test_case()
update_parameters_with_momentun(parameters,grads,v,beta=0.9,learning_rate=0.01)
print("W1="+str(parameters["W1"]))
print('v["dW1"]='+str(v["dW1"]))
"""
def initialize_adam(parameters):
    """
    初始化v和s,都是字典型变量,包含以下字段:
        -keys :'dW1','db1'......
        -values: 与相应的梯度/参数维度相同的值为0的矩阵
    参数:
        parameters -一个字典,包含来以下参数:
        parameters["W"+str(l)=Wl
        parameters["b"+str(l)=bl
    返回:
        v -一个字典变量,包含梯度的指数加权平均值,有如下字段
         v["dW"+str(l)]=...
         v["db"+str(l)]=...
        s -一个字典型变量,包含平方指数的加权平均值,字段如下
         s["dW"+str(l)]=...
         s["dW"+str(l)]=...
    """
    L=len(parameters)//2
    v={}
    s={}
    
    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

        s["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        s["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

    return (v,s)
"""
#Test initialize_adam function
parameters=testCase.initialize_adam_test_case()
v,s=initialize_adam(parameters)
print('v["dW1"]='+str(v["dW1"]))
print('v["dW1"]='+str(v["dW1"]))
"""

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    """
        使用Adam更新参数
    参数:
        parameters -字典型变量,包含了以下字段
            parameters["Wl"]=Wl
            parameters["bl"]=bl
        grads -字典型变量,包含一下字段
            grads["dWl"]=dWl
            grads["dbl"]=dbl
        v -Adam的变量,第一个梯度的移动平均值,字典型变量
        s -Adam的变量,平方梯度的移动平均值
        t -当前迭代的次数
        learning_rate -学习率
        beta1 -动量,超参数,用于第一阶段,使得曲线的Y值不从0开始
        beta2 -RMSprop的参数,超参数
        epsilon -防止除0操作
    返回:
        parameters -更新后的参数
        v -第一个梯度的移动平均值,字典型变量
        s -平方梯度的移动平均值,字典型变量
    """
    L=len(parameters)//2
    v_corrected={}
    s_corrected={}

    for l in range(L):
        v["dW"+str(l+1)]=beta1*v["dW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]

        v_corrected["dW"+str(l+1)]=v["dW"+str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db"+str(l+1)]=v["db"+str(l+1)]/(1-np.power(beta1,t))

        s["dW"+str(l+1)]=beta2*s["dW"+str(l+1)]+(1-beta2)*np.square(grads["dW"+str(l+1)])
        s["db"+str(l+1)]=beta2*s["db"+str(l+1)]+(1-beta2)*np.square(grads["db"+str(l+1)])

        s_corrected["dW"+str(l+1)]=s["dW"+str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db"+str(l+1)]=s["db"+str(l+1)]/(1-np.power(beta2,t))

        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*(v_corrected["dW"+str(l+1)]/np.sqrt(s_corrected["dW"+str(l+1)]+epsilon))
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*(v_corrected["db"+str(l+1)]/np.sqrt(s_corrected["db"+str(l+1)]+epsilon))

    return (parameters,v,s)

"""
#Test update_parameters_with_adam function
parameters,grads,v,s=testCase.update_parameters_with_adam_test_case()
update_parameters_with_adam(parameters,grads,v,s,t=2)
print("W1="+str(parameters["W1"]))
print('v["dW1"]='+str(v["dW1"]))
print('s["dW1"]='+str(s["dW1"]))
"""

def model(X,Y,layers_dims,optimizer,learning_rate=0.0007,mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,epsilon=1e-8,num_epochs=10000,print_cost=True,is_plot=True):
    """
        可以运行在不同优化器模式下的3层神经网络模型
    参数:
        X -输入数据,维度(2,样本数量)
        Y -与X对应的标签
        layers_dims -包含层数和节点数量的列表
        optimizer -字符串类型的参数,['gd'|'momentum'|'adam']
        learning_rate -学习率
        mini_batch_size -每个小批量数据集的大小
        beta -用于动量优化的超参数
        beta1 -用于计算梯度后的指数衰减的估计的超参数
        beta2 -用于计算平方梯度后的指数衰减的估计的超参数
        epsilon -Adam中避免除零操作的超参数
        num_epochs -整个训练集的遍历次数
        print_cost -是否打印误差值
        is_plot -是否绘制曲线图

    返回:
        parameters -学习后的参数
    """
    L=len(layers_dims)
    costs=[]
    t=0 #after learning a minibatch +1
    seed=10

    parameters=opt_utils.initialize_parameters(layers_dims)

    if optimizer=="gd":
        pass
    elif optimizer=="momentum":
        v=initialize_velocity(parameters)
    elif optimizer=="adam":
        v,s=initialize_adam(parameters)
    else:
        print("optimizer parameter error! exit.")
        exit(1)

    for i in range(num_epochs):
        seed=seed+1
        minibatches=random_mini_batches(X,Y,mini_batch_size,seed)


        for minibatch in minibatches:
            (minibatch_X,minibatch_Y)=minibatch
            
            A3,cache=opt_utils.forward_propagation(minibatch_X,parameters)
            cost=opt_utils.compute_cost(A3,minibatch_Y)
            grads=opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)
            
            if optimizer=="gd":
                parameters=update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer=="momentum":
                parameters,v=update_parameters_with_momentun(parameters,grads,v,beta,learning_rate)
            elif optimizer=="adam":
                t=t+1
                parameters,v,s=update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        
        if i%100==0:
            costs.append(cost)
            if print_cost and i%1000==0:
                print("Number "+str(i)+"times iteration,current cost :"+str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs(per 100)')
        plt.show()

    return parameters

train_X,train_Y=opt_utils.load_dataset(is_plot=False)

layers_dims=[train_X.shape[0],5,2,1]
#parameters=model(train_X,train_Y,layers_dims,optimizer="gd",is_plot=True)
#parameters=model(train_X,train_Y,layers_dims,beta=0.9,optimizer="momentum",is_plot=True)
parameters=model(train_X,train_Y,layers_dims,beta=0.9,optimizer="adam",is_plot=True)

