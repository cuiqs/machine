
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils #part one initialization
import reg_utils #part two regulation
import gc_utils  #part three grads test

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    """
        实现一个三层神经网络:LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    参数:
        X -输入的数据,维度(2,训练/测试的数量)
        Y -标签[0(蓝色)|1(红色)],维度为(1,对应输入数据的标签)
        lambd -正则化的超参数,实数
        keep_prob 随机删除节点的概率
    返回:
        parameters -学习后的参数
    """
    grads={}
    costs=[]
    m=X.shape[0]
    layers_dims=[X.shape[0],20,3,1]

    parameters=reg_utils.initialize_parameters(layers_dims)

    for i in range(0,num_iterations):
        #forward propagation
        if keep_prob==1:
            a3,cache=reg_utils.forward_propagation(X,parameters)
        elif keep_prob<1:
            a3,cache=forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("keep_prob error,quit!")
            exit
        
        if lambd==0:
            cost=reg_utils.compute_cost(a3,Y)
        else:
            cost=compute_cost_with_regularization(a3,Y,parameters,lambd)
        
        #backward propagation
        assert(lambd==0 or keep_prob==1)
        if(lambd==0 and keep_prob==1):
            grads=reg_utils.backward_propagation(X,Y,cache)
        elif lambd!=0:
            grads=backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:
            grads=backward_propagation_with_dropout(X,Y,cache,keep_prob)

        parameters=reg_utils.update_parameters(parameters,grads,learning_rate)
        
        if i%1000==0:# print cost
            costs.append(cost)
            if(print_cost and i%10000==0):
                print("number time"+str(i)+" cost is "+str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(*1000)')
        plt.title('Learing rate='+str(learning_rate))
        plt.show()
    
    return parameters 

def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
        L2 正则化成本计算
    参数:
        A3 -正向传播的输出结果,维度(输出节点数量,训练/测试数量)
        Y  -标签向量    维度(输出节点数量,训练/测试数量)
        parameters -模型学习后的参数字典
        lambd -正则化超参数
    返回:
        cost -正则化损失的值
    """
    m=Y.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]

    cross_entropy_cost=reg_utils.compute_cost(A3,Y)
    L2_regularization_cost=lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m)

    cost=cross_entropy_cost+L2_regularization_cost
    return cost

def backward_propagation_with_regularization(X,Y,cache,lambd):
    """
    L2正则化的模型的后向传播
    参数:
        X -输入数据集 维度(输入节点数量,训练/测试数量)
        Y -标签,维度(输出节点数量,训练/测试数量)
        cache -来自forward_propagation()中的cache输出
        lambd -regularization超参数,实数
    返回:
        gradients -包含来每个参数,激活值和预激活变量的梯度的字典
    """
    
    m=X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    
    dZ3=A3-Y
    dW3=(1/m)*np.dot(dZ3,A2.T)+((lambd*W3)/m)
    db3=(1/m)*np.sum(dZ3,axis=1,keepdims=True)

    dA2=np.dot(W3.T,dZ3)
    dZ2=np.multiply(dA2,np.int64(A2>0))
    dW2=(1/m)*np.dot(dZ2,A1.T)+((lambd*W2)/m)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)
    dZ1=np.multiply(dA1,np.int64(A1>0))
    dW1=(1/m)*np.dot(dZ1,X.T)+((lambd*W1)/m)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    gradients={"dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,"dZ2":dZ2,
        "dW2":dW2,"db2":db2,"dA1":dA1,"dW1":dW1,"db1":db1,"dZ1":dZ1}
    return gradients

def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    """
        实现具有随机舍弃节点的前向传播
        LINEAR->RELU+DROPOUT->LINEAR->RELU+DROPOUT->LINEAR-SIGMOID
    参数:
        X -输入数据集,维度(2,示例数)
        parameters 包含参数W1,b1,W2,b2,W3,b3的python字典
         W1 -权重矩阵,维度(20,2)
         b1 -偏向量,维度(20,1)
         W2 -权重矩阵,维度(3,20)
         b2 -偏向量,维度(3,1)
         W3 -权重矩阵,维度(1,3)
         b3 -偏向量,维度(1,1)
         keep_prob -随机删除节点的概率,实数
    返回:
        A3 -最后的激活值,正向传播的输出
        cache -存储了用于反向传播的数值的元组
    """
    np.random.seed(1)
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    b3=parameters["b3"]

    Z1=np.dot(W1,X)+b1
    A1=reg_utils.relu(Z1)

    D1=np.random.rand(A1.shape[0],A1.shape[1])
    D1=D1<keep_prob
    A1=A1*D1
    A1=A1/keep_prob

    Z2=np.dot(W2,A1)+b2
    A2=reg_utils.relu(Z2)

    D2=np.random.rand(A2.shape[0],A2.shape[1])
    D2=D2<keep_prob
    A2=A2*D2
    A2=A2/keep_prob

    Z3=np.dot(W3,A2)+b3
    A3=reg_utils.sigmoid(Z3)

    cache=(Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
        实现随机删除节点模型的后向传播
    参数:
        X -输入数据集,维度(2,示例数)
        Y -标签,维度(输出节点数量,示例数)
        cache -来自前向传播的cache输出
        keep_prob -随机删除节点的概率
    返回:
        gradients -关于每个参数,激活值和预激活变量的梯度值的字典
    """
    m=X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)=cache

    dZ3=A3-Y
    dW3=(1/m)*np.dot(dZ3,A2.T)
    db3=1./m*np.sum(dZ3,axis=1,keepdims=True)
    dA2=np.dot(W3.T,dZ3)

    dA2=dA2*D2
    dA2=dA2/keep_prob

    dZ2=np.multiply(dA2,np.int64(A2>0))
    dW2=1./m*np.dot(dZ2,A1.T)
    db2=1./m*np.sum(dZ2,axis=1,keepdims=True)

    dA1=np.dot(W2.T,dZ2)
    dA1=dA1*D1
    dA1=dA1/keep_prob

    dZ1=np.multiply(dA1,np.int64(A1>0))
    dW1=1./m*np.dot(dZ1,X.T)
    db1=1./m*np.sum(dZ1,axis=1,keepdims=True)

    gradients={"dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,"dZ2":dZ2,
        "dW2":dW2,"db2":db2,"dA1":dA1,"dW1":dW1,"db1":db1,"dZ1":dZ1}

    return gradients

train_X,train_Y,test_X,test_Y=reg_utils.load_2D_dataset(is_plot=False)
#parameters=model(train_X,train_Y,lambd=0.7,is_plot=False)
parameters=model(train_X,train_Y,keep_prob=0.86,learning_rate=0.3,is_plot=True)
print("train set")
predictions_train=reg_utils.predict(train_X,train_Y,parameters)
print("test set")
predictions_test=reg_utils.predict(test_X,test_Y,parameters)
plt.show()

