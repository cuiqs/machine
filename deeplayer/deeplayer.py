#多层神经网络示例


import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils
import scipy 
from scipy import ndimage
#np.random.seed(1)
def initialize_parameters(n_x,n_h,n_y):
    """
    此函数用来初始化两层网络参数而使用的函数
    参数:
    n_x:输入层节点的数量
    n_h:隐藏层节点的数量
    n_y:输出层节点的数量
    返回值:
    parameters:包含给的参数的字典
    W1:权重矩阵,维度(n_h,n_x)
    b1:偏向量,维度(n_h,1)
    W2:权重矩阵,维度(n_y,n_h)
    b2:偏向量,维度为(n_y,1)
    """
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    
    return parameters
"""
#Test initialize_parametes function
parameters=initialize_parameters(3,2,1)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))
"""
def initialize_parameters_deep(layers_dims):
    """
    此函数用来初始化多层网络参数而使用的函数
    参数:
    layers_dims-包含网络中每个图层的节点数量的列表
    返回值:
    parameters:包含给的参数的字典
    W1:权重矩阵,维度(layers_dims[1],layers_dims[1-1]????)
    b1:偏向量,维度(layers_dims[1],1)
    ......
    """
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
    
    return parameters

"""
#Test initialize_parameters_deep function
layers_dims=[5,4,3]
parameters=initialize_parameters_deep(layers_dims)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))
"""
def linear_forward(A,W,b):
    """
    参数:
    A-来自上一层(或输入数据的激活,维度(上一层节点数量,示例节点数量)
    W-权重矩阵,numpy数组,维度(当前图层节点数量,前一图层节点数量)
    b-偏向量,numpy向量,维度(当前图层节点数量,1)
    返回:
    Z-激活功能的输入,也称为预激活参数
    cache-一个包含A,W,b的字典,存储这些变量用以有些计算后向传递
    """
    Z=np.dot(W,A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)

    return Z,cache

"""    
#Test linear_forward
A,W,b=testCases.linear_forward_test_case()
Z,linear_cache=linear_forward(A,W,b)
print("Z="+str(Z))
"""
def linear_activation_forward(A_prev,W,b,activation):
    """
    实现LINEAR->ACTIVATION这一层的前向传播
    参数:
    A_prev-来自上一层的(或者输入层)的激活,维度(上一层节点数量,示例数)
    W - 权重矩阵,numpy数组,维度(当前层的节点数量,前一层的大小)
    b - 偏向量,维度(当前层的节点数量,1)
    activation -选择在该层使用的激活函数,字符串类型[sigmoid|relu]
    返回:
     A -激活函数的输出,也称为激活后的值
     cache - 一个包含linear_cache和activation_cache的字典,存储塔用来有效的计算后向传递
    """
    if activation=="sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    elif activation=="relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)

    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)

    return A,cache
"""
#Test linear_activation_forward
A_prev,W,b=testCases.linear_activation_forward_test_case()
A,linear_activation_cache=linear_activation_forward(A_prev,W,b,activation="sigmoid")
print("sigmoid A="+str(A))

A,linear_activation_cache=linear_activation_forward(A_prev,W,b,activation="relu")
print("relu A="+str(A))
"""

def L_model_forward(X,parameters):
    """
    实现多层网络的前向传播,为后面每一层都执行LINEAR和ACTIVATION
    参数:
        X - 数据,numpy数组,维度(输入节点数量,示例数)
        parameters -initialize_parameters_deep()的输出
    返回:
        AL -最后的激活值
        cache -包含以下内容的缓存列表:
            linear_relu_forward()的每个cache(有L-1个,索引从0到L-2)
            linear_sigmoid_forward()的cache(1个,索引为L-1)
    """
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)

    assert(AL.shape==(1,X.shape[1]))

    return AL,caches

"""
#Test L_model_forward function
X,parameters=testCases.L_model_forward_test_case()
AL,caches=L_model_forward(X,parameters)
print("AL="+str(AL))
print("length of caches is :"+str(len(caches)))
"""

def compute_cost(AL,Y):
    """
        计算成本函数
    参数:
        AL -与标签预测相对应的概率向量,维度(1,示例数量)
        Y -标签向量
    返回:
        cost -交叉熵成本
    """
    m=Y.shape[1]
    cost=-np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m

    cost=cost.squeeze(int(cost))
    assert(cost.shape==())
    return cost

"""
#Test compute_cost
Y,AL=testCases.compute_cost_test_case()
print("cost="+str(compute_cost(AL,Y)))
"""

def linear_backward(dZ,cache):
    """
    为单层实现反向传播的线性部分(第L层)
    参数:
    dZ -相对于(当前第l层的)线性输出的成本梯度
    cache - 来自当前层前向传播的元组(A_prev,W,b)
    返回:
    dA_prev -相对于激活(前一层l-1)成本梯度,与A_prev维度相同
    dW -相对于W(当前层l)的成本梯度,与W维度相同
    db -相对于b(当前层l)的成本梯度,与b维度相同
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)

    return dA_prev,dW,db

"""
#Test linear_backward function
dZ,linear_cache=testCases.linear_backward_test_case()
dA_prev,dW,db=linear_backward(dZ,linear_cache)
print("dA_prev="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db))
"""

def linear_activation_backward(dA,cache,activation="relu"):
    """
    实现linear->activation层的后向传播
    参数:
    dA -当前层的激活后的梯度值
    cache -前向过程中存储的用于计算反向传播的值的元组
    activation -激活函数名称 sigmoid或者relu
    返回:
    dA_prev -相对于激活(前一层l-1)成本梯度,与A_prev维度相同
    dW -相对于W(当前层l)的成本梯度,与W维度相同
    db -相对于b(当前层l)的成本梯度,与b维度相同
    """
    linear_cache,activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    if activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    return dA_prev,dW,db

"""    
#Test linear_activation_backward function
AL,linear_activation_cache=testCases.linear_activation_backward_test_case()
dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="sigmoid")
print("sigmoid")
print("dA_prev="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db))

dA_prev,dW,db=linear_activation_backward(AL,linear_activation_cache,activation="relu")
print("relu")
print("dA_prev="+str(dA_prev))
print("dW="+str(dW))
print("db="+str(db))
"""

def L_model_backward(AL,Y,caches):
    """
        实现多层网络的向后传播
    参数:
        AL -概率向量,正向传播(L_model_forward())的输出
        Y -标签向量,维度(1,数量)
        cache -包含以下内容的cache列表:
            linear_activation_forward(relu)的cache
            linear_activation_forward(sigmoid)的cache
    返回:
        grad -具有梯度值的字典
            grad['dA'+str(l)]=...
            ...
   """
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache=caches[L-1]    
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,"sigmoid")

    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp

    return grads

"""
#Test L_model_backward
AL,Y,caches=testCases.L_model_backward_test_case()
grads=L_model_backward(AL,Y,caches)
print("dW1="+str(grads["dW1"]))
print("dA1="+str(grads["dA1"]))
print("db1="+str(grads["db1"]))
"""
def update_parameters(parameters,grads,learning_rate):
    """
    使用梯度下降更新参数
    参数:
        parameters -包含你的参数的字典
        grads -包含梯度值的字典,L_model_backward的输出
    返回:
        parameters -更新来参数的字典
    """
    L=len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
        
    return parameters
"""    
#Test updata_parameters function
parameters,grads=testCases.update_parameters_test_case()
parameters=update_parameters(parameters,grads,0.1)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))
"""

def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """
        实现一个两层的神经网络,[linear->relu]->[linear-sigmoid]
    参数:
        X -输入的数据,维度(n_x,例子数)
        Y -标签向量,0为非猫,1为猫
        layers_dims -层数的向量,维度为(n_x,n_h,n_y)
        learing_rate -学习率
        num_iterations -迭代的次数
        print_cost -是否打印成本值,每迭代100次打印一次
        isPlot -是否绘制误差值的图谱
    返回:
        parameters -一个包含W1,b1,W2,b2的字典变量
    """
    np.random.seed(1)
    grads={}
    costs=[]
    (n_x,n_h,n_y)=layers_dims

    parameters=initialize_parameters(n_x,n_h,n_y)
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    for i in range(0,num_iterations):
        A1,cache1=linear_activation_forward(X,W1,b1,"relu")
        A2,cache2=linear_activation_forward(A1,W2,b2,"sigmoid")

        cost=compute_cost(A2,Y)
        
        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))
        
        dA1,dW2,db2=linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,"relu")

        grads["dW1"]=dW1
        grads["db1"]=db1
        grads["dW2"]=dW2
        grads["db2"]=db2

        parameters=update_parameters(parameters,grads,learning_rate)
        
        W1=parameters["W1"]
        b1=parameters["b1"]
        W2=parameters["W2"]
        b2=parameters["b2"]

        if i%100==0:
            costs.append(cost)
            if print_cost:
                print("Number ",i,"times. cost value:",np.squeeze(cost))
    
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration(pertens)')
        plt.title("Learning rate="+str(learning_rate))
        plt.show()
    
    return parameters



def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False,isPlot=True):
    """
        实现一个L层神经网络
    参数:
        X -训练数据集 维度(n_x,例子数)
        Y -标签,向量,0为非猫,1为猫,维度(1,例子数量)
        layers_dims -层数的向量,维度为(n_y,n_h,...n_y)
        learning_rate -学习率
    返回:
        parameters -模型学习的参数.然后可以用它进行预测
    """
    np.random.seed(1)
    costs=[]
    
    parameters=initialize_parameters_deep(layers_dims)
    
    for i in range(0,num_iterations):
        AL,caches=L_model_forward(X,parameters)
         
        cost=compute_cost(AL,Y)
        grads=L_model_backward(AL,Y,caches)
        
        parameters=update_parameters(parameters,grads,learning_rate)
        
        if i%100==0:
            costs.append(cost)
            if print_cost:
                print("Number",i,"times cost is ",np.squeeze(cost))
     
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration(pertens)')
        plt.title("Learning rate="+str(learning_rate))
#        plt.show()

    return parameters

def predict(X,Y,parameters):
    """
    该函数用于预测L层神经网络模型的结果
    参数:
        X -测试集
        Y -标签
        parameters -经过训练的模型参数
    返回:
       p -对数据集X的预测
    """
    m=X.shape[1]
    n=len(parameters)//2 #神经网络的层数
    p=np.zeros((1,m))

    probas,caches=L_model_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if  probas[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    
    print("accuracy is "+str(float(np.sum((p==Y))/m)))

    return p

# write parameters in a csv file
def write_parameters_file(filename,parameters):
    with open(filename,mode='w') as fw:
        for key in parameters.keys():
            fw.write(key+'\n')
            np.save(key,parameters[key])

    

# read parameters from a csv file
def read_parameters_file(filename):
    parameters={}
    with open(filename,mode='r') as fr:
        lines=fr.readlines()
        for row in lines:
            row=row.strip('\n')
            ar=np.load(row+'.npy')
            parameters[row]=ar
    
    return parameters

# predict a single picture
def predict_one(filename,is_cat=1):

    if is_cat==1:
        my_label_y=[1]
    else:
        my_label_y=[0]
    image=np.array(scipy.ndimage.imread(filename,flatten=False))
    my_image=scipy.misc.imresize(image,size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    my_predicted_image=predict(my_image,my_label_y,parameters)
    print("y="+str(np.squeeze(my_predicted_image)))
    

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=lr_utils.load_dataset()
train_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
num_px=train_set_x_orig.shape[1]

train_x=train_x_flatten/255
train_y=train_set_y
test_x=test_x_flatten/255
test_y=test_set_y
#n_x=12288
#n_h=7
#n_y=1

#layers_dims=(n_x,n_h,n_y)
#parameters=two_layer_model(train_x,train_set_y,layers_dims=(n_x,n_h,n_y),num_iterations=2500,print_cost=True,isPlot=True)

layers_dims=(12288,20,7,5,1)
#parameters=L_layer_model(train_x,train_y,layers_dims,num_iterations=2500,print_cost=True,isPlot=True)
parameters=read_parameters_file("parameters.txt")
#p_train=predict(train_x,train_y,parameters)
#p_test=predict(test_x,test_y,parameters)

#write_parameters_file("parameters.txt",parameters)
predict_one("cat.jpg")
