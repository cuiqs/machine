
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils #part one initialization
import reg_utils #part two regulation
import gc_utils  #part three grads test

plt.rcParams['figure.figsize']=(7.0,4.0) #set default size of plots
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

train_X,train_Y,test_X,test_Y=init_utils.load_dataset(is_plot=False)
#plt.show()

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization='he',is_plot=True):
    """
        实现一个三层网络:LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    参数:
        X -输入的数据,维度(2,训练/测试的数量)
        Y -标签 [0|1],维度(1,训练/测试的数量)
        learning_rate -学习速率
        num_iterations -迭代次数
        print_cost -是否打印成本值,每迭代1000次打印一次
        initialization -字符串类型,[zeros|random|he]
        is_plot -是否绘制梯度下降的曲线图
    返回:
        parameters -学习后的参数
    """
    grads={}
    costs=[]
    m=X.shape[1]
    layers_dims=[X.shape[0],10,5,1]

    if initialization=="zeros":
        parameters=initialize_parameters_zeros(layers_dims)
    elif initialization=="random":
        parameters=initialize_parameters_random(layers_dims)
    elif initialization=="he":
        parameters=initialize_parameters_he(layers_dims)
    else:
        print("Fault initializtion argument! quit")
        exit
    
    for i in range(0,num_iterations):
        a3,cache=init_utils.forward_propagation(X,parameters)
        cost=init_utils.compute_loss(a3,Y)
        grads=init_utils.backward_propagation(X,Y,cache)
        parameters=init_utils.update_parameters(parameters,grads,learning_rate)
        
        if i%1000==0:
            costs.append(cost)
            if print_cost:
                print("number times"+str(i)+" cost is "+str(cost))
    
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per hundreds)') 
        plt.title('learing_rate:'+str(learning_rate))
        plt.show()
    
    return parameters

def initialize_parameters_zeros(layers_dims):
    """
        All arguments of model be set zero
    Arguments:
        layers_dims -list,model's layers and number of layer's node
    return:
        parameters - dict including all W and b
    """
    parameters={}
    L=len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)]=np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

        assert(parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape==(layers_dims[l],1))

    return parameters

def initialize_parameters_random(layers_dims):
    """
        All arguments of model be set random
    Arguments:
        layers_dims -list,model's layers and number of layer's node
    return:
        parameters - dict including all W and b
    """
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

        assert(parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape==(layers_dims[l],1))

    return parameters

def initialize_parameters_he(layers_dims):
    """
        All arguments of model be set random
    Arguments:
        layers_dims -list,model's layers and number of layer's node
    return:
        parameters - dict including all W and b
    """
    np.random.seed(3)
    parameters={}
    L=len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

        assert(parameters["W"+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert(parameters["b"+str(l)].shape==(layers_dims[l],1))

    return parameters
"""
parameters=initialize_parameters_random([3,2,1])
print("W1"+str(parameters["W1"]))    
print("b1"+str(parameters["b1"]))    
print("W2"+str(parameters["W2"]))    
print("b2"+str(parameters["b2"]))    
"""
"""
parameters=model(train_X,train_Y,initialization="he",is_plot=True)
print("train set:")
predictions_train=init_utils.predict(train_X,train_Y,parameters)
print("test set:")
predictions_test=init_utils.predict(test_X,test_Y,parameters)
print(predictions_train)
print(predictions_test)

plt.title("Model with large random initialization")
axes=plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters,x.T),train_X,train_Y)
"""

train_X,train_Y,test_X,test_Y=reg_utils.load_2D_dataset(is_plot=True)
plt.show()

