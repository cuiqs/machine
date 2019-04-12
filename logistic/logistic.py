import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py
from lr_utils import load_dataset


def sigmoid(z):
    """
        z为任何大小的标量或numpy数组
    """
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
        创建一个维度为(dim,1)的0向量,并将b初始化为0
    """
    w=np.zeros(shape=(dim,1))
    b=0
    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return (w,b)


def propagate(w,b,X,Y):
    """
    实现前向和后向传播的成本函数及其梯度
    参数:
        w:权重,大小不等的数组(num_px*num_px*3,1)
        b:偏差,一个标量
        X:训练矩阵(num_px*num_px*3,训练样本数量)
        Y:训练结果矩阵(1,训练样本数量)
    返回:
        cost:逻辑回归的负对数似然成本
        dw: 相对于w的损失梯度,与w形状相同
        db: 相对于b的损失梯度,与b形状相同
    """
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))

    dw=(1/m)*np.dot(X,(A-Y).T)
    db=(1/m)*np.sum(A-Y)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())

    grads={
            "dw":dw,
            "db":db
          }
    
    return (grads,cost)

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    通过运行梯度下降法优化w,b
    num_iterations:优化循环的迭代次数
    learning_rate:梯度下降更新规则的学习率
    print_cost:每100步打印一次损失值
    """
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        
        w=w-learning_rate*dw
        b=b-learning_rate*db
        
        if i%100==0:
            costs.append(cost)
        if (print_cost) and (i%100==0):
            print("times:%i, error value:%f"%(i,cost))
    
    params={
            "w":w,
            "b":b}
    grads={
            "dw":dw,
            "db":db}

    return(params,grads,costs)

def predict(w,b,X):
    """
    使用学习逻辑回归参数预测标签是0还是1
    w:权重
    b:偏差,一个标量
    X:数据集
    返回包含对X中所有图片预测的一个向量
    """
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)

    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0

    assert(Y_prediction.shape==(1,m))
    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    """
        调用之前实现的函数构建逻辑回归模型
        X_train:训练集,维度为(num_px*num_px*3,样本数量)
        Y_train:训练标签集,维度(1,样本数量)
        X_test:测试集
        Y_test:测试标签集
        num_iterations:用于优化参数的迭代次数
        learning_rate:更新规则中使用的学习速率的超参数
        print_cost:设置为True时,每100次打印损失函数值
        返回包含模型信息的字典
    """
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w,b=parameters["w"],parameters["b"]
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)

    print("train set accurate:",format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test set accurate:",format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d={
        "costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations}
    return d    
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()
m_train=train_set_y.shape[1]
m_test=test_set_y.shape[1]
num_px=train_set_x_orig.shape[1]
"""
print("count of train set:"+str(m_train))
print("count of test set:"+str(m_test))
print("every picture width/height:"+str(num_px))
print("train set picture dimemsion:"+str(train_set_x_orig.shape))
print("train set tag dimension:"+str(train_set_y.shape))
print("test set picture dimemsion:"+str(test_set_x_orig.shape))
print("test set tag dimension:"+str(test_set_y.shape))
"""
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255
d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=True)
costs=np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("learning_rate="+str(d["learning_rate"]))
plt.show()
#pylab.show()
