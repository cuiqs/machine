
"""
tensorflow 入门
"""

import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import time
import tf_utils

np.random.seed(1)

def linear_function():
	"""
		initialize W(4,3),X(3,1),b(4,1)
	return result is Y=WX+b
	"""
	np.random.seed(1)
	
	X=np.random.randn(3,1)
	W=np.random.randn(4,3)
	b=np.random.randn(4,1)

	Y=tf.add(tf.matmul(W,X),b)

	sess=tf.Session()
	result=sess.run(Y)
	sess.close()

	return result

def sigmoid(z):

	x=tf.placeholder(tf.float32,name="x")
	sigmoid=tf.sigmoid(x)

	with tf.Session() as sess:
		result=sess.run(sigmoid,feed_dict={x:z})

	return result

#print("sigmoid(12)="+str(sigmoid(12)))

def one_hot_matrix(lables,C):
	"""
	creat a matrix,in it if number j sample match number j label then entry(i,j)=1
	"""
	C=tf.constant(C,name="C")
	one_hot_matrix=tf.one_hot(indices=labels,depth=C,axis=0)

	with tf.Session() as sess:
		one_hot=sess.run(one_hot_matrix)

	return one_hot

"""
labels=np.array([1,2,3,0,2,1])
one_hot=one_hot_matrix(labels,C=4)
print(str(one_hot))
"""

def ones(shape):
	ones=tf.ones(shape)
	sess=tf.Session()
	one=sess.run(ones)
	sess.close()
	return ones

def create_placeholder(n_x,n_y):
	"""
		为tensorflow创建占位符
	参数：
	  n_x -一个实数，图片向量的大小（64*64*3=12288)
	  n_y -一个实数，分类数（从0到5）
	返回：
	  X -一个数据输入的占位符，维度(n_x,none),dtype="float"
	  Y -一个对应输入的标签的占位符，维度（n_y,none),dtype="float"
	 使用none可以灵活的处理占位符提供的样本数量。
	"""
	X=tf.placeholder(tf.float32,[n_x,None],name="X")
	Y=tf.placeholder(tf.float32,[n_y,None],name="Y")

	return X,Y

def initialize_parameters():
	tf.set_random_seed(1)
	
	W1=tf.get_variable("W1",[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	b1=tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer())
	W2=tf.get_variable("W2",[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	W3=tf.get_variable("W3",[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
	b2=tf.get_variable("b2",[12,1],initializer=tf.zeros_initializer())
	b3=tf.get_variable("b3",[6,1],initializer=tf.zeros_initializer())

	parameters={"W1":W1,"W2":W2,"W3":W3,"b1":b1,"b2":b2,"b3":b3}

	return parameters

#Test initialize_parameters()
"""
tf.reset_default_graph()
with tf.Session() as sess:
	parameters=initialize_parameters()
	print("W1="+str(parameters["W1"]))
	print("W2="+str(parameters["W2"]))
	print("b1="+str(parameters["b1"]))
	print("b2="+str(parameters["b2"]))
"""

def forward_propagation(X,parameters):
#LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
	W1=parameters['W1']
	b1=parameters['b1']
	W2=parameters['W2']
	b2=parameters['b2']
	W3=parameters['W3']
	b3=parameters['b3']

	Z1=tf.add(tf.matmul(W1,X),b1)
	A1=tf.nn.relu(Z1)
	Z2=tf.add(tf.matmul(W2,A1),b2)
	A2=tf.nn.relu(Z2)
	Z3=tf.add(tf.matmul(W3,A2),b3)

	return Z3

#Test forward_propagation()
"""
tf.reset_default_graph()
with tf.Session() as sess:
	X,Y=create_placeholder(12288,6)
	parameters=initialize_parameters()
	Z3=forward_propagation(X,parameters)
	print('Z3='+str(Z3))
"""

def compute_cost(Z3,Y):
#Z3 is result from forward_propagation,Y is tag
	logits=tf.transpose(Z3)
	labels=tf.transpose(Y)

	cost=tf.reduce_mean(tf.nn.sotfmax_cross_entropy_with_logits(logits=logits,labels=labels))
	return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=1500,minibatch_size=32,print_cost=True,is_plot=True):
	"""
		实现3层神经网络
		num_epochs -整个训练集的遍历次数
	"""
	ops.reset_default_graph()
	tf.set_random_seed(1)
	seed=3
	(n_x,m)=X_train.shape
	cost=[]
	
	X,Y=create_placeholder(n_x,n_y)
	parameters=initialize_parameters()
	Z3=forward_propagation(X,parameters)
	cost=compute(Z3,Y)

#backward propagation,optimize with adam
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init=tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		
		for epoch in range(num_epochs):
			epoch_cost=0
			num_minibatches=int(m/minibatch_size)
			seed=seed+1
			minibatches=tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

				for minibatch in minibatches:
					(minibatch_X,minibatch_Y)=minibatch
					minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
					epoch_cost=epoch_cost+minibatch_cost/num_minibatches


			if epoch%5==0:
				cost.append(epoch_cost)
				if print_cost==True and epoch%100==0:
					print("epoch="+str(epoch)+"  epoch_cost="+str(epoch_cost))

		parameters=sess.run(parameters)#save learned parameters

		correct_prediction=tf.equal(tf.argmax(Z3),tf.argmax(Y))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

		print("Train set accuracy:",accuracy.eval({X:X_train,Y:Y_train}))
		print("Test set accuracy:",accuracy.eval({X:X_test,Y:Y_test}))

	return parameters	
		

		
	
		


X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=tf_utils.load_dataset()
X_train_flatten=X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten=X_test_orig.reshape(X_test_orig.shape[0],-1).T
#make every element between 1 and 0
X_train=X_train_flatten/255
X_test=X_test_flatten/255

Y_train=tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test=tf_utils.convert_to_one_hot(Y_test_orig,6)

