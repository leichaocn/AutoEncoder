# region 载入库,加载数据,参数配置
import numpy as np
#需要使用sklearn.preprocessing其中的StandardScaler()来进行数据标准化
import sklearn.preprocessing as prep 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
#训练20轮
training_epochs = 20
#每轮训练中，batch大小为128
batch_size = 128
#每轮训练中，batch_num即batch的总个数,429个。
batch_num = int(mnist.train.num_examples / batch_size)
#每隔display_step轮，显示一次cost
display_step = 0
# endregion

# region 构造计算图
#实现一个标准均匀分布的Xaiver初始化器,功能是让权重被初始化到一个合适的范围，即均值为0，方差为2/(n_in+n_out)。
#fan_in表示输入节点的个数，fan_out表示输出节点的个数。
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    #返回一个均匀分布，shape为[fan_in,fan_out],范围为（low，high），均值为0，方差为2/(fan_in+fan_out)
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

#定义去噪自编码器，包括神经网络的结构、函数定义
class AdditiveGaussianNoiseAutoencoder(object):
    #初始化
    #n_input为初始化变量个数；n_hidden为隐层节点个数；transfer_function为隐层激活函数，默认为softplus;选用Adam优化器；设置高斯噪声系数为0.1。
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        #给输入向量x加噪声，表示为self.x + scale * tf.random_normal((n_input,))
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']),
                self.weights['b1']))
        #decoder.即把encoder后的self.hidden解码，不需要激活。
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # cost
        #self.x是输入，self.re是结果，即求平方差的和，E(self.re-self.x)^2。 
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initialize_weights(self):
        all_weights = dict()
        #shape=[n_inpute,n_hidden]
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        #shape=[n_hidden,n_inpute],由于未使用激活函数，因此W2和b2全部初始化为0
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
     
    #用于训练集求cost。且self.training_scale为高斯噪声系数，缺省值已在初始化中设为0.1
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,
                                                                            self.scale: self.training_scale})
        return cost

    #用于测试集上评测阶段。只求cost，不触发训练。
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
                                                     self.scale: self.training_scale})
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
                                                       self.scale: self.training_scale
                                                       })

    def generate(self, hidden):
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

    #用于XX
    def reconstruct(self, X): 
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
                                                               self.scale: self.training_scale})
    
    #获取隐含层的W1
    def getWeights1(self):
        return self.sess.run(self.weights['w1'])
    #获取隐含层的b1
    def getBiases1(self):
        return self.sess.run(self.weights['b1'])
    #获取隐含层的W2
    def getWeights2(self):
        return self.sess.run(self.weights['w2'])
    #获取隐含层的b2
    def getBiases2(self):
        return self.sess.run(self.weights['b2'])

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                               scale = 0.01)
# endregion

# region 执行计算图
#对数据进行标准化，即让数据变成均值为0，标准差为1的分布。数据标准化主要功能就是消除变量间的量纲关系，从而使数据具有可比性。
def standard_scale(X_train, X_test):
    #在训练集上fit
    preprocessor = prep.StandardScaler().fit(X_train)
    #保证训练数据和测试数据使用相同的scaler
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
#mnist.train.images.shape=[55000,784],mnist.test.images.shape=[10000,784]
#标准化后，shape不变。即X_train.shape=[55000,784],X_test.shape=[10000,784]
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

def get_random_block_from_data(data, batch_size):
    #取一个在(0, len(data) - batch_size)范围内的随机整数。
    start_index = np.random.randint(0, len(data) - batch_size)
    #返回一个data,start_index ???????????????属于不放回抽样。
    return data[start_index:(start_index + batch_size)]
#把所有的epoch都循环一遍。
for epoch in range(training_epochs):
    total_cost = 0.
    # 在每一个epoch上，共计有batch_num个batch，累加求平均得出总cost。
    for i in range(batch_num):
        #从X_train(55000)中随机选了batch_size个(128个)样本，组成batch_xs，shape为[128,784]
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        total_cost+=cost
    #显示total cost in train
    if epoch % (display_step+1) == 0:
        #epoch是从0到（training_epochs-1）。
        print("Epoch:", '%03d' % (epoch + 1), "  Total cost in train :", "{:.9f}".format(total_cost))
print('-------------------------------')
print("Finally,Total cost in test:" + str(autoencoder.calc_total_cost(X_test)))

# endregion

'''
print('-------------------------------')
X_testfunc=get_random_block_from_data(X_train, 5)
print('X_testfunc=',X_testfunc.shape)
print('-------------------------------')
WW1=autoencoder.getWeights1()
print ('W1=',WW1.shape)
bb1=autoencoder.getBiases1()
print ('b1=',bb1.shape)
print('-------------------------------')
bottle_neck =autoencoder.transform(X_testfunc)
print ('bottle_neck=',bottle_neck.shape)
print('-------------------------------')
WW2=autoencoder.getWeights2()
print ('W2=',WW2.shape)
bb2=autoencoder.getBiases2()
print ('b2=',bb2.shape)
print('-------------------------------')
#因为generate()函数默认送入hidden的值，所有前面的placeholder（包括x）就没用了，所以不用输入one_X来feed占位符。
final_StartFromBN =autoencoder.generate()#bottle_neck)
print ('final_StartFromBN=',final_StartFromBN.shape)
final_StartFromX=autoencoder.reconstruct(X_testfunc)
print ('final_StartFromX=',final_StartFromX.shape)
'''
