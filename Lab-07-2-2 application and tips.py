import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])


x_train = xy[:,0:-1]
y_train = xy[:,[-1]]


# Normalizetion X(max) = X(max) - X(min)/ X-X(min)

def normalization(data):
    numerator = data - np.min(data,0) # 분자
    denominator = np.max(data,0) - np.min(data,0) #분모
    return numerator / denominator



xy = normalization(xy) # 정규화
print(xy)

x_train = xy[:,0:-1]
y_train = xy[:,[-1]]



dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(len(x_train))# Batch size는 한번에 학습시킬 size로 정한다.

W = tf.Variable(tf.random_normal([4,1]),dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype = tf.float32)

# Linear Regression(Wx+b)
def linearReg_fn(features):
    hypthesis = tf.matmul(features,W) +b
    return hypthesis

# weight의 수가 많아지면 수만큼 더한다 tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
def l2_loss(loss,beta = 0.01):
    W_reg = tf.nn.l2_loss(W) # output = sum(t**2)/2
    loss = tf.reduce_mean(loss + W_reg * beta)
    return loss

# flag를 True로 한다면 l2_loss function 호출
def loss_fn(hypthesis,features,labels,flag = False):
    cost = tf.reduce_mean(tf.square(hypthesis - labels))
    if (flag):
        cost = l2_loss(cost)
    return cost

is_decay = True
starter_learning_rate = 0.1

if(is_decay):
    global_step = tf.Variable(0,trainable = False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,50,0.96,staircase = True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = starter_learning_rate)

def grad(hypothesis,features, labels,l2_flag):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(linearReg_fn(features),features,labels,l2_flag)
    return tape.gradient(loss_value,[W,b]), loss_value

EPOCHS = 101
for step in range(EPOCHS):
    for features, labels  in tfe.Iterator(dataset):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads, loss_value = grad(linearReg_fn(features), features, labels, False)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]), global_step=global_step)
        if step % 10 == 0:
            print("Iter: {}, Loss: {:.4f}, Learning Rate: {:.8f}".format(step, loss_value, optimizer._learning_rate()))