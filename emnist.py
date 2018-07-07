# start date : 06/22/2018

# imports
import numpy as np
import scipy.io as sio
import math
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

# load data
data = sio.loadmat('Data/emnist-byclass.mat')

# # preprocessing and obtaining training and testing data and labels

# height and width of each image
h =28
w = 28

# no. of training and testing samples
len_tr = len(data['dataset'][0][0][0][0][0][0])
len_ts = len(data['dataset'][0][0][1][0][0][0])

# extracting data and reshaping images as per our requirements
X_train = data['dataset'][0][0][0][0][0][0].reshape(len_tr,h,w,1)
y_train = data['dataset'][0][0][0][0][0][1]

X_test = data['dataset'][0][0][1][0][0][0].reshape(len_ts,h,w,1)
y_test = data['dataset'][0][0][1][0][0][1]

# print(X_train[0].shape)

# function to rotate images (images in the dataset are flipped and rotated)
def rotate(image):
    flip = np.fliplr(image)
    out = np.rot90(flip)
    return out

# rotate all images to actual orientation
for i in range(len_tr):
    X_train[i] = rotate(X_train[i])

for j in range(len_ts):
    X_test[j] = rotate(X_test[j])

# verifying and checking samples
# test = X_test[1809]
# print(y_test[1809])
# cv2.imshow('test',test)
# cv2.waitKey(0)

# convert data to float32 and labels to int32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# standardize
X_train/= 255
X_test/= 255
# try normalization too later

# transpose y to get a row of labels and squeeze to one list
y_train = y_train.reshape(1,len_tr)[0]
y_test = y_test.reshape(1,len_ts)[0]

# function for one hot encoding
def one_hot(y, indices = 62):
    tot = len(y)
    out = np.zeros((tot,indices))
    out[np.arange(tot),y] = 1
    return out

# one hot to train and test labels
yoh_train = one_hot(y_train)
yoh_test = one_hot(y_test)
# verifying
print(yoh_train.shape)
print(yoh_test.shape)
# print(yoh_train[:5])
# print(yoh_test[:5])
# print(y_train[:5])
# print(y_test[:5])

# Save all of the preprocessed X and y data for later use
np.savez('Data/AllData.npy',
    X_train = X_train, X_test = X_test,
    y_train = y_train, y_test = y_test,
    yoh_train = yoh_train, yoh_test = yoh_test)

# Let the magic begin!
# Load preprocessed data
Data = np.load('Data/AllData.npy.npz')
# print(Data.files)
X_train = Data['X_train']
X_test = Data['X_test']
y_train = Data['yoh_train']
y_test = Data['yoh_test']

# split test set to validation and test sets
Val_size = 20000
X_val = X_test[:Val_size]
y_val = y_test[:Val_size]

X_test = X_test[Val_size:]
y_test = y_test[Val_size:]

# # Verify sizes of matrices
# print(y_train.shape)
# print(y_test.shape)
# print(y_val.shape)
# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

# Create placeholders for x and y
X = tf.placeholder(tf.float32,shape = [None,28,28,1])
y = tf.placeholder(tf.float32,shape = [None,62])

# Required Function definitions
# Weight and Bias initializers
def weight(name,shape):
    # using Xavier initialization
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name,shape = shape,initializer = initial)

def bias(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

# Conv2d, ReLU, Max Pool and Batch Norm
def conv2d(x,W,s):
    return tf.nn.conv2d(x,W,strides = [1,s,s,1],padding = 'SAME')

def ReLU(z):
    return tf.nn.relu(z)

def max_pool(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x,center = True,scale = True)

# To divide the data into minibatches for Minibatch Gradient Descent
def batch(X,Y,batch_size,seed = 0):
    np.random.seed(seed)
    m = X.shape[0] #total no of images
    batches = []
    # Shuffle data
    perm = list(np.random.permutation(m))
    shuffled_X = X[perm,:]
    shuffled_Y = Y[perm,:]
    # Partition (shuffled_X, shuffled_Y)
    num_minibatches = math.floor(m/batch_size) # number of mini batches of required size in our partitioning
    for k in range(0, num_minibatches):
        mini_batch_X = shuffled_X[(batch_size*k):(batch_size*(k+1)),:]
        mini_batch_Y = shuffled_Y[(batch_size*k):(batch_size*(k+1)),:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)
    # Handling the end case (if size (last mini-batch) < batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[(batch_size*(k+1)):m, :]
        mini_batch_Y = shuffled_Y[(batch_size*(k+1)):m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)
    return batches

# Layers
# Conv_1
W_c1 = weight('W_c1',[5,5,1,32])
b_c1 = bias([32])
z_c1 = conv2d(X,W_c1,1) + b_c1
h_c1 = ReLU(z_c1)
# print(h_c1.shape)

# Conv_2
W_c2 = weight('W_c2',[5,5,32,64])
b_c2 = bias([64])
z_c2 = conv2d(h_c1,W_c2,2) + b_c2
h_c2 = ReLU(z_c2)
# print(h_c2.shape)

# Conv_3
W_c3 = weight('W_c3',[5,5,64,128])
b_c3 = bias([128])
z_c3 = conv2d(h_c2,W_c3,2) + b_c3
h_c3 = ReLU(z_c3)
# print(h_c3.shape)

# Batch_Norm_1
h_b1 = batch_norm(h_c3)
# Max_Pool_1
h_p1 = max_pool(h_b1)
# print(h_p1.shape)

# obtain shape of previous layer for Dense layer
_,l,w,c = h_p1.get_shape().as_list()
# Flatten output of previous layer for fully connected layers
h_flat = tf.reshape(h_p1,[-1,l*w*c])

# FC_1 (Dense)
W_fc1 = weight('W_fc1',[l*w*c,784])
b_fc1 = bias([784])
z_fc1 = tf.matmul(h_flat,W_fc1) + b_fc1
h_fc1 = ReLU(z_fc1)
# print(h_fc1.shape)

# Batch_Norm_2
h_b2 = batch_norm(h_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_d = tf.nn.dropout(h_b2,keep_prob)

# FC_2
W_fc2 = weight('W_fc2',[784,62])
b_fc2 = bias([62])
z_fc2 = tf.matmul(h_d,W_fc2) + b_fc2
# Prediction (softmax will be applied at cross entropy step)
y_pred = z_fc2
# print(y_pred.shape)

# Hyperparameters (TUNABLE)
Batch_size = 128
iter = 101
drop_prob = 0.5

# Learning rate with exponential decay
global_step = tf.Variable(0, trainable=False)
initial_lr = 0.1
Decay_rate = 0.96
Decay_steps = 10
learning_rate = tf.train.exponential_decay(initial_lr, global_step, Decay_steps, Decay_rate, staircase=True)

# Evaluation metrics
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels = y,logits = y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
prediction = tf.argmax(y_pred,1)
correct_pred = tf.equal(prediction,tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Some required variables
m = X_train.shape[0]
seed = 3 # for random batches
losses = []

# Start the Session (Computation graph)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Start the training process
for i in range(iter):
    minibatch_loss = 0
    num_batches = int(m/Batch_size)
    seed+=1 #to ensure the shuffling doesn't happen using the same permutation for all iterations
    minibatches = batch(X_train,y_train,Batch_size,seed) #getting (m/Batch_size) minibatches of size Batch_size

    # iterating over all minibatches
    for minibatch in minibatches:
        (X_mb,y_mb) = minibatch

        # evaluate loss for current iteration (SLIGHTLY UNSURE ABOUT WHATS HAPPENING HERE)
        _,temp_loss = sess.run([optimizer,cross_entropy],
            feed_dict = {X:X_mb,y:y_mb,keep_prob:drop_prob})
        minibatch_loss += temp_loss/num_batches

    # print accuracy and loss every 10 iterations
    losses.append(temp_loss)
    train_accuracy = accuracy.eval(
        feed_dict = {X:X_mb,y:y_mb,keep_prob:1.0})
    print('Epoch %d, Loss %g, Training Accuracy %g'
        %(i,loss,train_accuracy))

# Validation Accuracy
print('Validation Accuracy %g'%accuracy.eval(
    feed_dict = {X:X_val,y:y_val, keep_prob:1.0}))
# Visualize stats
plt.plot(np.squeeze(losses))
plt.ylabel('cost')
plt.xlabel('iteration (tens)')
plt.title('loss curve')
plt.show()

# Test Accuracy
print('Testing Accuracy %g'%accuracy.eval(
    feed_dict = {X:X_test,y:y_test, keep_prob:1.0}))

# end session
sess.close()
