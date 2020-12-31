import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
colnames = ['Users', 'Activity', 'Timestamp', 'x-axis', 'y-axis', 'z-axis']
dataset = pd.read_csv('WISDM_ar_v1.1_raw.txt', header=None, names=colnames, comment=';')
dataset = dataset.dropna()



N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(dataset) - N_TIME_STEPS, step):
    xs = dataset['x-axis'].values[i: i + N_TIME_STEPS]
    ys = dataset['y-axis'].values[i: i + N_TIME_STEPS]
    zs = dataset['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(dataset['Activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

print("reduced size of data", np.array(segments).shape)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
print (reshaped_segments.dtype)

print("Reshape the segments", np.array(reshaped_segments).shape)

X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.2,
                                                    random_state=RANDOM_SEED)

print (reshaped_segments.shape)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

X_train = X_train.reshape(-1, 3)
print (X_train.shape)
print (X_train[0])

X_test = X_test.reshape(-1, 3)
print (X_test.shape)

y_train = y_train.reshape(-1, 6)
print (y_train.shape)

y_test = y_test.reshape(-1, 6)
print (y_test.shape)

X_train = X_train.reshape(-1, 200, 3, 1)
print (X_train.shape)

X_test = X_test.reshape(-1, 200, 3, 1)
print (X_test.shape)

total = X_train.shape[0]
batch_size = 64
each_epoch = int(total/batch_size)

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        start = (int)(start)
        end = (int)(end)
        yield [d[start:end] for d in data]


gen_data_by_batch = batch_generator([X_train, y_train], batch_size)





def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], \
                          padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 200, 3, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 6])



# Input Layer
x_image = tf.reshape(x, [-1, 200, 3, 1])

# Convolutional Layer #1
W_conv1 = weight_variable([5, 1, 1, 50])
b_conv1 = bias_variable([50])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print ("hhh", h_pool1.shape)
# Convolutional Layer #2
W_conv2 = weight_variable([5, 1, 50, 50])
b_conv2 = bias_variable([50])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Convolutional Layer #3
W_conv3 = weight_variable([5, 1, 50, 50])
b_conv3 = bias_variable([50])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

print ("123", h_pool3.shape)

# FC Layer
W_fc1 = weight_variable([25 * 3 * 50, 128])
b_fc1 = bias_variable([128])

h_pool3_flat = tf.reshape(h_pool3, [-1, 25 * 3 * 50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout Regularization
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#print ("1222", h_fc1.shape)
# FC Layer #2
W_fc2 = weight_variable([128, 6])
b_fc2 = bias_variable([6])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#print ("1333", y_conv.shape)
# Softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
                                   (labels=y_, logits=y_conv))

# Training and testing
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_steps = each_epoch*5
print(num_steps)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num=0
    print("train sss")
    for i in range(num_steps):
        if (i % 5000) == 0:
            print (i)

        X, Y = next(gen_data_by_batch)
        if (i % each_epoch) == 0:
            num = num + 1
            train_accuracy = accuracy.eval(feed_dict={x: X, y_: Y, \
                                                  keep_prob: 1.0})
            print("step %d, training accuracy %g" % (num, train_accuracy))

        train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.7})
        #print("i end")
    print("train end")

    print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, \
                                                        y_: y_test, keep_prob: 1.0}))
    print("test end")



print("all end")


