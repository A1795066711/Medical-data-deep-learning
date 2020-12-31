# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:47:40 2018

@author: ji
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

if __name__ == "__main__":
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv("WISDM_ar_v1.1_raw.txt", header=None, names=column_names)
    n = 10
    print (df.head(n))
    subject = pd.DataFrame(df["user-id"].value_counts(), columns=["Count"])
    subject.index.names = ['Subject']
    print (subject.head(n))
    activities = pd.DataFrame(df["activity"].value_counts(), columns=["Count"])
    activities.index.names = ['Activity']
    print (activities.head(n))
    activity_of_subjects = pd.DataFrame(df.groupby("user-id")["activity"].value_counts())
    print (activity_of_subjects.unstack().head(n))
    activity_of_subjects.unstack().plot(kind='bar', stacked=True, colormap='Blues', title="Distribution")
    plt.show()
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 绘图使用 ggplot 的 style
plt.style.use('ggplot')

# 加载数据集
file_path = 'WISDM_ar_v1.1_raw.txt'#不知道有木有用
def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    #data = pd.read_csv(file_path, header=None, names=column_names)    
    data = pd.read_csv('WISDM_ar_v1.1_raw.txt', header=None, names=column_names)    
    return data

# 数据标准化
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

# 绘图
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

# 为给定的行为画出一段时间（180 × 50ms）的波形图
def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

dataset = read_data('WISDM_ar_v1.1_raw.txt')
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

# 为每个行为绘图
for activity in np.unique(dataset["activity"]):
    subset = dataset[dataset["activity"] == activity][:180]
    plot_activity(activity, subset)

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf

# 加载数据集
def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    #data = pd.read_csv(file_path, header=None, names=column_names)
    data = pd.read_csv('WISDM_ar_v1.1_raw.txt', header=None, names=column_names)
    return data

# 数据标准化
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

# 创建时间窗口，90 × 50ms，也就是 4.5 秒，每次前进 45 条记录，半重叠的方式。
def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)

# 创建输入数据，每一组数据包含 x, y, z 三个轴的 90 条连续记录，
# 用 `stats.mode` 方法获取这 90 条记录中出现次数最多的行为
# 作为该组行为的标签，这里有待商榷，其实可以完全使用同一种行为的数据记录
# 来创建一组数据用于输入的。
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    print (len(data['timestamp']))
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        print (count)
        count += 1
        start = int(start)
        end = int(end) #下面的函数需要取整
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(dataset['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

# 初始化神经网络参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化神经网络参数
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# 执行卷积操作
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

# 为输入数据的每个 channel 执行一维卷积，并输出到 ReLU 激活函数
def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))

# 在卷积层输出进行一维 max pooling
def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')

dataset = read_data('WISDM_ar_v1.1_raw.txt')
dataset['x-axis'] = feature_normalize(dataset['x-axis'])
dataset['y-axis'] = feature_normalize(dataset['y-axis'])
dataset['z-axis'] = feature_normalize(dataset['z-axis'])

segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
# 创建输入
reshaped_segments = segments.reshape(len(segments), 1,90, 3)

# 在准备好的输入数据中，分别抽取训练数据和测试数据，按照 70/30 原则来做。
#train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

# 定义输入数据的维度和标签个数
input_height = 1
input_width = 90
num_labels = 6
num_channels = 3

batch_size = 10
kernel_size = 60
depth = 60

# 隐藏层神经元个数
#num_hidden = 1000
num_hidden = 1800

learning_rate = 0.0001

# 降低 cost 的迭代次数
training_epochs = 8

total_batchs = reshaped_segments.shape[0] // batch_size

# 下面是使用 Tensorflow 创建神经网络的过程。
X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
p = apply_max_pool(c,20,2)
c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)

# 开始训练
with tf.Session() as session:
    tf.initialize_all_variables().run()
    # 开始迭代
    for epoch in range(training_epochs):
        for b in range(total_batchs):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)
        print ("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
            epoch, c, session.run(accuracy, feed_dict={X: train_x, Y: train_y})))
    y_p = tf.argmax(y_, 1)
    y_true = np.argmax(test_y, 1)
    final_acc, y_pred = session.run([accuracy, y_p], feed_dict={X: test_x, Y: test_y})
    print ("Testing Accuracy: {}".format(final_acc))
    temp_y_true = np.unique(y_true)
    temp_y_pred = np.unique(y_pred)
    np.save("y_true", y_true)
    np.save("y_pred", y_pred)
    print ("temp_y_true", temp_y_true)
    print ("temp_y_pred", temp_y_pred)
    # 计算模型的 metrics
    print ("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print ("Recall", recall_score(y_true, y_pred, average='weighted'))
    print ("f1_score", f1_score(y_true, y_pred, average='weighted'))
    print ("confusion_matrix")
    print (confusion_matrix(y_true, y_pred))

    
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

plt.style.use('ggplot')

def plot_confusion_matrix(cm, title='Normalized Confusion matrix', cmap=plt.cm.get_cmap("Blues")):
    cm = cm / cm.astype(np.float).sum(axis=1)
    print ("confusion_matrix: \n{}".format(cm))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, [1, 2, 3, 4, 5, 6], rotation=45)
    plt.yticks(tick_marks, [1, 2, 3, 4, 5, 6])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_metrics(y_test, predicted):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
    predicted = label_binarize(predicted, classes=[0, 1, 2, 3, 4, 5])

    n_classes = y_test.shape[1]

    print ("n_classes:", n_classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def test_har():
    y_true = np.load("y_true.npy")
    y_pred = np.load("y_pred.npy")
    print ("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print ("Recall", recall_score(y_true, y_pred, average='weighted'))
    print ("f1_score", f1_score(y_true, y_pred, average='weighted'))
    print ("confusion_matrix")
    print (confusion_matrix(y_true, y_pred))
    plot_metrics(y_true, y_pred)
    plot_confusion_matrix(confusion_matrix(y_true, y_pred))

    if __name__ == "__main__":
        test_har()