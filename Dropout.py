import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
# add output layers
# 在输出结果的时候都要sum 全部weights*xs 的结果, 越多 neurones 越多 weights, sum 越大,
# sum 太大的时候会爆炸,算出来全是 NAN. 所以一种方法是减少 neurones, 一种是初始 weights 时的 standard distribution 缩小点
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)


# the loss between prediction and real data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary
train_writer = tf.summary.FileWriter("C:/Users/jxy01/AppData/Local/Programs/Python/Python36/Scripts/log/train",
                                     sess.graph)
test_writer = tf.summary.FileWriter("C:/Users/jxy01/AppData/Local/Programs/Python/Python36/Scripts/log/test",
                                    sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        # record loss

        # cross_entropy的描述是有包含keep_prob: 0.5
        # 這個隨機的因素在，所以你run兩次會因為兩次隨機的結果不同導致cross_entropy不同，但如果把keep_prob設成1，那就表示保留所以權重，也就是沒有隨機了，所以run一百次cross_entropy應該也都會有一樣的結果，
        # 另外我覺得keep_prob: 0.5
        # 是訓練的時候用的，為了讓權重訓練的更漂亮，
        # 如果是要看訓練完的結果，應該要keep_prob設成1，也就是不要再忽略某些權重了﻿

        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)


