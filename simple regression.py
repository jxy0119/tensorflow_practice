import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
### 定义神经层 x_data 只有一个特征 所以输入神经元个数为1 隐藏层神经元设置为10 输出层神经元设置为1

xs = tf.placeholder(tf.float32, [None, 1])# [None,1] None 表示给多少个例子都是可以的 1 表示xs 是特征数量为1
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

plt.ion()

for i in range(1000):
    _, l, pred = sess.run([train_step, loss, prediction], feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        plt.cla()
        plt.scatter(x_data,y_data)
        plt.plot(x_data, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()



