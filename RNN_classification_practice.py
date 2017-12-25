import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001               # learning rate
trainning_iters = 100000 # 迭代次数
batch_size = 128


n_inputs = 28        # MNIST data input (img shape: 28x28) 每次input 一行28个像素
n_steps = 28         # time steps 输入28次 28行
n_hidden_units = 128  # neurons in hidden layer 自己设置的
n_classes = 10       # MNIST classes(0-9)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # 128
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # 10
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weight, biases):
    # hidden layer for input to cell
    # X(128batch,28 steps,28 inputs)
    # 先转换成 ==>(128*28, 28)
    X = tf.reshape(X, [-1, n_inputs])
    # for input hidden layer enter the cell
    #  ==>(128batch*28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in'])+biases['in']
    # 再转成三维的 (weights 是二维)
    # ==>(128batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # state_is_tuple=True 会生成状态元组， lstm cell is divided into two parts (c_state,m_state) c 主线state m 分线
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # 运算
    # dynamic rnn 效果更好 time_major :step
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    # states[1] = outputs[-1]
    # results = tf.matmul(states[1], weights['out'])+biases['out']
    # transpose 维度交换 [0 batch_size,1 step,2 output_size] ->[1 step,0 batch_size,2 output_size]
    # unstack 将第一维度转换为列向量 list [(batch, outputs)] * step (28 个) 取最后一个output
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out'])+biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < trainning_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)       # batch_size 批处理样本数 每次（一批）处理128个
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])  # 把batch 重构成每个28x28 的图片 然后128个
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1

