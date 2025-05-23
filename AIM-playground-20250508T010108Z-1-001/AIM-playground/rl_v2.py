import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

# Hyper Parameters
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 40000
BATCH_SIZE = 32

KERNEL_INITIALIZER = None
CONST_INITIALIZER = None
REGULARIZER = None


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32) # this is where we store transitions (s, a, r, s_)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.compat.v1.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True) #seelf.a is action using the main actor
            a_ = self._build_a(self.S_, scope='target', trainable=False) #a_ is action using the target actor for the next state
        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)  #q is the q value using the main critic for the current state
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False) #q_ is the q value using the target critic + actor for the next state

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.compat.v1.assign(ta, (1 - TAU) * ta + TAU * ea), tf.compat.v1.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)

        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(input_tensor=q)    # maximize the critic's q and take actions that lead to hig rew  
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net1 = tf.compat.v1.layers.dense(s, 256, activation=tf.nn.elu, name='l1', kernel_initializer=KERNEL_INITIALIZER, bias_initializer=CONST_INITIALIZER, trainable=trainable)
            a = tf.compat.v1.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, name='a', kernel_initializer=KERNEL_INITIALIZER, bias_initializer=CONST_INITIALIZER, trainable=trainable, kernel_regularizer=REGULARIZER)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 512
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.elu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable, kernel_initializer=KERNEL_INITIALIZER, kernel_regularizer=REGULARIZER)  # Q(s,a)

    def save(self):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, 'models/params', write_meta_graph=True, write_state=True)
        print('Model checkpoint saved')

    def restore(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, 'models/params')
