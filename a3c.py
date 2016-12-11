import os
import sys

import tensorflow as tf
import numpy as np
import math
import threading
import signal

from a3c_network import A3CFFNetwork, A3CLSTMNetwork
from a3c_actor_thread import A3CActorThread
from config import *

from redis_queue import RedisQueue
import cPickle


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


class A3C(object):

    def __init__(self):
        self.device = '/gpu:0' if USE_GPU else '/cpu:0'
        self.stop_requested = False
        self.global_t = 0
        if USE_LSTM:
            self.global_network = A3CLSTMNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, self.device, -1)
        else:
            self.global_network = A3CFFNetwork(STATE_DIM, STATE_CHN, ACTION_DIM, self.device)
        self.global_network.create_loss(ENTROPY_BETA)

        self.initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE)
        print 'initial_learning_rate:', self.initial_learning_rate
        self.learning_rate_input = tf.placeholder('float')
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_input,
                                                   decay=RMSP_ALPHA, momentum=0.0, epsilon=RMSP_EPSILON)

        grads_and_vars = self.optimizer.compute_gradients(
            self.global_network.total_loss, self.global_network.get_vars())
        self.apply_gradients = self.optimizer.apply_gradients(grads_and_vars)

        self.actor_threads = []
        for i in range(PARALLEL_SIZE):
            actor_thread = A3CActorThread(i, self.global_network)
            self.actor_threads.append(actor_thread)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

        self.reward_input = tf.placeholder(tf.float32)
        tf.scalar_summary('reward', self.reward_input)

        self.time_input = tf.placeholder(tf.float32)
        tf.scalar_summary('living_time', self.time_input)

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(LOG_FILE, self.sess.graph)

        self.saver = tf.train.Saver()
        self.restore()

        self.lock = threading.Lock()
        self.rq = RedisQueue(REDIS_QUEUE_NAME)
        return

    def restore(self):
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print(">>> global step set: ", self.global_t)
        else:
            print("Could not find old checkpoint")
        return

    def backup(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step=self.global_t)
        return

    def predict_function(self, parallel_index, lock):
        actor_thread = self.actor_threads[parallel_index]
        while True:
            if self.stop_requested or (self.global_t > MAX_TIME_STEP):
                break
            diff_global_t = actor_thread.process(
                self.sess, self.global_t,
                self.summary_writer, self.summary_op,
                self.reward_input, self.time_input
            )

            self.global_t += diff_global_t
            if self.global_t % 1000000 < LOCAL_T_MAX:
                self.backup()
            # print 'global_t:', self.global_t
        return

    def train_function(self, lock):
        batch_state = []
        batch_action = []
        batch_td = []
        batch_R = []

        train_count = 0
        while True:
            if self.stop_requested or (self.global_t > MAX_TIME_STEP):
                break
            data = self.rq.get()
            (state, action, td, R) = cPickle.loads(data)

            batch_state.append(state)
            batch_action.append(action)
            batch_td.append(td)
            batch_R.append(R)

            if len(batch_R) < 32:
                continue

            self.sess.run(self.apply_gradients, feed_dict={
                self.global_network.state_input: batch_state,
                self.global_network.action_input: batch_action,
                self.global_network.td: batch_td,
                self.global_network.R: batch_R,
                self.learning_rate_input: self.initial_learning_rate
            })

            batch_state = []
            batch_action = []
            batch_td = []
            batch_R = []

            train_count += 1
            print 'train_count:', train_count
        return

    def signal_handler(self, signal_, frame_):
        print 'You pressed Ctrl+C !'
        self.stop_requested = True
        return

    def run(self):
        predict_treads = []
        for i in range(PARALLEL_SIZE):
            predict_treads.append(threading.Thread(target=self.predict_function, args=(i, self.lock)))

        signal.signal(signal.SIGINT, self.signal_handler)

        for t in predict_treads:
            t.start()

        train_thread = threading.Thread(target=self.train_function, args=(self.lock, ))
        train_thread.start()

        print 'Press Ctrl+C to stop'
        signal.pause()

        print 'Now saving data....'
        for t in predict_treads:
            t.join()

        self.backup()
        return


if __name__ == '__main__':
    print 'a3c.py'
    net = A3C()
    net.run()
