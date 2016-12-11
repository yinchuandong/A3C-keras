import tensorflow as tf
import numpy as np
import random
import time

from config import *
from game.game_state import GameState
from redis_queue import RedisQueue
import cPickle


def timestamp():
    return time.time()


class A3CActorThread(object):

    def __init__(self, thread_index, global_network):

        self.thread_index = thread_index
        self.local_network = global_network
        self.game_state = GameState(thread_index)
        self.local_t = 0

        # for log
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.prev_local_t = 0

        self.rq = RedisQueue(REDIS_QUEUE_NAME)
        return

    def choose_action(self, policy_output):
        values = []
        sum = 0.0
        for rate in policy_output:
            sum += rate
            values.append(sum)

        r = random.random() * sum
        for i in range(len(values)):
            if values[i] >= r:
                return i
        return len(values) - 1

    def _record_log(self, sess, global_t, summary_writer, summary_op, reward_input, reward, time_input, living_time):
        summary_str = sess.run(summary_op, feed_dict={
            reward_input: reward,
            time_input: living_time
        })
        summary_writer.add_summary(summary_str, global_t)
        return

    def process(self, sess, global_t, summary_writer, summary_op, reward_input, time_input):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False
        # reduce the influence of socket connecting time
        if self.episode_start_time == 0.0:
            self.episode_start_time = timestamp()

        sess.run(self.reset_gradients)
        # copy weight from global network
        sess.run(self.sync)

        start_local_t = self.local_t

        for i in range(LOCAL_T_MAX):
            policy_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            if self.thread_index == 0 and self.local_t % 1000 == 0:
                print 'policy=', policy_
                print 'value=', value_

            action_id = self.choose_action(policy_)

            states.append(self.game_state.s_t)
            actions.append(action_id)
            values.append(value_)

            self.game_state.process(action_id)
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward
            rewards.append(np.clip(reward, -1.0, 1.0))

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                episode_end_time = timestamp()
                living_time = episode_end_time - self.episode_start_time

                self._record_log(sess, global_t, summary_writer, summary_op,
                                 reward_input, self.episode_reward, time_input, living_time)

                print ("global_t=%d / reward=%.2f / living_time=%.4f") % (global_t, self.episode_reward, living_time)

                # reset variables
                self.episode_reward = 0.0
                self.episode_start_time = episode_end_time
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_lstm_state()
                break
            # log
            if self.local_t % 2000 == 0:
                living_time = timestamp() - self.episode_start_time
                self._record_log(sess, global_t, summary_writer, summary_op,
                                 reward_input, self.episode_reward, time_input, living_time)
        # -----------end of batch (LOCAL_T_MAX)--------------------

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)
        # print ('global_t: %d, R: %f') % (global_t, R)

        states.reverse()
        actions.reverse()
        rewards.reverse()
        values.reverse()

        batch_state = []
        batch_action = []
        batch_td = []
        batch_R = []

        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            action = np.zeros([ACTION_DIM])
            action[ai] = 1

            batch_state.append(si)
            batch_action.append(action)
            batch_td.append(td)
            batch_R.append(R)

            # put in into redis queue for asychronously train
            data = cPickle.dumps((si, action, td, R))
            self.rq.put(data)

        diff_local_t = self.local_t - start_local_t
        return diff_local_t


if __name__ == '__main__':
    # game_state = GameState()
    # game_state.process(1)
    # print np.shape(game_state.s_t)
    print timestamp()
    print time.time()
