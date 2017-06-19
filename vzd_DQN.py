#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vizdoom import *
import itertools as it
import random as rnd
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import signal

# Q-LEARNING SETTINGS
LEARNING_RATE = 0.00025
# LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.90
EPOCHS = 20
LEARNING_STEPS_PER_EPOCH = 2000
MEMORY_CAPACITY = 10000

# NN LEARNING SETTINGS
BATCH_SIZE = 64

# TRAINING REGIME
TEST_EPISODES_PER_EPOCH = 100

# OTHER PARAMETERS
FRAME_REPEAT = 12
RESOLUTION = (32, 24)
#RESOLUTION = (84, 84)
EPISODES_TO_WATCH = 10

MODEL_SAVEFILE = "./ckpt/model.ckpt"

LOAD_MODEL = False
SKIP_LEARNING = False
TRAIN_WITH_TEST = True
# Configuration file path
CONFIG_FILE_PATH = "./scenarios/deadly_corridor.cfg"


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

class Util:
# Converts and down-samples the input image
    @staticmethod
    def preprocess(img):
        img = skimage.transform.resize(img, RESOLUTION, mode='constant')
        img = img.astype(np.float32)
        return img


class Memory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, RESOLUTION[0], RESOLUTION[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, reward, s2, isterminal):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = rnd.sample(range(0, self.size), sample_size) #take rows
        return self.s1[i], self.a[i], self.r[i], self.s2[i], self.isterminal[i]

NUM_CHANNELS = 1
class DQN:
    
    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        
        # Create the input variables
        self.s1_ = tf.placeholder(tf.float32, [None] + list(RESOLUTION) + [NUM_CHANNELS], name="State")
        self.a_ = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q_ = tf.placeholder(tf.float32, [None, self.actionCnt], name="TargetQ")
        
        self._create_network()
    
    def set_session(self, session):
        self.session = session
        
    def _create_network(self):
            
        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(self.s1_, num_outputs=32, data_format='NHWC',
                                                kernel_size=[8, 8], stride=[4, 4], padding='VALID',
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, data_format='NHWC', 
                                                kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
       
    #    conv3 = tf.contrib.layers.convolution2d(conv2, num_outputs=64, data_format='NHWC', 
    #                                            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
    #                                            activation_fn=tf.nn.relu,
    #                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    #                                            biases_initializer=tf.constant_initializer(0.1))
    #  
    #    
        conv_last_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv_last_flat, num_outputs=256, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=self.actionCnt, activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=tf.constant_initializer(0.1))
        self.best_a = tf.argmax(self.q, 1)
    
        self.loss = tf.losses.mean_squared_error(self.q, self.target_q_)
    
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = self.optimizer.minimize(self.loss)
    
    def learn(self, s1, target_q):
        feed_dict = {self.s1_: s1, self.target_q_: target_q}
        l, _ = self.session.run([self.loss, self.train_step], feed_dict=feed_dict)
        return l

    def get_q_values(self, state):
        return self.session.run(self.q, feed_dict={self.s1_: state})

    def get_best_action(self, state):
        return self.session.run(self.best_a, feed_dict={self.s1_: state})

    def simple_get_best_action(self, state):
        return self.get_best_action(state.reshape([1, RESOLUTION[0], RESOLUTION[1], 1]))[0]

class Agent:
    def __init__(self, actions):
        self.actions = actions
        self.brain = DQN(len(actions))
        
        # Create replay memory which will store the transitions
        self.memory = Memory(MEMORY_CAPACITY)
   
    def set_session(self, session):
        self.brain.set_session(session)
        
    def _learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """
    
        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > BATCH_SIZE:
            s1, a, r, s2, isterminal = self.memory.get_sample(BATCH_SIZE)
    
            q2 = np.max(self.brain.get_q_values(s2), axis=1)
            target_q = self.brain.get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + DISCOUNT_FACTOR * (1 - isterminal) * q2
            self.brain.learn(s1, target_q)
        
    def choose_act(self, epoch, s1):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * EPOCHS  # 10% of learning time
            eps_decay_epochs = 0.6 * EPOCHS  # 60% of learning time
    
            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps
        
        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if rnd.random() <= eps:
            a = rnd.randint(0, len(self.actions) - 1)
        else:
            # Choose the best action according to the network.
            a = self.brain.simple_get_best_action(s1)  
        return a

    def learn_from_act(self, s1, a, reward, s2, isterminal):
        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, reward, s2, isterminal)
    
        self._learn_from_memory()
    

    
class Environment:
    def __init__(self, config_file_path):
        # Create Doom instance
        self.game = self._configure_common(config_file_path)
        # Creates and initializes ViZDoom environment.
        self.init_train()
        
    def _configure_common(self, config_file_path):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(config_file_path)     
        return game
   
    def init_train(self):
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.init()
        print("Doom train initialized.")
        
    def init_test(self):
        # Reinitialize the game with window visible
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.init()
        print("Doom test initialized.")
        
    def get_state(self):
        return Util.preprocess(self.game.get_state().screen_buffer)
    
    def step_smooth(self, a):
        # Instead of make_action(a, FRAME_REPEAT) in order to make the animation smooth
        self.game.set_action(a)
        for _ in range(FRAME_REPEAT):
            self.game.advance_action()
    
    def step(self, action, FRAME_REPEAT):
        return self.game.make_action(action, FRAME_REPEAT) 
        
    def reset(self):
        return self.game.new_episode()
    
    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def get_total_reward(self):
        return self.game.get_total_reward()
        
    def get_actions(self):
     # Action = which buttons are pressed
        n = self.game.get_available_buttons_size()
        actions = [list(a) for a in it.product([0, 1], repeat=n)]
        return actions
     
    def close(self):
        self.game.close()
        
        

class Manager: 
    
    def __init__(self):
        self.session = tf.Session()
                   
        self.env = Environment(CONFIG_FILE_PATH)
        self.actions = self.env.get_actions()        
        self.agent = Agent(self.actions)
        self.agent.set_session(self.session)
    
        self.saver = tf.train.Saver()
        
        if LOAD_MODEL:
            print("Loading model from: ", MODEL_SAVEFILE)
            self.saver.restore(self.session, MODEL_SAVEFILE)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)
            signal.signal(signal.SIGINT, self.save_handler) 
            
    def save_handler(self, signum, frame):
        print("Interruption. Saving the network weigths to:", MODEL_SAVEFILE) 
        self.saver.save(self.session, MODEL_SAVEFILE)
        
    def _train(self, epoch, indetail = False):
        train_episodes_finished = 0
        train_scores = []
   
        self.env.init_train()
        print("Training...")
        self.env.reset()
        for learning_step in trange(LEARNING_STEPS_PER_EPOCH, leave=False):                    
            s1 = self.env.get_state()
            a = self.agent.choose_act(epoch, s1)
            reward = self.env.step(self.actions[a], FRAME_REPEAT)
            isterminal = self.env.is_episode_finished()
            s2 = self.env.get_state() if not isterminal else None
            self.agent.learn_from_act(s1, a, reward, s2, isterminal)
            
            if self.env.is_episode_finished():
                score = self.env.get_total_reward()
                train_scores.append(score)
                self.env.reset()
                train_episodes_finished += 1

        print("%d training episodes played." % train_episodes_finished)

        if indetail:
            train_scores = np.array(train_scores)
            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())    
        
        return train_scores
        
    def _test(self, epoch, indetail = False):
        print("\nTesting...")
        test_episode = []
        test_scores = []
        for test_episode in trange(TEST_EPISODES_PER_EPOCH, leave=False):
            self.env.reset()
            while not self.env.is_episode_finished():
                state = self.env.get_state()
                best_action_index = self.agent.brain.simple_get_best_action(state)

                self.env.step(self.actions[best_action_index], FRAME_REPEAT)
            r = self.env.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        print("Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
              "max: %.1f" % test_scores.max())
        
        return test_scores
  
            
    def run(self, with_test = False):
        time_start = time()
        if not SKIP_LEARNING:
            print("Starting the training!")
            for epoch in range(EPOCHS):
                print("\nEpoch %d\n-------" % (epoch + 1))
                self._train(epoch)               
                if with_test:
                    self._test(epoch)
    
                print("Saving the network weigths to:", MODEL_SAVEFILE)
                self.saver.save(self.session, MODEL_SAVEFILE)
    
                print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
    
        self.env.close()


        
    def run_trained(self):
        print("======================================")
        print("Training finished. It's time to watch!")    

        self.env.init_test()
        
        for _ in range(EPISODES_TO_WATCH):
            self.env.reset()
            while not self.env.is_episode_finished():
                state = self.env.get_state()
                best_action_index = self.agent.brain.simple_get_best_action(state)
    
                # Instead of make_action(a, FRAME_REPEAT) in order to make the animation smooth
                self.env.step_smooth(self.actions[best_action_index])
    
            # Sleep between episodes
            sleep(1.0)
            score = self.env.get_total_reward()
            print("Total score: ", score)
            
if __name__ == '__main__':   
    man = Manager()
    man.run(TRAIN_WITH_TEST)
    man.run_trained()

    
