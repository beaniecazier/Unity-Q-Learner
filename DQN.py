
# coding: utf-8

# In[ ]:


import tensorflow as tf      # Deep Learning library
from collections import namedtuple
import numpy as np           # Handle matrices
import random                # Handling random number generation
import time                  # Handling time calculation
from collections import deque# Ordered collection with ends
from keras.models import Sequential
from keras.layers import * #or use import Dense, Activation, Flatten
from keras.optimizers import * # or use import Adam


# Initialize Environment E  
# Initialize replay Memory M with capacity N (= finite capacity)  
# Initialize the DQN weights w  
# for episode in max_episode:  
#     s = Environment state  
#     for steps in max_steps:  
#          Choose action a from state s using epsilon greedy.  
#          Take action a, get r (reward) and s' (next state)  
#          Store experience tuple <s, a, r, s'> in M  
#          s = s' (state = new_state)  
#          
#          Get random minibatch of exp tuples from M  
#          Set Q_target = reward(s,a) +  γmaxQ(s')  
#          Update w =  α(Q_target - Q_value) *  ∇w Q_value  
# 
# Hyper Parameters
# There are some parameters that have to be passed to a reinforcement learning agent. You will see these over and over again.  
# 
# episodes - a number of games we want the agent to play.  
# gamma - aka decay or discount rate, to calculate the future discounted reward.  
# epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.  
# epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.  
# epsilon_min - we want the agent to explore at least this amount.  
# learning_rate - Determines how much neural net learns in each iteration.  

# In[ ]:


# Deep Q-learning Agent
class OurDQNAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, explore_rate, num_hidden_node):
        self.state_size = state_size
        self.action_size = action_size
        self.num_hidden_node = num_hidden_node
        self.memory = deque(maxlen=1000)
        self.gamma = discount_rate    # discount rate
        self.epsilon = explore_rate  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.num_hidden_node, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_hidden_node, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def bellman(reward):
        return reward + self.gamma * np.amax(self.model.predict(next_state)[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = bellman(reward)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

