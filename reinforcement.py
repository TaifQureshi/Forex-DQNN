from backtesting import Backtest,Strategy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import talib
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM,Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import time
from empyrical import sortino_ratio
import random
import os


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 2000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -50  # For model save
MEMORY_FRACTION = 0.20
AGGREGATE_STATS_EVERY = 50
ep_rewards=[-200]


def get_state(index):
    d = data.copy().iloc[index-20:index]
    d.drop(labels=['Volume'],axis=1,inplace=True)
    d = scaler.fit_transform(d)
    d = pd.DataFrame(d)
    x = []
    for i in range(16,len(d)+1):
        x.append(d.iloc[i-16:i].values)
    x = x[-1:]
    # d = np.array(d,dtype=float)
    # balance = np.array(balance)
    # trades = np.array(trades)
    # print(x)
    return np.array(x,dtype=float)



class DQNAgent(Strategy):
    price_delta = .004
    def init(self,path='models/2x256_1601105543_V1.h5',load=True):
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.001
        self.episode = 0
        self.average_reward = 0
        self.min_reward = -200
        self.max_reward = 0
        self.MIN_REWARD = -50
        self.epsilon = 1
        self.load = load
        self.reward_length = 50
        # self.profit = []
        # Main model
        if load:
            self.model = load_model(path)
            self.epsilon = 0.1
            print("model loaded")
        else:
            self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.last_action = deque(maxlen=2)
        self.step = 0
        self.account_history = [self.equity]
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(LSTM(16,activation='relu',input_shape=(16,13)))
        model.add(Dropout(0.3))
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]).reshape(-1,16,13)
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]).reshape(-1,16,13)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X).reshape(-1,16,13), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            print("Weights updated")
            self.target_update_counter = 0



    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1,16,13))[0]
     
    def step_act(self,action):
        if action == 1:
            self.buy(size=.1)
            for i in self.trades:
                if i.is_short:
                    i.close()
        elif action == 2:
            self.sell(size=.1)
            for i in self.trades:
                if i.is_long:
                    i.close()
        self.account_history.append(self.equity) 
        length = min(self.step,self.reward_length)
        ret = np.diff(self.account_history)[-length:]
        r = sortino_ratio(ret)
        if abs(r) != np.inf and not np.isnan(r):
            reward = r
        else:
            reward = 0
        if self.step > 5:
            if self.last_action[-1] == self.last_action[-2] and reward < 0:
                reward -= 5
        done = False
        if reward > 10:
            done = True
        c = self.data.index[-1]
        new_state = get_state(c+1)
        return new_state, reward, done

    #loop function run every step
    def next(self):
        
        self.step += 1
        self.episode += 1
        c = self.data.index[-1]
        current_state = get_state(c)
        
        if np.random.random() > self.epsilon:
            # Get action from Q table
            try:
                action = self.get_qs(current_state)
                print("Predict ")
                action = np.argmax(action)
            except Exception as identifier:
                print("Error,",identifier)
                action = np.random.randint(0,3)
               
                
        else:
            # Get random action
            action = np.random.randint(0, 3)
        
        
        new_state, reward, done = self.step_act(action)
        self.last_action.append(action)
        self.update_replay_memory((current_state, action, reward, new_state, done))
        self.train(done, self.step)
        print(f'Reward: {reward}, Max: {self.max_reward}, Min: {self.min_reward}, Steps:{self.step}, Action:{action}, Balance:{self.equity}')
        ep_rewards.append(reward)
        if not self.episode % AGGREGATE_STATS_EVERY or self.episode == 1:
            self.average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            self.min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            self.max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        if self.min_reward > self.MIN_REWARD or self.step == 3800:
            self.MIN_REWARD = self.min_reward
            if self.load:
              self.model.save(f'models/{MODEL_NAME}_v2_{int(time.time())}.h5')
            else:
                self.model.save(f'models/{MODEL_NAME}_{int(time.time())}_V1.h5')

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        # self.profit.append(self.equity)
        # if self.step > 3000 and self.load:
        #     plt.plot(self.profit)
        #     plt.show()
        #     return

            




data = pd.read_csv('GBPUSD.csv')
data['ROC'] = talib.RSI(data.Close,timeperiod=10)
data['ROCP'] = talib.ROCP(data.Close, timeperiod=10)
data['EMA-9'] = talib.EMA(data.Close,timeperiod=9)
data['EMA-28']= talib.EMA(data.Close,timeperiod=28)
data['SAR'] = talib.SAR(data.High, data.Low, acceleration=0, maximum=0.2)
data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(data.Close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
data['Real'] = talib.TRANGE(data.High, data.Low, data.Close)
data.drop(labels=['Gmt time'],axis=1,inplace=True)
data.dropna(inplace=True)
data.replace([np.inf, -np.inf],0,inplace=True)
# print(data.head(20))
scaler = MinMaxScaler()
# x = get_state(100)
# print(x.shape)
# print(x)
# if x.shape != (None,16):
# #     print('t')
bt = Backtest(data[100:4000], DQNAgent, commission=.0002, margin=.05)

print(bt.run())
bt.plot()