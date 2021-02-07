#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Deep Reinforcement Learning for Stock Trading from Scratch: Single Stock Trading
# 
# Tutorials to use OpenAI DRL to trade single stock in one Jupyter Notebook | Presented at NeurIPS 2020: Deep RL Workshop
# 
# * This blog is based on our paper: FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, presented at NeurIPS 2020: Deep RL Workshop.
# * Check out medium blog for detailed explanations: https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac
# * Please report any issues to our Github: https://github.com/AI4Finance-LLC/FinRL-Library/issues
# * **Pytorch Version** 
# 
# 
# 

# ## Content

# * [1. Problem Definition](#0)
# * [2. Getting Started - Load Python packages](#1)
#     * [2.1. Install Packages](#1.1)    
#     * [2.2. Check Additional Packages](#1.2)
#     * [2.3. Import Packages](#1.3)
#     * [2.4. Create Folders](#1.4)
# * [3. Download Data](#2)
# * [4. Preprocess Data](#3)        
#     * [4.1. Technical Indicators](#3.1)
#     * [4.2. Perform Feature Engineering](#3.2)
# * [5.Build Environment](#4)  
#     * [5.1. Training & Trade Data Split](#4.1)
#     * [5.2. User-defined Environment](#4.2)   
#     * [5.3. Initialize Environment](#4.3)    
# * [6.Implement DRL Algorithms](#5)  
# * [7.Backtesting Performance](#6)  
#     * [7.1. BackTestStats](#6.1)
#     * [7.2. BackTestPlot](#6.2)   
#     * [7.3. Baseline Stats](#6.3)   
#     * [7.3. Compare to Stock Market Index](#6.4)             

# <a id='0'></a>
# # Part 1. Problem Definition

# This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.
# 
# The components of the reinforcement learning environment are:
# 
# 
# * Action: The action space describes the allowed actions that the agent interacts with the
# environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent
# selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use
# an action space {−k, ..., −1, 0, 1, ..., k}, where k denotes the number of shares. For example, "Buy
# 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or −10, respectively
# 
# * Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s',  i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio
# values at state s′ and s, respectively
# 
# * State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so
# our trading agent observes many different features to better learn in an interactive environment.
# 
# * Environment: single stock trading for AAPL
# 
# 
# The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.
# 
# We use Apple Inc. stock: AAPL as an example throughout this article, because it is one of the most popular and profitable stocks.

# <a id='1'></a>
# # Part 2. Getting Started- Load Python Packages

# <a id='1.1'></a>
# ## 2.1. Install all the packages through FinRL library
# 

# In[1]:


## install finrl library
# get_ipython().system('pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git')


# 
# <a id='1.2'></a>
# ## 2.2. Check if the additional packages needed are present, if not install them. 
# * Yahoo Finance API
# * pandas
# * numpy
# * matplotlib
# * stockstats
# * OpenAI gym
# * stable-baselines
# * tensorflow
# * pyfolio

# <a id='1.3'></a>
# ## 2.3. Import Packages

# In[2]:


# get_ipython().system('pip3 install pandas')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot

import sys
sys.path.append("../FinRL-Library")


# In[4]:


#Diable the warnings
import warnings
warnings.filterwarnings('ignore')


# <a id='1.4'></a>
# ## 2.4. Create Folders

# In[5]:


import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# <a id='2'></a>
# # Part 3. Download Data
# Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.
# * FinRL uses a class **YahooDownloader** to fetch data from Yahoo Finance API
# * Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
# 

# 
# 
# -----
# class YahooDownloader:
#     Provides methods for retrieving daily stock data from
#     Yahoo Finance API
# 
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
# 
#     Methods
#     -------
#     fetch_data()
#         Fetches data from yahoo API
# 

# In[6]:


# from config.py start_date is a string
config.START_DATE


# In[7]:


# from config.py end_date is a string
config.END_DATE


# ticker_list is a list of stock tickers, in a single stock trading case, the list contains only 1 ticker

# In[8]:


# Download and save the data in a pandas DataFrame:
data_df = YahooDownloader(start_date = '2009-01-01',
                          end_date = '2021-01-01',
                          ticker_list = ['AAPL']).fetch_data()


# In[9]:


data_df.shape


# In[10]:


data_df.head()


# In[11]:


data_df=pd.read_csv(r'C:\Users\e0690420\rl_forex\finrl\preprocessing\datasets\chicago_pmi\EURUSD\ohlc\EURUSD_Chicago_Pmi_2018-01-31 - Copy.csv')
data_df.rename(columns = {"time": "date"},  
          inplace = True) 
data_df.date = pd.to_datetime(data_df.date)
data_df['day'] = data_df['date'].dt.dayofweek
data_df['tic'] = 'A'
data_df.head()


# <a id='3'></a>
# # Part 4. Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * FinRL uses a class **FeatureEngineer** to preprocess the data
# * Add **technical indicators**. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc.
# 

# class FeatureEngineer:
# Provides methods for preprocessing the stock price data
# 
#     Attributes
#     ----------
#         df: DataFrame
#             data downloaded from Yahoo API
#         feature_number : int
#             number of features we used
#         use_technical_indicator : boolean
#             we technical indicator or not
#         use_turbulence : boolean
#             use turbulence index or not
# 
#     Methods
#     -------
#     preprocess_data()
#         main method to do the feature engineering

# <a id='3.1'></a>
# 
# ## 4.1 Technical Indicators
# * FinRL uses stockstats to calcualte technical indicators such as **Moving Average Convergence Divergence (MACD)**, **Relative Strength Index (RSI)**, **Average Directional Index (ADX)**, **Commodity Channel Index (CCI)** and other various indicators and stats.
# * **stockstats**: supplies a wrapper StockDataFrame based on the **pandas.DataFrame** with inline stock statistics/indicators support.
# 
# 

# In[12]:


## we store the stockstats technical indicator column names in config.py
tech_indicator_list=config.TECHNICAL_INDICATORS_LIST
print(tech_indicator_list)


# In[13]:


## user can add more technical indicators
## check https://github.com/jealous/stockstats for different names
tech_indicator_list=tech_indicator_list+['kdjk','open_2_sma','boll','close_10.0_le_5_c','wr_10','dma','trix']
print(tech_indicator_list)


# <a id='3.2'></a>
# ## 4.2 Perform Feature Engineering

# In[14]:


fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = tech_indicator_list,
                    use_turbulence=False,
                    user_defined_feature = False)

data_df = fe.preprocess_data(data_df)


# In[15]:


data_df.head()


# <a id='4'></a>
# # Part 5. Build Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
# 
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
# 
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# <a id='4.1'></a>
# ## 5.1 Training & Trade data split
# * Training: 2009-01-01 to 2018-12-31
# * Trade: 2019-01-01 to 2020-09-30

# In[16]:


#train = data_split(data_df, start = config.START_DATE, end = config.START_TRADE_DATE)
#trade = data_split(data_df, start = config.START_TRADE_DATE, end = config.END_DATE)
train = data_split(data_df, start = '2009-01-01', end = '2019-01-01')
trade = data_split(data_df, start = '2019-01-01', end = '2021-01-01')


# In[17]:


## data normalization, this part is optional, have little impact
#feaures_list = list(train.columns)
#feaures_list.remove('date')
#feaures_list.remove('tic')
#feaures_list.remove('close')
#print(feaures_list)
#from sklearn import preprocessing
#data_normaliser = preprocessing.StandardScaler()
#train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
#trade[feaures_list] = data_normaliser.transform(trade[feaures_list])


# In[18]:


data_df


# In[19]:


train


# In[20]:


trade


# <a id='4.2'></a>
# ## 5.2 User-defined Environment: a simulation environment class 

# <a id='4.3'></a>
# ## 5.3 Initialize Environment
# * **stock dimension**: the number of unique stock tickers we use
# * **hmax**: the maximum amount of shares to buy or sell
# * **initial amount**: the amount of money we use to trade in the begining
# * **transaction cost percentage**: a per share rate for every share trade
# * **tech_indicator_list**: a list of technical indicator names (modified from config.py)

# In[21]:


## we store the stockstats technical indicator column names in config.py
## check https://github.com/jealous/stockstats for different names
tech_indicator_list


# In[22]:


# the stock dimension is 1, because we only use the price data of AAPL.
len(train.tic.unique())


# In[23]:


stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[24]:


env_kwargs = {
    "hmax": 100, 
    "initial_amount": 100000, 
    "buy_cost_pct": 0.001, 
    "sell_cost_pct": 0.001, 
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}

e_train_gym = StockTradingEnv(df = train, **env_kwargs)


# In[25]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))


# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# In[26]:


agent = DRLAgent(env = env_train)


# ### Model Training: 5 models, A2C DDPG, PPO, TD3, SAC
# 
# 

# ### Model 1: A2C

# In[27]:


agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)


# In[28]:


trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000)


# In[ ]:


agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

for i in range(20):
    for j in range(20):
        agent = DRLAgent(env = env_train)
        A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005+(i-10)/1000, "learning_rate": 0.0002+(j-10)/200000}
        model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
        trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000)
        print('i:',i, 'j:',j)


# In[ ]:


# from bayes_opt import BayesianOptimization

# pbounds = {'ent_coef': (0.0001, 0.02), 'learning_rate': (0.00001, 0.002)}

# optimizer = BayesianOptimization(
#     f=black_box_function,
#     pbounds=pbounds,
#     verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )


# ### Model 2: DDPG

# In[28]:


agent = DRLAgent(env = env_train)
DDPG_PARAMS = {"batch_size": 64, "buffer_size": 500000, "learning_rate": 0.0001}


model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)


# In[29]:


trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=30000)


# In[30]:


agent = DRLAgent(env = env_train)

DDPG_PARAMS = {"batch_size": 64, "buffer_size": 500000, "learning_rate": 0.0001}
model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)

for i in range(20):
    for j in range(20):
        agent = DRLAgent(env = env_train)
        DDPG_PARAMS = {"batch_size": 64, "buffer_size": 500000+(i-10)*10000, "learning_rate": 0.0001+(j-10)/200000}
        model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
        trained_ddpg = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps=30000)
        print('i:',i,'j:',j)


# ### Model 3: PPO

# In[31]:


agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)


# In[32]:


trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=80000)


# In[33]:


agent = DRLAgent(env = env_train)

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

for i in range(20):
    for j in range(20):
        agent = DRLAgent(env = env_train)
        PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005+(i-10)/1000,
    "learning_rate": 0.0001+(j-10)/200000,
    "batch_size": 128,
}
        
        model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
        trained_ppo = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=80000)
        print('i:',i,'j:',j)


# ### Model 4: TD3

# In[34]:


agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 128, 
              "buffer_size": 1000000, 
              "learning_rate": 0.0003}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)


# In[35]:


trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=30000)


# In[36]:


agent = DRLAgent(env = env_train)

TD3_PARAMS = {"batch_size": 128, 
              "buffer_size": 1000000, 
              "learning_rate": 0.0003}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

for i in range(20):
    for j in range(20):
        agent = DRLAgent(env = env_train)
        TD3_PARAMS = {"batch_size": 128, 
              "buffer_size": 1000000+(i-10)*100000, 
              "learning_rate": 0.0003+(j-10)/20000}

        model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
        trained_td3 = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=30000)
        print('i:',i,'j:',j)


# ### Model 4: SAC

# In[ ]:


agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.00003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)


# In[ ]:


trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=30000)


# In[ ]:


agent = DRLAgent(env = env_train)

SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.00003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

for i in range(20):
    for j in range(20):
        for k in range(20):
            agent = DRLAgent(env = env_train)
            SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000+(i-10)*10000,
        "learning_rate": 0.00003+(j-10)/200000,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
            model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
            trained_sac = agent.train_model(model=model_sac, 
                                tb_log_name='sac',
                                total_timesteps=30000)
            print('i:',i,'j:',j,'k:',k)


# ### Trading
# * we use the environment class we initialized at 5.3 to create a stock trading environment
# * Assume that we have $100,000 initial capital at 2019-01-01. 
# * We use the trained model of PPO to trade AAPL.

# In[23]:


trade.head()


# In[25]:


## make a prediction and get the account value change
trade = data_split(data_df, start = '2019-01-01', end = '2021-01-01')

e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_sac,
                                           test_data = trade,
                                           test_env = env_trade,
                                           test_obs = obs_trade)


# <a id='6'></a>
# # Part 7: Backtesting Performance
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
# 

# In[26]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = BackTestStats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


# In[ ]:





# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# In[ ]:


print("==============Compare to AAPL itself buy-and-hold===========")
get_ipython().run_line_magic('matplotlib', 'inline')
BackTestPlot(account_value=df_account_value, 
             baseline_ticker = 'AAPL',
             baseline_start = '2019-01-01',
             baseline_end = '2021-01-01')


# <a id='6.3'></a>
# ## 7.3 Baseline Stats

# In[ ]:


print("==============Get Baseline Stats===========")
baesline_perf_stats=BaselineStats('AAPL')


# In[ ]:


print("==============Get Baseline Stats===========")
baesline_perf_stats=BaselineStats('^GSPC')


# <a id='6.4'></a>
# ## 7.4 Compare to Stock Market Index

# In[ ]:


print("==============Compare to S&P 500===========")
get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
BackTestPlot(df_account_value, baseline_ticker = '^GSPC')


# In[ ]:




