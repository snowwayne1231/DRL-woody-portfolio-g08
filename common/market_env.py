# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


# 手續費 
# CHANGE_WEIGHT_FEE_RATIO = 0.00  #  sharp ratio 用在無手續費時績效好
CHANGE_WEIGHT_FEE_RATIO = 0.02
CHANGE_WEIGHT_STOCK_FEE_RATIO = 0.02
# 只選前幾名標的
NUM_CHOICE_STOCK = 9
# 最大持倉
RATIO_CONDITION_MAX_HOLD = 0.75


# 改變持倉比率
def proration_env_weights(env, action):
    if action.sum() == 0:
        action = np.random.rand(*action.shape)
    
    _before_weights = env.weights
    _argsort = np.argsort(action)
    _chosen_indices = _argsort[::-1][:NUM_CHOICE_STOCK]
    # _next_weights = action / action.sum()

    _next_weights = np.zeros(*action.shape)
    _redistribute_weights = np.array([action[_i] / 1+(idx * 0.2) for idx, _i in enumerate(_chosen_indices)])
    _redistribute_weights = _redistribute_weights / _redistribute_weights.sum()

    if _redistribute_weights[0] > RATIO_CONDITION_MAX_HOLD: # 排序過 第一個一定最大
        assert RATIO_CONDITION_MAX_HOLD > 0.5
        _ratio_splited = (_redistribute_weights[0] - RATIO_CONDITION_MAX_HOLD) / (len(_redistribute_weights) -1) # 平均分散
        _redistribute_weights = _redistribute_weights + _ratio_splited
        _redistribute_weights[0] = RATIO_CONDITION_MAX_HOLD
        
    
    for idx, _i in enumerate(_chosen_indices):
        _next_weights[_i] = _redistribute_weights[idx]
    
    # print('_next_weights: ', _next_weights)
    _gap_weights = _next_weights - _before_weights
    return _gap_weights


def simple_return_reward(env, **kwargs):
    reward = env.profit
    return reward


def g8_focus_profit_reward(env, threshold, alpha, feerate, **kwargs):
    profit = env.profit
    if profit > threshold:
        profit += 1
    elif profit < 0:
        profit -= 1
    return (profit - (env.charge_fee_ratio * feerate)) * alpha
    


def sharpe_ratio_reward_g8_v2(env, threshold, alpha, **kwargs):
    if env.std == 0:
        return 0
    reward = env.mean / env.std
    # if env.profit < 0:
    #     reward = (env.profit-1) * alpha
    # elif env.profit > threshold:
    #     reward * alpha
    return reward


def risk_adjusted_reward(env, threshold: float=float("inf"), 
                        alpha = 1,
                        drop_only: bool = False):
    reward = env.profit
    if (abs(reward) < threshold):
        return reward
    if (reward >= 0 and drop_only):
        return reward
    # reward = reward - alpha*(abs(reward) - threshold)
    if reward < 0: # 賠錢加強 reward 扣分
        reward = reward - alpha*(abs(reward) - threshold) 

    return reward * alpha


def resample_backfill(df, rule):
    return df.apply(lambda x: 1+x).resample(rule).backfill()


def resample_relative_changes(df, rule):
    return df.apply(lambda x: 1+x).resample(rule).prod().apply(lambda x: x-1)


class MarketEnv(gym.Env):
    def __init__(self, returns: DataFrame, features: DataFrame, show_info=False, trade_freq='days',
                 action_to_weights_func=proration_env_weights,
                 reward_func=simple_return_reward,
                 reward_func_kwargs=dict(),
                 noise=0,
                 state_scale=1,
                 trade_pecentage=0.1):
        self._load_data(returns=returns, features=features, show_info=show_info, trade_freq=trade_freq)
        self._init_action_space()
        self._init_observation_space()
        self.trade_pecentage = trade_pecentage
        self.start_index, self.current_index, self.end_index = 0, 0, 0
        self.action_to_weights_func = action_to_weights_func
        self.reward_func = reward_func
        self.reward_func_kwargs = reward_func_kwargs
        self.noise = noise
        self.state_scale = state_scale
        self.seed()
        self.reset()

    def _load_data(self, returns: DataFrame, features: DataFrame, show_info, trade_freq):
        resample_rules = {
            'days': 'D',
            'weeks': 'W',
            'months': 'M'
        }

        if(trade_freq not in resample_rules.keys()):
            raise ValueError(f"trade_freq '{trade_freq}' not supported, must be one of {list(resample_rules.keys())}")

        if returns.isnull().values.any():
            raise ValueError('At least one null value in investments')
        if features.isnull().values.any():
            raise ValueError('At least one null value in investments')
        # make sure dataframes are sorted
        returns = returns.sort_index().sort_index(axis=1)
        features = features.sort_index().sort_index(axis=1)

        features = resample_backfill(features, resample_rules[trade_freq]).dropna()
        # Scale features to -1 and 1
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            features[col] = (features[col]-mean)/std
        # resample based on trade frequency e.g. weeks or months
        returns = resample_relative_changes(returns, resample_rules[trade_freq])
        # Only keep feature data within peroid of investments returns
        features = features[(features.index.isin(returns.index))]
        # Only keep investment retuns with features
        returns = returns[(returns.index.isin(features.index))]
        self.features = features
        self.returns = returns
        # self.__ = 0.01
        if show_info:
            sd = returns.index[0]
            ed = returns.index[-1]
            print(f'Trading Frequency: {trade_freq}')
            print(f'{self.investments_count} investments loaded')
            print(f'{self.features_count} features loaded')
            print(f'Starts from {sd} to {ed}')

    def _init_action_space(self):
        action_space_size = self.investments_count
        self.min_action = np.zeros(action_space_size)
        self.max_action = np.ones(action_space_size)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

    def _init_observation_space(self):
        observation_space_size = self.features_count
        self.low_state = np.full(observation_space_size, -1)
        self.high_state = np.ones(observation_space_size)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    @property
    def investments_count(self):
        return len(self.returns.columns)

    @property
    def features_count(self):
        return len(self.features.columns)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def step(self, action):
    #     #print(f'step {self.returns.index[self.current_index]}')
    #     if self.current_index > self.end_index:
    #         raise Exception(f'current_index {self.current_index} exceed end_index{self.end_index}')

    #     done = True if (self.current_index >= self.end_index) else False
    #     # update weight
    #     self.weights = self.action_to_weights_func(action)
    #     self.episode += 1
    #     self.current_index += 1
    #     # update investments and wealth
    #     previous_investments = self.investments
    #     target_investments = self.wealth * self.weights

    #     # todo add trading cost??
    #     self.investments = target_investments

    #     inv_return = self.returns.iloc[self.current_index]
    #     previous_wealth = self.wealth
    #     # w_n = w_n-1 * (1+r)
    #     self.wealth = np.dot(self.investments, (1 + inv_return))
    #     self.profit = (self.wealth - previous_wealth)/previous_wealth
    #     # todo define new reward function
    #     reward = self.reward_func(self,**self.reward_func_kwargs)
    #     self.reward = reward

    #     self.max_weath = max(self.wealth, self.max_weath)
    #     self.drawdown = max(0, (self.max_weath - self.wealth) / self.max_weath)
    #     self.max_drawdown = max(self.max_drawdown, self.drawdown)
    #     self.mean = (self.mean * (self.episode-1) + self.profit)/self.episode
    #     self.mean_square = (self.mean_square * (self.episode-1) + self.profit ** 2)/self.episode

    #     info = self._get_info()
    #     state = self._get_state()
    #     return state, reward, done, info

    def step(self, action):
        # print('action (step): ', action)
        if self.current_index > self.end_index:
            raise Exception(f'current_index {self.current_index} exceed end_index{self.end_index}')

        done = self.current_index >= self.end_index
        _next_index = self.current_index + 1 # 下一筆 index
        returns_now = self.returns.iloc[self.current_index] # 報酬率
        returns_next = self.returns.iloc[_next_index]

        

        _previous_wealth = self.wealth # 資產比
        _now_weights = self.weights # 持股
        _investments = _previous_wealth * _now_weights
        _return_wealth = np.dot(_investments, (1 + returns_now)) if _investments.sum() > 0 else 1 # 之前持股帶來的收益

        
        _change_weights = self.action_to_weights_func(self, action) # 改變持股比例
        _next_weights = _now_weights + _change_weights # action執行完後的持股
        _investments_next = _return_wealth * _next_weights
        _return_wealth_next = np.dot(_investments_next, (1 + returns_next))

        
        self.charge_fee_ratio = (np.abs(_change_weights) * CHANGE_WEIGHT_FEE_RATIO).sum() # 持倉變動算出 手續費比率
        self.total_fee_ratio += self.charge_fee_ratio
        _charged_wealth = _return_wealth * (1 - self.charge_fee_ratio) # 收完收續費後的 資產

        self.none_charge_wealth = np.dot((self.none_charge_wealth * _now_weights), 1 + returns_now) if _now_weights.sum() > 0 else 1

        # reward = _return_wealth_next - _charged_wealth
        
        self.episode += 1
        
        self.profit = (_return_wealth_next - _previous_wealth) / _previous_wealth
        self.wealth = max(_charged_wealth, 0.0001)
        # self.wealth = _charged_wealth * (1+self.__)
        
        self.weights = _next_weights
        self.df_weights.iloc[self.current_index] = _next_weights
        
        reward = self.reward_func(self,**self.reward_func_kwargs)
        self.total_reward += reward

        self.max_weath = max(self.wealth, self.max_weath)
        self.drawdown = max(0, (self.max_weath - self.wealth) / self.max_weath)
        self.max_drawdown = max(self.max_drawdown, self.drawdown)
        self.mean = (self.mean * (self.episode-1) + self.profit)/self.episode
        self.mean_square = (self.mean_square * (self.episode-1) + self.profit ** 2)/self.episode
        if self.episode > 1:
            k = ((self.episode)/(self.episode-1))**0.5
            a = self.mean
            b = self.mean_square
            self.std = k*(b-a**2)**0.5
        info = self._get_info()
        state = self._get_state()
        
        self.current_index += 1

        if done:
            self.df_weights.iloc[self.current_index] = _next_weights

        return state, reward, done, info

    def render(self):
        # print('', self.weights * self.wealth)
        pass

    def reset(self):
        total_index_count = len(self.returns.index)
        last_index = total_index_count-2
        if (self.trade_pecentage >= 1):
            self.start_index = 0
            self.end_index = last_index
        else:
            self.start_index = np.random.randint(low=0, high=last_index*(1-self.trade_pecentage))
            self.end_index = int(self.start_index + total_index_count*self.trade_pecentage)
            self.end_index = min(self.end_index, last_index)
        #print(f'total: {total_index_count}, start index: {self.start_index}, end index: {self.end_index}')
        self.current_index = self.start_index
        self.investments = np.zeros(self.investments_count)
        self.weights = np.zeros(self.investments_count)
        self.wealth = 1
        self.max_weath = self.wealth
        self.max_drawdown = 0
        self.mean = 0
        self.mean_square = 0
        self.episode = 0
        self.profit=0
        self.total_reward=0
        self.drawdown = 0
        self.std = 0
        self.charge_fee_ratio = 0
        self.total_fee_ratio = 0
        self.df_weights = self.returns.copy()
        self.df_weights.iloc[-1] = np.zeros(self.investments_count)
        self.none_charge_wealth = 1
        # self.__ = min(2, self.__ * 1.04)
        return self._get_state()

    def _get_state(self):
        noise = np.random.normal(0, self.noise, self.observation_space.shape)
        state = self.features.iloc[self.current_index].to_numpy()*self.state_scale
        state = state + noise
        np.clip(state, -1, 1, out=state)
        if (state.shape != self.observation_space.shape):
            raise Exception('Shape of state {state.shape} is incorrect should be {self.observation_space.shape}')
        return state

    def _get_info(self):
        start_date = self.returns.index[self.start_index]
        current_date = self.returns.index[self.current_index]
        trade_days = (current_date-start_date).days
        if(trade_days ==0):
            cagr =0
        else:
            cagr = math.pow(self.wealth, 365/trade_days) - 1
        if (self.episode <= 1):
            std = 0
        else:
            k = ((self.episode)/(self.episode-1))**0.5
            a = self.mean
            b = self.mean_square
            std = k*(b-a**2)**0.5

        info = {
            'trade_days': trade_days,
            'wealths': self.wealth,
            'max_weath': self.max_weath,
            'cagr': cagr,
            'std': std,
            'mean': self.mean,
            'mean_square': self.mean_square,
            'mdd': self.max_drawdown,
            'profit': self.profit,
            'total_reward': self.total_reward,
            'dd': self.drawdown,
            'episode': self.episode,
            'date':current_date.value,
            'total_fee_ratio': self.total_fee_ratio,
            'none_charge_wealth': self.none_charge_wealth,
        }
        return info
