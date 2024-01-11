# 构建一个类State描述平台的状态空间
import numpy as np
import collections
from agents import *

class State(object):
    def __init__(self, media_size, netizen_size, K, k = 0):
        # 初始化媒体和网民规模
        self.media_size = media_size
        self.netizen_size = netizen_size

        # 初始化媒体信誉,热度,报道以及网民信念
        self.media_reputation = np.random.uniform(low = 0, high = 1000, size = media_size)
        self.media_heat = np.random.uniform(low = 0, high = 1000, size = media_size)
        self.media_report = np.random.uniform(low = 0, high = 1, size = media_size)
        self.netizen_belif = np.random.uniform(low = 0, high = 1, size = netizen_size)

        # self.demand_history = demand_history
        self.K = K
        self.k = k


    def to_array(self):
        return np.concatenate( (self.media_reputation, 
                                self.media_heat, 
                                self.media_report, 
                                self.netizen_belif, [self.k]) )


# 构建一个类Action描述平台的动作空间：给每个media设置一个曝光率rate
class Action(object):
    def __init__(self, media_size):
        self.rate = np.repeat(1/media_size, media_size)


# 构建一个类PlatformEnvironment描述平台决策环境
class PlatformEnvironment(object):
    def __init__(self):
        self.K = 3               # episode duration
        self.T = 2                # sub_episode duration
        self.media_size = 3      # the number of medias located on the platform
        self.netizen_size = 20   # the number of netizens located on the platform
    
        self.w1, self.w2, self.w3 = 0.1, 0.2, 0.3    # weights in reputation, heat, polorization
        self.c1, self.c2, self.c3 = 0.4, 0.5, 0.6    # power coefficient in reputation, heat, polorization   

        self.probability = 0.5
        self.propensity = np.random.random(self.media_size)*1e-1
        self.cost = np.random.random(self.media_size)*1e-3
        
        self.mweight_lb, self.mweight_ub, self.nweight_lb, self.nweight_ub = 0.5, 0.95, 0.1, 0.45

        self.ub, self.lb = 0.8, 0.3
        self.pos = np.random.random(self.media_size)*1e-1
        self.neg = np.random.random(self.media_size)*1e-1

        self.reset()
  
    # define initial state
    def initial_state(self):
        return State(self.media_size, self.netizen_size, self.K)
    
    def reset(self):
        # print("restart")
        self.k = 0

    # define evolutionary step

    def step(self, state, action):

        # calculating the reward function
        total_reputation = np.sum(state.media_reputation)
        total_heat = np.sum(state.media_heat)
        total_polorization = (np.sum(np.absolute(state.media_report)) + np.sum(np.absolute(state.netizen_belif)))/(self.media_size + self.netizen_size)
        reward = self.w1*pow(total_reputation, self.c1) + self.w2*pow(total_heat, self.c2) + self.w3*pow(total_polorization, self.c3) 

        # calculating the next state
        media_reputation_T = np.tile(state.media_reputation, (self.T + 1, 1))
        media_heat_T = np.tile(state.media_heat, (self.T + 1, 1))
        media_report_T = np.tile(state.media_report, (self.T + 1, 1))
        netizen_belif_T = np.tile(state.netizen_belif, (self.T + 1, 1))
        for t in range(0, self.T):
            media = Media(self.netizen_size, self.probability, self.nweight_lb, self.nweight_ub,
                           self.media_size, action.rate, media_reputation_T[t], media_heat_T[t], self.mweight_lb, self.mweight_ub)
            media_reputation_T[t + 1],  media_heat_T[ t + 1] , netizen_belif_T[ t + 1], media_report_T[t + 1] = media.media_state_update(netizen_belif_T[t], media_report_T[t],self.pos, self.neg, self.propensity, self.cost, self.lb, self.ub)
        
        next_state = State(self.media_size, self.netizen_size, self.K, self.k)
        next_state.media_reputation, next_state.media_heat, next_state.media_report, next_state.netizen_belif =  media_reputation_T[self.T], media_heat_T[self.T], media_report_T[self.T], netizen_belif_T[self.T]

        self.k += 1
        return next_state, reward, self.k == self.K - 1 