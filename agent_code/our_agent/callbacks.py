
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from time import time, sleep

from settings import s
from settings import e

import random
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor



GAMMA = 0.95 # hyperparameter
ALPHA = 0.01 # hyperparameter Learning rate
EPSILON = 0.2 # hyperparameter exploration, exploitation

def short_dist(self, pois):
    x, y, _, _ = self.game_state['self']
    pois = np.array(pois)

    if pois.size == 0:
        return int(np.sqrt(s.cols**2+s.rows**2)),-1
    else:
        pois[:,0] -= x
        pois[:,1] -= y

        dist = np.sqrt(pois[:,0]**2+pois[:,1]**2)
        m    = np.argmin(dist)

        return int(dist[m]), m


def statereduction(self):
    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    # statereduction-function needs to be changed!!
    coins_state,_ = short_dist(self,coins)
    bombs_state,_ = short_dist(self,bomb_xys)
    others_state,_ = short_dist(self,others)

    #weighted_state = int(0.4*coins_state+0.4*bombs_state+0.1*others_state)

    #state = min(weighted_state,24)

    state = (coins_state,bombs_state,others_state)

    s_dim = (self.dim,self.dim,self.dim)

    return state,s_dim

def get_reward(self):
    """
    REWARD Function, needs to be optimized manually to train the agent with rewards
    """
    reward = 0
    if e.COIN_COLLECTED in self.events:
        reward += 100
    if e.KILLED_OPPONENT in self.events:
        reward += 500
    if e.CRATE_DESTROYED in self.events:
        reward += 20

    if e.GOT_KILLED in self.events:
        reward -= 100
    if e.KILLED_SELF in self.events:
        reward -= 200


    return reward


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')

    self.dim=int(np.sqrt(s.cols**2+s.rows**2))+1
    self.state_dim = (self.dim,self.dim,self.dim) # warning: needs to be changed
    self.q = np.zeros((*self.state_dim,6))

    self.r = []
    self.a = []
    self.state = [] # warning: needs to be changed
    self.bomb_history = []
    self.reduced_state = (24,24,24)

    self.train_flag=True #temp fix

    if not self.train_flag:
        self.q = np.load('agent_code/our_agent/q.npy')



def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    self.reduced_state, self.state_dim = statereduction(self)
    #self.logger.debug(f'reduced_state: {self.reduced_state}')

    # possible actions to be taken
    # adapted from simple agent:
    # Check which moves make sense at all

    possible_actions = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB','WAIT']
    x, y, _, bombs_left = self.game_state['self']
    arena = self.game_state['arena']
    bombs = self.game_state['bombs']
    bomb_map = np.ones(arena.shape) * 5
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in self.game_state['others']]
    coins = self.game_state['coins']

    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
            (self.game_state['explosions'][d] <= 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x-1,y) in valid_tiles: valid_actions.append(1)
    if (x+1,y) in valid_tiles: valid_actions.append(0)
    if (x,y-1) in valid_tiles: valid_actions.append(2)
    if (x,y+1) in valid_tiles: valid_actions.append(3)
    if (x,y)   in valid_tiles: valid_actions.append(5)
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x,y) not in self.bomb_history: valid_actions.append(4)

    valid_actions = np.array(valid_actions)
    possible_actions = np.array(possible_actions)
    #self.logger.debug(f'valid_actions: {possible_actions[valid_actions]}')
    # in case of exploration, choose random:
    prob_actions = np.array([.23, .23, .23, .23, .08, 0.])
    if len(valid_actions)>0: #reduces errors
        prob_actions = prob_actions[valid_actions]/np.sum(prob_actions[valid_actions]) #norm and drop others
        # take decision based on exploaration and exploitation strategy
        if (random.random() < EPSILON and self.train_flag):
            self.next_action = np.random.choice(possible_actions[valid_actions], p=prob_actions)
        else:
            self.next_action = possible_actions[valid_actions][np.argmax(self.q[self.reduced_state][valid_actions])]

        if self.next_action == 'BOMB':
            self.bomb_history.append((x,y))
    else:
        self.next_action = 'WAIT'

def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')
    if e.INVALID_ACTION in self.events:
        self.logger.debug(f'Encountered INVALID_ACTION')

    if e.MOVED_RIGHT in self.events:
        self.a.append(0)
    elif e.MOVED_LEFT in self.events:
        self.a.append(1)
    elif e.MOVED_UP in self.events:
        self.a.append(2)
    elif e.MOVED_DOWN in self.events:
        self.a.append(3)
    elif e.BOMB_DROPPED in self.events:
        self.a.append(4)
    elif e.WAITED in self.events:
        self.a.append(5)
    else:
        self.a.append(-1) #releases error

    self.r.append(get_reward(self))
    self.state.append(self.reduced_state)

def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    episode = self.game_state['step'] #Länge der Runde.
    max_steps = s.max_steps

    h = np.zeros_like(self.q)

    for i in range(episode-1):
        h[self.state[i]][self.a[i]] = self.r[i] + GAMMA*np.max(self.q[self.state[i+1]])-self.q[self.state[i]][self.a[i]]

    #########################
    # do regression:
    regr = RandomForestRegressor(max_depth=2, n_estimators=100)
    regr.fit(self.q.reshape(self.dim*self.dim*self.dim,6),h.reshape(self.dim*self.dim*self.dim,6))

    #update q:
    self.q += ALPHA * regr.predict(self.q.reshape(self.dim*self.dim*self.dim,6)).reshape(self.dim,self.dim,self.dim,6)


    ##########################
    # flush Y,r,s,a:
    self.r = []
    self.a = []
    self.state = []

    #########################
    # save q
    np.save('agent_code/our_agent/q.npy',self.q)
