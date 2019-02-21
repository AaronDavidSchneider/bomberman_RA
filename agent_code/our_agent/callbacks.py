
import numpy as np
from time import time, sleep

from settings import s

import random
from sklearn.cluster import KMeans

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
    state = (x,y) #simplest case
    s_dim = arena.shape

    return state,s_dim



def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    self.reduced_state,self.state_dim = self.statereduction()

    self.q = np.zeros((*self.state_dim,6))
    self.epsilon = 0.2 # Exploration, Exploitation tradeoff




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

    # possible actions to be taken
    possible_actions = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB','WAIT']

    # needs to be adjusted!!!
    valid_actions =    range(6)

    # in case of exploration, choose random:
    prob_actions = [.23, .23, .23, .23, .08, 0.]

    # take decision based on exploaration and exploitation strategy
    if (random.random() < self.epsilon):
        self.next_action = np.random.choice(possible_actions[valid_actions], p=prob_actions[valid_actions])
    else:
        self.next_action = possible_actions[np.argmax(self.q[...,valid_actions],axis=-1)]


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s)')


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')
