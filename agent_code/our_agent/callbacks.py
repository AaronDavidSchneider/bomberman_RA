
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import time

from collections import deque
from random import shuffle

from settings import s
from settings import e

import random
from sklearn.ensemble import RandomForestRegressor
import sys #only needed to escape process in case of error in end_of_episode


###############################################################################
# HYPERPARAMETER
###############################################################################

GAMMA                = 0.95    # hyperparameter
ALPHA                = 0.1    # hyperparameter Learning rate
EPSILON              = 1       # hyperparameter exploration, exploitation
C                    = 10      # hyperparameter: Slope of EPSILON-decay
TRAIN                = True    # set manually as game_state is not existant before act
START_FROM_LAST      = False   # caution: Continue last Training

###############################################################################
# HELP-FUNCTIONS
###############################################################################
action_dict = {'RIGHT':0, 'LEFT':1, 'UP':2, 'DOWN':3, 'BOMB':4, 'WAIT':5}

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def get_action_ideas(self,arena,d,x,y,others,bombs,bomb_xys,coins):
    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)
    f_index = [0,0,0,0]

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    targets = coins + dead_ends + crates

    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x,y), targets, self.logger)
    if d == (x,y-1): action_ideas.append('UP')
    if d == (x,y+1): action_ideas.append('DOWN')
    if d == (x-1,y): action_ideas.append('LEFT')
    if d == (x+1,y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')
    f_index.append(1)

    # Add proposal to drop a bomb if at dead end
    if (x,y) in dead_ends:
        action_ideas.append('BOMB')
        f_index.append(2)
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
            f_index.append(3)
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x,y) and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(1) > 0):
        action_ideas.append('BOMB')
        f_index.append(4)

    # above the features covered by Q
    ##############################################
    # below the features not covered by Q, instead if those action_ideas exist:
    #   safety-margin, the simple_agent behaviour will be executed

    # Add proposal to run away from any nearby bomb about to blow
    for xb,yb,t in bombs:
        if (xb == x) and (abs(yb-y) < 4):
            # Run away
            if (yb > y):
                action_ideas.append('UP')
                f_index.append(5)
            if (yb < y):
                action_ideas.append('DOWN')
                f_index.append(5)
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
            f_index.extend([5,5])
        if (yb == y) and (abs(xb-x) < 4):
            # Run away
            if (xb > x):
                action_ideas.append('LEFT')
                f_index.append(6)
            if (xb < x):
                action_ideas.append('RIGHT')
                f_index.append(6)
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
            f_index.extend([6,6])

    # Try random direction if directly on top of a bomb
    for xb,yb,t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])
            f_index.extend([7,7,7,7])

    return action_ideas, f_index

def ideas_to_feature(self,action_ideas,f_index):
    """
    Uses the action ideas and their index to create 6 one-hot-feature-vectores

    Features in Q:
        Feature 1: one for direction of target
        Feature 2: one if bomb at deadend possible
        Feature 3: one if bomb at opponent possible
        Feature 4: one if bomb at target crate possible

    Features not in Q:
        Feature 5-7: Run away from bomb if possible
        Feature 0: one for all directions
    """
    features = []
    action_ideas = [action_dict[k] for k in action_ideas]
    for a in range(6):
        f_a = np.zeros(self.f_dim, dtype=np.int32)
        for i in range(len(action_ideas)):
            if action_ideas[i]==a:
                f_a[f_index[i]]=1
        features.append(f_a)
    return features


def get_actions(self):
    # init the dimension variables during setup process
    if not hasattr(self, 'game_state'):

        self.f_dim = 8 #number of features
        self.a = int(5) # initialize a
        self.feature = [np.zeros(self.f_dim, dtype=np.int32)] * 6
        self.reduced_feature = [np.zeros(self.f_dim-4, dtype=np.int32)] * 6
        #self.weights = np.array([np.random.rand(self.f_dim)] * 6) # 8 features!

        # logging:
        self.a_list = []
        self.f_list = []
        self.r_list = []
        self.Q_list = []

        return

    # Gather information about the game state
    arena = self.game_state['arena']
    x, y, _, bombs_left, score = self.game_state['self']
    bombs = self.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in self.game_state['others']]
    coins = self.game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    if self.coordinate_history.count((x,y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x,y))
    self.long_coordinate_history.append((x,y))

    ####################################
    # Determine valid actions
    # adapted from simple agent:
    # Check which moves make sense at all

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
    self.logger.debug(f'Valid actions: {valid_actions}')


    action_ideas, f_index = get_action_ideas(self,arena,d,x,y,others,bombs,bomb_xys,coins)
    features = ideas_to_feature(self,action_ideas,f_index)

    valid_actions = np.array(valid_actions, dtype=np.int32)

    return valid_actions, features, action_ideas

def get_reward(self):
    """
    REWARD Function, needs to be optimized manually to train the agent with rewards
    """
    reward = 0
    no_useful_action = True #negative reward by default

    if e.COIN_COLLECTED in self.events:
        reward += 500
        no_useful_action = False
    if e.KILLED_OPPONENT in self.events:
        reward += 500
        no_useful_action = False
    if e.CRATE_DESTROYED in self.events:
        reward += 500
        no_useful_action = False
    if e.GOT_KILLED in self.events:
        reward -= 50
        no_useful_action = False
    if e.KILLED_SELF in self.events:
        reward -= 100
        no_useful_action = False
    if no_useful_action:
        reward -= 10

    return reward

# make EPSILON greedy
def eps_greedy(self):
    return EPSILON/(C*self.round/s.n_rounds+1)

def get_deterministic_action(self,possible_actions,valid_actions,action_ideas):
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in possible_actions[valid_actions]:
            self.next_action = a
            self.a = int(action_dict[a])
            break

def is_loop(self,valid_actions):
    x, y, _, _, _ = self.game_state['self']
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    actions_2_dir = [5,0,1,3,2]
    #valid_tiles, valid_actions = [], []
    for i in range(5):
        d = directions[i]
        a = actions_2_dir[i]
        if ((a in valid_actions) and
            (self.coordinate_history.count(d) > 0)):
            valid_actions = valid_actions[np.where(valid_actions != a)]
    return valid_actions



###############################################################################
# MAIN-FUNCTIONS
###############################################################################

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    get_actions(self) #init the dimension variables
    self.Q = [np.zeros((2,2,2,2))] * 6
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 10)
    self.long_coordinate_history = deque([], 10)
    self.ignore_others_timer = 0

    if not TRAIN or START_FROM_LAST:
        self.Q = np.load('q.npy')

    if TRAIN:
        self.timer = time.time()
        self.round = 0 #Fortschrittsanzeige


def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of th
    e game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """

    valid_actions, self.feature, action_ideas = get_actions(self)
    self.reduced_feature = [self.feature[a][1:5] for a in range(6)]
    possible_actions = np.array(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB','WAIT'])
    train = self.game_state['train']
    self.logger.debug(f'reduced_feature : {self.reduced_feature[0]}')

    ##############################
    # choose action

    self.Q = np.array([self.clf[a].predict(self.reduced_feature[a].reshape(1, -1)) for a in range(6)])

        if ((train and random.random() < eps_greedy(self)) or
            ([self.feature[i][j] for i in range(6) for j in [5,6,7]].count(1) > 0) or
            ([self.feature[i][j] for i in range(6) for j in [1,2,3,4]].count(1)==0)):
            get_deterministic_action(self,possible_actions,valid_actions,action_ideas)
        else:
            valid_actions = is_loop(self,valid_actions) #remove tiles that result in loop
            if (([self.feature[4][j] for j in [2,3,4]].count(1) == 0)):
                # do not drop bomb if not needed!
                valid_actions = valid_actions[np.where(valid_actions != 4)]
            if (([self.feature[a][j] for a in range(4) for j in [1]].count(1) > 0) and
                (len(valid_actions)>1)):
                # do not wait, if not needed!
                valid_actions = valid_actions[np.where(valid_actions != 5)]

            if len(valid_actions) > 0:
                #Q = np.array([self.Q[a][tuple(self.reduced_feature[a])] for a in range(6)])
                self.next_action = possible_actions[valid_actions][np.argmax(self.Q[valid_actions])]
            else:
                self.next_action = 'WAIT'

        if self.next_action == 'BOMB':
            x, y, _, bombs_left, score = self.game_state['self']
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

    self.a_list.append(self.a)
    self.f_list.append(self.reduced_feature)
    #self.Q_list.append([self.Q[a][tuple(self.reduced_feature[a])] for a in range(6)])
    self.Q_list.append(self.Q)
    self.r_list.append(get_reward(self)) #log reward


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    self.logger.debug(f'Encountered {len(self.events)} game event(s) in final step')

    self.f_list = np.array(self.f_list)
    self.Q_list = np.array(self.Q_list)

    # normalize rewards:
    self.r_list = np.array(self.r_list)
    r_list_normed = self.r_list.shape[0]*self.r_list / np.sum(self.r_list)

    # update each classifier for each possible action:
    for a in range(6):
        Y = np.array(self.r_list)[1:] + GAMMA * self.Q_list[1:,a] # SARSA
        rho = Y - self.Q_list[:-1,a]
        try:
            regr = RandomForestRegressor(max_depth=3,n_estimators=100)
            regr.fit(self.f_list[:-1,a], rho.ravel())
            for i in range(len(self.f_list)):
                f = tuple(self.f_list[i][a])
                self.Q[a][f] += ALPHA * regr.predict(self.f_list[i,a].reshape(1, -1))
        except:
            print('ERROR: execution stuck! check the logs for more information', flush=True)
            sys.exit(1)

    np.save('q.npy', self.Q)

    self.round += 1
    print(f'Next Round: {self.round}   ({np.round(self.round/s.n_rounds*100,2)}%), Time since starting: '+time.strftime("%H:%M:%S", time.gmtime(time.time()-self.timer)), flush=True)

    # save rewards
    reward_file = open("rewards.txt","a+")
    reward_file.write("{:d}\n".format(int(np.sum(np.array(self.r_list))/self.r_list.shape[0])))
    reward_file.close()

    ##########################
    # flush lists:
    self.a_list = []
    self.f_list = []
    self.r_list = []
    self.Q_list = []
