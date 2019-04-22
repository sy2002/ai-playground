# Maze Runner using Reinforcement Learning: Q-Learning variant
#
# Generates a random maze and then uses Q-Learning to solve the maze.
# The results are a numbered series of PNG files:
# "mrunner_q_w_*.png": shows a "walking-heatmap" of the Q-Learning
# "mrunner_q_p_*.png": shows the result
#
# Assumptions to make it more "realistic" than the "godmode" dynamic
# programming version that you find in my other source "maze_runner.py":
# 
# 1. We do not initialze the Q function with all possible states at
#    the beginning, as we are not "knowing" them, yet, but we do so
#    on the go.
#
# 2. We are exploring the maze from the start position (versus
#    knowing all states and "exploring backwards" as in the
#    original version)
#
# 3. No "god mode", i.e. transition probabilities, etc. but in
#    contrast to e.g. the Lecture 59, our agent can "see" the walls
#    where he is currently standing, so we should have a bit
#    better performance, than at the lecture sample code "q_learning.py"
#    where the agent "runs into walls all the time".
#
# inspired by Lecture 59/q_learning.py of Udemy/Lazy Programmer Inc/Reinforcement Learning
#
# done by sy2002 on 27th of May, 17th and 23rd of June 2018

from collections import defaultdict
from copy import deepcopy
from multiprocessing import Process
import grid_world
import maze_generator
import numpy as np
import progressbar

# maze size
mx = 180
my = 100

# imgage size
img_dx = 1350
img_dy = 750

# some learnings on 17th of June: when you have larger step sizes per episode,
# e.g. in a 70x70 maze, make sure, that ALPHA is rather large and 
# the ALPHA_DECAY is not that huge, otherwise the sheer amount of steps will
# drag down alpha so much, that the system  "is not learning anything anymore".
# Also make sure, that the randomness EPSILON is not adding to much noise
# The POLICY_BAILOUT needs to be large enough (an upper boundary is mx * my)
# A fair value set for a 80x80 maze is given here:

WIN_REWARD          = 100000       # needs to be large due to float accuracy
STEP_PUNISH         = -0.1         # cost of "walking"
EPISODES            = 10000        # number of episodes to play
ALPHA               = 0.8          # learning rate
ALPHA_DECAY         = 0.0005       # learning rate decay: added to divisor on each learning step
GAMMA               = 0.999        # discount factor
EPSILON             = 0.4          # epsilon-greedy algorithm
EPSILON_DECAY_t     = 0.25
EPSILON_DECAY_m     = 50           # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

PROBE               = 100          # after how many episodes to we try the current policy
POLICY_BAILOUT      = 2500         # not solving the maze after this amount of steps disqualifies a policy 

# index 0 = wall, index = 1 path, index 2 = start/end, index 3 = shortest path from start to end, index 4 = summand for probing
rgb_color_map = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]

Saver_Procs = []
Walk_Mazes = []

# =================================================================
# Generate maze and initialize Grid World
# =================================================================

print("sy2002's Maze Runner (Multiprocess Version) done on 23rd of June 2018")
print(mx, "x", my, " maze with ", img_dx, "x", img_dy, " pixels\n")

# generate maze and save the "unsolved" maze
maze = maze_generator.make_maze(mx, my)
maze_generator.add_start_and_end(maze, rgb_color_map)
new_maze = deepcopy(maze)
Walk_Mazes.append(new_maze)
proc = Process(target=maze_generator.save_maze, args=(new_maze, "mrunner_q_1.png", img_dx, img_dy, rgb_color_map, ))
Saver_Procs.append(proc)
proc.start()

# move the maze to "Grid World" by doing the following:
# 1. Make sure that each step "costs something", so set everything to a reward of STEP_PUNISH
# 2. Reward the end point with WIN_REWARD
# 3. For each state aka position: define the possible movements (based on the maze layout)
# 4. Make the  end point a terminal point

start, end = maze_generator.find_start_end(maze)
terminal_state = (end[1], end[0])
start_state = (start[1], start[0])
grid = grid_world.Grid(mx, my, start_state)

rewards = {}
for y in range(my):
    for x in range(mx):
        if maze[y][x] == 1:                 # 1 means "path", 0 means "wall"
            rewards[(y, x)] = STEP_PUNISH   # make sure, each step "costs something"
rewards[terminal_state] = WIN_REWARD      # the goal offers 10.0 reward
rewards[start_state] = -WIN_REWARD
        
# create possible actions for each position, because we assume, that our agent
# can "see" where a wall is and where a path is, so we can have a slightly more
# efficient Q-Learning algorithm (less "supid running into walls")
actions = {}
for y in range(my):
    for x in range(mx):
        possible_actions = []
        if y > 0      and maze[y - 1][x] > 0: possible_actions.append("U")
        if y < my - 1 and maze[y + 1][x] > 0: possible_actions.append("D")
        if x > 0      and maze[y][x - 1] > 0: possible_actions.append("L")
        if x < mx - 1 and maze[y][x + 1] > 0: possible_actions.append("R")
        # only care, if we are standing at start/end or path but not "inside" a wall and if there are any actions
        if maze[y][x] > 0 and len(possible_actions):
            actions[(y, x)] = set(possible_actions)
actions.pop(terminal_state) # make the end point a terminal state

grid.set(rewards, actions)

# =================================================================
# Helpers
# =================================================================

# returns the argmax of the dictionary (key) and the maximum value
def max_of_dict(d):
    argmax = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            argmax = k
            max_val = v
    return argmax, max_val

# we are initializing Q as we go, therefore we need a way to know
# the key that was searched for, so that we can only add actions to Q
# that are actually allowed: a normal defaultdict is not giving us
# this capability, see:
# https://stackoverflow.com/questions/2912231/is-there-a-clever-way-to-pass-the-key-to-defaultdicts-default-factory
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret    

# =================================================================
# Execute Q-Learning
# =================================================================

# Q[state][action], i.e. a dictionary of dictionarys
# this nested structure is necessary for easily finding
# best action given a state; we use a custom dictionary, that facilitates
# the "on the go" initialization and we initialize only by actions that
# are allowed at the current step to make our algorithm a bit faster
Q = keydefaultdict(lambda key: {action: 0.0 for action in grid.actions[key]})
update_counts = keydefaultdict(lambda key: {action: 1.0 for action in grid.actions[key]})

t = 1.0

bar = None

for episode in range(EPISODES):
    # progress bar restarts between the PROBE intervals
    if bar is None:
        if episode > 0:
            print("\n")
            bar = progressbar.ProgressBar(max_value=PROBE)
            bar.start()
    else:
        bar.update((episode % PROBE) + 1)

    # let epsilon decay over time
    if episode % EPSILON_DECAY_m == 0:
        t += EPSILON_DECAY_t

    # each episode starts again at the begining of the maze
    s = start_state
    grid.set_state(s)

    # get best action (argmax) and we need to init Q on the go
    a, _ = max_of_dict(Q[s])

    average_alpha = 0
    episode_step_count = 0
    walk_maze = [[0 for x in range(mx)] for y in range(my)]
    
    # and the beginning: struggle through due to the effect of Q-learning:
    # zero (init) is better than punishment, so the algorithm naturally
    # expands its search area; and the closer to start places get more and
    # more negative, as long as the goal is not found, which motivates the
    # agent to move away from the start quickly
    while not grid.game_over():
        episode_step_count += 1

        #epsilon-greedy: do a random move with the decayed EPSILON probability
        p = np.random.random()
        eps = EPSILON / t
        if p > (1 - eps):
            a = np.random.choice(list(grid.actions[s]))

        # exploit or explore and collect reward
        r = grid.move(a)
        s2 = grid.get_state()

        # record each step the algorithm does to generate the "walk_heatmap"
        walk_maze[s2[0]][s2[1]] += 1

        # adaptive learning rate
        alpha = ALPHA / update_counts[s][a]
        average_alpha += alpha
        update_counts[s][a] += ALPHA_DECAY

        # Q-Learning
        old_qsa = Q[s][a]
        # if this is not the terminal position (which has no actions), then we can proceed as normal
        if s2 != terminal_state:
            # "Q-greedily" grab the gain of the best step forward from here
            a2, max_q_s2a2 = max_of_dict(Q[s2])
        else:
            a2 = a
            max_q_s2a2 = 0 # G (future rewards) of terminal position is 0 although the reward r is high
        Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2a2 - Q[s][a])

        # next state = current state, next action is the "Q-greedy" best action
        s = s2
        a = a2        

    # terminal state reached during Q function run does not mean, that the Q values already lead to solving the maze
    # therefore, also the policy is not necessarily useful at this moment in time, so we need a bail out criterium
    grid.set_state(start_state)
    policy = {}
    V = {}
    bailout_counter = 0
    while not grid.game_over():
        s = grid.get_state()
        a, max_q = max_of_dict(Q[s])
        policy[s] = a
        V[s] = max_q
        grid.move(a)
        bailout_counter += 1
        if bailout_counter == POLICY_BAILOUT:
            break
    solved_it = grid.game_over()

    if episode % PROBE == 0 or solved_it:
        bar = None
        print("\nEpisode: ", episode)
        print("Steps needed in current episode: ", episode_step_count, "\tEpsilon: ", eps, "\tAverage Alpha: ", average_alpha / episode_step_count)
        
        # Multiprocessing: Make sure the new process runs on completely separate data
        # store "everything" in the lists Walk_Mazes and Saver_Procs so that no garbage collection takes place
        new_walk_maze = deepcopy(walk_maze)
        Walk_Mazes.append(new_walk_maze)
        new_org_maze = deepcopy(maze)
        Walk_Mazes.append(new_org_maze)
        new_map = deepcopy(rgb_color_map)
        Walk_Mazes.append(new_map)
        proc = Process(target=maze_generator.save_maze, args=(new_walk_maze, "mrunner_q_w_{0:0>5}.png".format(episode), img_dx, img_dy, new_map, True, new_org_maze, ))
        Saver_Procs.append(proc)
        proc.start()

        if solved_it:
            print("SOLVED :-)")

            tmp_maze = deepcopy(maze)
            grid.set_state(start_state)
            for i in range(POLICY_BAILOUT):
                grid.move(policy[grid.get_state()])
                if grid.game_over():
                    break
                gs = grid.get_state()
                print(V[s], gs)
                tmp_maze[gs[0]][gs[1]] = 3
            maze_generator.save_maze(tmp_maze, "mrunner_q_p_{0:0>5}.png".format(episode), img_dx, img_dy, rgb_color_map)
            break
        else:
            print("NOT SOLVED!")

# wait for all processes to finish
for saver_proc in Saver_Procs:
    saver_proc.join()
