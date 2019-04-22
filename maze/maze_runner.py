# Maze Runner using Reinforcement Learning
#
# Generates a random maze and then uses a uniform random walk policy and
# the Iterative Policy Evaluation algorithm to solve the maze.
# The initial maze and the result are stored as a PNG image.
#
# inspired by Lecture 36 of Udemy/Lazy Programmer Inc/Reinforcement Learning
#
# done by sy2002 on 19th of May 2018

import grid_world
import maze_generator
import numpy as np

# maze size
mx = 40
my = 40

# imgage size
img_dx = 500
img_dy = 500

WIN_REWARD   = 1e12         # needs to be large due to float accuracy
STEP_PUNISH  = -0.1         # cost of "walking"
SMALL_ENOUGH = 1e-3         # threshold for convergence
gamma        = 0.999        # discount factor

# index 0 = wall, index = 1 path, index 2 = start/end, index = shortest path from start to end
rgb_color_map = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]

# generate maze and save the "unsolved" maze
maze = maze_generator.make_maze(mx, my)
maze_generator.add_start_and_end(maze, rgb_color_map)
maze_generator.save_maze(maze, "mrunner_1.png", img_dx, img_dy, rgb_color_map)

# move the maze to "Grid World" by doing the following:
# 1. Make sure that each step "costs something", so set everything to a reward of STEP_PUNISH
# 2. Reward the end point with WIN_REWARD
# 3. For each state aka position: define the possible movements (based on the maze layout)
# 4. Make the  end point a terminal point

start, end = maze_generator.find_start_end(maze)
grid = grid_world.Grid(mx, my, (start[1], start[0]))

rewards = {}
for y in range(my):
    for x in range(mx):
        if maze[y][x] == 1:                 # 1 means "path", 0 means "wall"
            rewards[(y, x)] = STEP_PUNISH   # make sure, each step "costs something"
rewards[(end[1], end[0])] = WIN_REWARD      # the goal offers 10.0 reward
        
# create possible actions for each position
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
actions.pop((end[1], end[0])) # make the end point a terminal state

grid.set(rewards, actions)

# =================================================================
# Execute the iterative policy evaluation
# =================================================================


states = grid.all_states()
V = {} # initialize V(s) = 0
for s in states:
    V[s] = 0

# repeat until convergence
verbose_counter = 0
while True:
    biggest_change = 0
    for s in states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        if s in grid.actions:
            new_v = 0 # we will accumulate the answer
            p_a = 1.0 / len(grid.actions[s]) # each action has equal probability
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                new_v += p_a * (r + gamma * V[grid.current_state()])
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    verbose_counter += 1
    if verbose_counter % 100 == 0:
        print(biggest_change)
    if biggest_change < SMALL_ENOUGH:
        break

V[(end[1], end[0])] = WIN_REWARD

# =================================================================
# Paint shortest path by following the value function
# =================================================================

grid.set_state((start[1], start[0]))
n = 0
while True:
    best_action = None
    best_action_value = -999999999
    for a in grid.actions[grid.get_state()]:
        grid.move(a)
        if V[grid.get_state()] > best_action_value:
            best_action_value = V[grid.get_state()]
            best_action = a
        grid.undo_move(a)
    if (best_action is not None):
        grid.move(best_action)

    gs = grid.get_state()

    if (gs[1], gs[0]) == (end[1], end[0]):
        break

    maze[gs[0]][gs[1]] = 3
    print(gs, V[gs])

    n = n + 1
    maze_generator.save_maze(maze, "mrunner_2_{0:0=3d}.png".format(n), img_dx, img_dy, rgb_color_map)

# save the last frame a couple of times so that when we make an animated GIF, the last frame stands still for a while
for z in range(25):
    maze_generator.save_maze(maze, "mrunner_2_{0:0=3d}.png".format(n + z), img_dx, img_dy, rgb_color_map)