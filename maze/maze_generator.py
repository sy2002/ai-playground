# maze_generator.py
#
# Run directly to generate 5 random mazes having random sizes or import
# it to use the various functions.
#
# I took this from http://code.activestate.com/recipes/578356-random-maze-generator/
# and then I changed it to fit my needs for my MazeRunner.
#
# done by sy2002 on 19th of May 2018
#
# Random Maze Generator using Depth-first Search
# http://en.wikipedia.org/wiki/Maze_generation_algorithm
# FB36 - 20130106

import numpy as np
import random
from PIL import Image

# generate a random maze of size mx x my, where "1" means "walkable way" and "0" means "wall"
# the output is a list of lists in the form maze[rows][columns]
def make_maze(mx, my):
    
    maze = [[0 for x in range(mx)] for y in range(my)]

    # 4 directions to move in the maze
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]

    # start the maze from a random cell
    cx = random.randint(0, mx - 1)
    cy = random.randint(0, my - 1)

    maze[cy][cx] = 1
    stack = [(cx, cy, 0)] # stack element: (x, y, direction)

    while len(stack) > 0:
        (cx, cy, cd) = stack[-1]
        # to prevent zigzags:
        # if changed direction in the last move then cannot change again
        if len(stack) > 2:
            if cd != stack[-2][2]: dirRange = [cd]
            else: dirRange = range(4)
        else: dirRange = range(4)

        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in dirRange:
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
                    ctr = 0 # of occupied neighbors must be 1
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 1: ctr += 1
                    if ctr == 1: nlst.append(i)

        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]; cy += dy[ir]; maze[cy][cx] = 1
            stack.append((cx, cy, ir))
        else: stack.pop()

    return maze

# return two tuples (row, col) that are as close as it can get to "top left = start and
# bottom right = end"
def find_start_end(maze):

    def search(search_start):
        ry = range(len(maze))
        rx = range(len(maze[0]))
        if not search_start:
            ry = reversed(ry)
            rx = reversed(rx)
        for y in ry:
            for x in rx:
                if maze[y][x] == 2 or maze[y][x] == 1:
                    return (x, y)

    return search(True), search(False)

# add start and end dots
def add_start_and_end(maze, color_map=[(0, 0, 0), (255, 255, 255), (255, 0, 0)]):
    start, end = find_start_end(maze)
    maze[start[1]][start[0]] = 2
    maze[end[1]][end[0]] = 2

# save a maze as PNG to a given file and size and use the given color coding
#
# optional: "add_mode" will treat 'maze' as a map of steps showing how the Q-algorithm
# (used e.g. in maze_runner_q.py) "walked" through the maze and how often it "stepped"
# on certain positions in the maze. In this case, "source_maze" is the raw maze to
# be shown as base image
def save_maze(maze, file, imgx, imgy, color_map=[(0, 0, 0), (255, 255, 255), (255, 0, 0)], add_mode=False, source_maze=None):
    mx = len(maze[0])
    my = len(maze)
    #square maze
    if mx == my:
        imgy_actual = imgy
    else:
        imgy_actual = int((my/mx) * imgx)

    image = Image.new("RGB", (imgx, imgy_actual))
    pixels = image.load()

    if add_mode:
        assert source_maze != None
        walk_maze = maze
        maze = source_maze
        s_start, s_end = find_start_end(maze)
        saturate_color = np.vectorize(lambda x: 255 if x > 255 else x)

        # find walk_col, which is a minimum color "addition" increment per walked step
        # and define col_offset, which is the minimum color to show to avoid a black-ish look
        min_saturation = 0.4 # at least 40% color saturation, otherwise the walk looks too dark
        max_walks = max(map(max, walk_maze)) # maximum amount of steps in the whole walk
        max_walks = min(max_walks, 255) # not more than 255
        walk_col = np.array(color_map[4]) * ((1.0 - min_saturation) / max_walks)
        col_offset = np.rint(min_saturation * np.array(color_map[4])).astype(int)

    # paint and save the maze
    for ky in range(imgy_actual):
        for kx in range(imgx):
            pixels[kx, ky] = color_map[maze[int(my * ky / imgy_actual)][int(mx * kx / imgx)]]

            if add_mode:
                mx_t = int(mx * kx / imgx)
                my_t = int(my * ky / imgy_actual)
                if s_start != (mx_t, my_t) and s_end != (mx_t, my_t):
                    temp_val = np.rint(walk_col * walk_maze[my_t][mx_t]).astype(int)
                    if np.sum(temp_val) > 0:
                        pixels[kx, ky] = tuple((saturate_color(col_offset + temp_val)))
                        
    image.save(file, "PNG")


# ===============================================================================================
# MAIN CODE (in case the file is run directly): Create 5 random mazes having random sizes
# ===============================================================================================

if __name__ == "__main__":

    # define amount of random mazes
    EPOCHS = 5

    # color start and end?
    color_start_and_end = True

    # define output image's x and y size
    # (if maze is not square, then y is adjusted to cater for non-squareness)
    imgx = 750
    imgy = 750

    # define the min/max properties of the random mazes
    mx_min = 10
    mx_max = 100
    my_min = 10
    my_max = 100
    maze_is_square = True # square maze based on mx_*, ignoring my_*

    # RGB colors of the maze: 0 = black for "wall" and 1 = white for "path"
    # 2 = red for "start and end"
    color = [(0, 0, 0), (255, 255, 255), (255, 0, 0)]

    for i in range(EPOCHS):
        mx = my = random.randint(mx_min, mx_max)
        if not maze_is_square:
            my = random.randint(my_min, my_max)
    
        maze = make_maze(mx, my)

        # add a start/end "dot"
        if color_start_and_end:
            add_start_and_end(maze, color)

        save_maze(maze, "tmp_Maze_" + str(mx) + "x" + str(my) + ".png", imgx, imgy, color)
