## Maze

Using Reinforcement-Learning to solve mazes of arbitrary sizes. Use the variables `mx` and `my`
in the Python code to configure the mazes' size in blocks and use `img_dx` and `img_dy` to
specify the amount of pixels of the output image. Here is an example maze featuring
40x40 blocks in 500x500 pixels:

![Maze Sample Image](maze_runner_sample.gif)

### Main Files

* `maze_runner.py`: Generates a random maze and then uses a **Uniform Random Walk Policy** and
  the **Iterative Policy Evaluation** algorithm to solve the maze. The empty maze and the solution
  is stored as a result a series of PNG images.
* `maze_runner_q_mp.py`: Generates a random maze and then uses **Q-Learning** to solve the maze.
  This version of the solution also utilizes Python's multi-processing capabilities to speed-up
  image saving. The results are a numbered series of PNG files. Assumptions to make it more "realistic"
  than the "godmode" dynamic programming version that you find in the other source `maze_runner.py`

### Helper Files

* `grid_world.py`: Helper class taken from the Udemy RL course: It represents a Grid World consisting
  basically of two dictionaries: one containing all possible actions at each position of the grid and
  one containing the rewards used for the RL algorithm.
* `maze_generator.py`: A set of helper functions to generate random mazes in the form of two
  dimensional lists and to save them as PNG files.

### How to Create Animated Mazes

The animated mazes shown at [http://www.sy2002.de/maze](http://www.sy2002.de/maze) have been
generated using `maze_runner.py`. The output are many PNG files named `mrunner_2_*.png`.
If you have [ImageMagick](https://www.imagemagick.org/) installed, you can use the
following commands to create an Animated GIF:

```
python maze_runner.py
convert -loop 0 -delay 10 mrunner_2_*.png animated.gif
rm *.png
```

The more advanced `maze_runner_q_mp.py` could also be used to generate animated mazes.
But by default, it is generating "heatmaps" of the Q-Learning's "learning walk" instead.
If you want `maze_runner_q_mp.py` to generate animated mazes instead, then
[this](https://github.com/sy2002/ai-playground/blob/master/maze/maze_runner_q_mp.py#L280)
line of code has to be changed: Move it inside the for-loop and use `i` instead of
`episode` as format variable.