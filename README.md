## ai-playground

Unsorted Playground for Machine Learning, Reinforcement Learning and other AI Experiments.

### Install the dependencies

`requirements.txt` contains the dependencies of all ai-playground programs. You might
want to create a virtual environment for sparing your local Python installation. Switch
to the root folder of `ai-playgound` and enter the following commands:

```
python3 -m venv ai-p-env
source ai-p-env/bin/activate
pip install -r requirements.txt
```

When the virtual environment is activated, `pip` and `python` should now point to
a Python 3 installation and you should be able to run the programs using `python`. You
can activate the environment at any later moment in time using `source ai-p-env/bin/activate`
from the project's root folder.

### Handwritten Digits Recognizer using a Perceptron Network

<img align="left" width="200" height="150" src="http://www.sy2002.de/nn_screenshot.jpg">

The HWD-Perceptron project was my first attempt to learn how a Perceptron works by implementing
it from scratch. I used the very well known MIST dataset to do handwritten digits recognition.

The project is a ready-to-go Python app that lets you learn more about Neural Networks and
Machine Learning in general and more about simple Perceptron based handwritten digits
recognition in particular.

Go to the [HWD-Perceptron Repository](https://github.com/sy2002/HWD-Perceptron) to learn more.

### Solve Mazes

<img align="right" width="150" height="150" src="http://www.sy2002.de/maze/ani1-20x20.gif">

The programs `maze_runner.py` and `maze_runner_q_mp.py` in the folder [maze](maze) are
using Reinforcement Learning to solve mazes of arbitrary sizes.
Have a look at these [Demos](http://www.sy2002.de/maze) to get an idea.

### OpenAI Gym's CartPole

[OpenAI Gym](https://gym.openai.com/) is a fantastic place to practice Reinforment Learning.
The classical CartPole balancing challenge has kind of a "Hello World" status.
Here is an [animation](http://gym.openai.com/envs/CartPole-v1/) of how the challenge looks like.
My solution [rl-q-rbf-cartpole.py](rl-q-rbf-cartpole.py) uses a Radial Basis Function network
to transform the four features of the Cart (Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip)
into a large amount of distances from the centers of RBF "Exemplars" and then use Linear Regression
to learn the Value Function using Q-Learning.

### RBFSampler actually is not using any Radial Basis Functions (RBFs)

As a naive beginner, I thought that scikit-learn's RBFSampler is basically a convenient way
to create a collection (or "network") of multiple Radial Basis Functions with random centers. Well, I
was wrong as [RBFSampler-Experiment.ipynb](RBFSampler-Experiment.ipynb) shows, but in the end,
everything is still kind of as you would expect from the name RBFSampler.

You can also try the experiment live and interactively on Kaggle using
[this Kaggle Kernel](https://www.kaggle.com/sy2002/rbfsampler-actually-is-not-using-any-rbfs).
Here is an [Article on Medium](https://medium.com/@mirkoholzer/rbfsampler-actually-is-not-using-any-radial-basis-functions-rbfs-771cd38a7a6f) that nicely explains the whole topic.

### ANALOG-CARTPOLE - A hybrid analog/digital computing experiment

<img align="left" width="200" height="150" src="http://sy2002.de/analog-model-1.jpg">

This is a hybrid computing experiment: Use the analog computer
"Model-1" by Analog Paradigm (http://analogparadigm.com) to simulate an
inverse pendulum. In parallel, use a digital computer to execute a digital
Reinforcement Learning algorithm that controls the analog computer.

Switch to the folder [analog](analog) to learn more or watch a short
[introduction to this experiment on YouTube](https://youtu.be/jDGLh8YWvNE).
