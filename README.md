## ai-playground
Unsorted Playground for Machine Learning, Reinforcement Learning and other AI Experiments.

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