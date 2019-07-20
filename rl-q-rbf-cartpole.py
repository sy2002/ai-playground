# Solves the CartPole problem using Q-Learning and a RBF Network
#
# The meta parameters below are chosen, so that on average, the system needs
# 220 episodes to learn to balance "endlessly", i.e. 5,000 cycles. If this is not working
# at the first time, this might be due to the stochastic nature of the whole thing: Just run again.
#
# uses http://gym.openai.com/envs/CartPole-v1
# inspired by Lecture 17 of Udemy/Lazy Programmer Inc/Advanced AI: Deep Reinforcement Learning
#
# done by sy2002 on 11th of August 2018 and tuned on 17th of April 2019

import gym
import gym.spaces
import numpy as np

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

# Basic idea of this "RBF Network" solution:
#
# * CartPole is offering 4 features: Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip.
# * When we use a collection of (aka "network") of Radial Basis Functions (RBFs), then we can transform
#   these 4 features into n distances from the centers of the RBFs, where n = RBF_EXEMPLARS x RBF_GAMMA_COUNT
# * The "Gamma" parameter controls the "width" of the RBF's bell curve. The idea is, to use multiple RBFSamplers
#   with multiple widths to construct a network with a good variety to sample from.
# * The RBF transforms the initial 4 features into plenty of features and therefore offers a lot of "variables"
#   or something like "resolution", where a Linear Regression algorithm can calcluate the weights for.
#   In contrast, the original four features of the observation space would yield a too "low resolution".
# * The Reinforcement Learning algorithm used to learn to balance the pole is Q-Learning.
# * The "State" "s" of the Cart is obtained by using the above-mentioned RBF Network to transform the four
#   original features into the n distances from a randomly chosen amount of RBF centers (aka "Exemplars").
#   Note that it is absolutely OK, that the Exemplars are chosen randomly, because each Exemplar defines one
#   possible combination of (Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip); and therefore
#   having lots of those random Exemplars just gives us the granularity ("resolution") we need for our Linear Regression.
# * The possible "Actions" "a" of the system are "push from left" or "push from right", aka 0 and 1.
# * For each possible action we are calculating one Value Function using Linear Regression. It can be illustrated by
#   asking the question: "For the state we are currently in, defined by the distances to the Exemplars, what is the
#   Value Function for pushing from left or puhsing from right. The larger one wins."

# On RBFSampler:
#
# Despite its name, the RBFSampler is actually not using any RBFs inside. I did some experiments about
# this fact. Go to https://github.com/sy2002/ai-playground/blob/master/RBFSampler-Experiment.ipynb 
# to learn more.

RBF_EXEMPLARS       = 250           # amount of exemplars per "gamma instance" of the RBF network
RBF_GAMMA_COUNT     = 10            # amount of "gamma instances", i.e. RBF_EXEMPLARS x RBF_GAMMA_COUNT features
RBF_GAMMA_MIN       = 0.05          # minimum gamma, linear interpolation between min and max
RBF_GAMMA_MAX       = 4.0           # maximum gamma
RBF_SAMPLING        = 30000         # amount of samples to take for sampling the observation space for fitting the Scaler

LEARN_EPISODES      = 220           # number of episodes to learn
TEST_EPISODES       = 10            # number of episodes that we use the visual rendering to test what we learned
GAMMA               = 0.999         # discount factor for Q-Learning
ALPHA               = 0.75          # initial learning rate
ALPHA_DECAY         = 0.10          # learning rate decay
EPSILON             = 0.5           # randomness for epsilon-greedy algorithm (explore vs exploit)
EPSILON_DECAY_t     = 0.1
EPSILON_DECAY_m     = 4             # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

PROBE               = 20            # after how many episodes we will print the status

print("\nCartPole-v1 solver by sy2002 on 17th of April 2019")
print("==================================================\n")
print("Sampling observation space by playing", RBF_SAMPLING, "random episodes...")


# create environment
env = gym.make('CartPole-v1')
env_actions = [0, 1] # needs to be in ascending order with no gaps, e.g. [0, 1]

# CartPole has an extremely large observation space, as Gym's observation_space high/low
# attributes are basically returning the whole floating point range as possible values
# for some of the parameters.
# So let's sample possible values by running an amount of experiments.
# Caveat: This is only sampling a small amount of possible values, because running
# the CartPole randomly does not explore too much of the space.
# Nevertheless: We need this only for the scaler, as RBFSampler's fit function does
# not care about the data you pass to it.
observation_samples = []
for i in range(RBF_SAMPLING):
    env.reset()
    done = False
    while not done:
        observation, _, done, _ = env.step(env.action_space.sample())
        observation_samples.append(observation)
print("Done. Amount of observation samples: ", len(observation_samples))

# create scaler and fit it to the sampled observation space
scaler = StandardScaler()
scaler.fit(observation_samples)

# the RBF network is built like this: create as many RBFSamplers as RBF_GAMMA_COUNT
# and do so by setting the "width" parameter GAMMA of the RBFs as a linear interpolation
# between RBF_GAMMA_MIN and RBF_GAMMA_MAX
gammas = np.linspace(RBF_GAMMA_MIN, RBF_GAMMA_MAX, RBF_GAMMA_COUNT)
models = [RBFSampler(n_components=RBF_EXEMPLARS, gamma=g) for g in gammas]

# we will put all these RBFSamplers into a FeatureUnion, so that our Linear Regression
# can regard them as one single feature space spanning over all "Gammas"
transformer_list = []
for model in models:
    model.fit([observation_samples[0]]) # RBFSampler just needs the dimensionality, not the data itself
    transformer_list.append((str(model), model))
rbfs = FeatureUnion(transformer_list)     #union of all RBF exemplar's output

# we use one Linear Regression per possible action to model the Value Function for this action,
# so rbf_net is a list; SGDRegressor allows step-by-step regression using partial_fit, which
# is exactly what we need to learn
rbf_net = [SGDRegressor(eta0=ALPHA, power_t=ALPHA_DECAY, learning_rate='invscaling', max_iter=5, tol=float("-inf"))
           for i in range(len(env_actions))]

# transform the 4 features (Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip)
# into RBF_EXEMPLARS x RBF_GAMMA_COUNT distances from the RBF centers ("Exemplars")
def transform_s(s):
    return rbfs.transform(scaler.transform(np.array(s).reshape(1, -1)))

# SGDRegressor expects a vector, so we need to transform our action, which is 0 or 1 into a vector
def transform_val(val):
    return np.array([val]).ravel()

# SGDRegressor needs "partial_fit" to be called before the first "predict" is called
# so let's do that with random (dummy) values for state and action
for net in rbf_net:
    net.partial_fit(transform_s(env.reset()), transform_val(np.random.choice(env_actions)))

# get the result of the Value Function for action a taken in state s
def get_Q_s_a(s, a):
    return rbf_net[a].predict(transform_s(s))

# learn Value Function for action a in state s
def set_Q_s_a(s, a, val):
    rbf_net[a].partial_fit(transform_s(s), transform_val(val))

# find the best action to take in state s
def max_Q_s(s):
    max_val = float("-inf")
    max_idx = -1
    for a in env_actions:
        val = get_Q_s_a(s, a)
        if val > max_val:
            max_val = val
            max_idx = a
    return max_idx, max_val

t                   = 1.0           # used to decay epsilon
won_last_interval   = 0             # for statistical purposes
steps_last_interval = 0             # dito

print("\nLearn:")
print("======\n")
print("Episode\t\t\tAvg. Steps\t\tepsilon")

for episode in range(LEARN_EPISODES + 1):
    # let epsilon decay over time
    if episode % EPSILON_DECAY_m == 0:
        t += EPSILON_DECAY_t

    # each episode starts again at the begining
    s = env.reset()
    a = np.random.choice(env_actions)

    episode_step_count = 0

    done = False
    while not done:
        episode_step_count += 1
        steps_last_interval += 1

        # epsilon-greedy: do a random move with the decayed EPSILON probability
        p = np.random.random()
        eps = EPSILON / t
        if p > (1 - eps):
            a = np.random.choice(env_actions)

        # exploit or explore and collect reward
        observation, r, done, _ = env.step(a)
        s2 = observation

        # Q-Learning
        old_qsa = get_Q_s_a(s, a)
        # if this is not the terminal position (which has no actions), then we can proceed as normal
        if not done:
            # "Q-greedily" grab the gain of the best step forward from here
            a2, max_q_s2a2 = max_Q_s(s2)

        # else change the rewards in the terminal position to have a clearer bias for "longevity"
        else:
            if episode_step_count < 100:
                r = -100
            elif episode_step_count < 200:
                r = 100
            elif episode_step_count < 300:
                r = 150
            elif episode_step_count < 400:
                r = 200
            elif episode_step_count <= 500:
                r = 300
            else:
                r = 1000
            a2 = a
            max_q_s2a2 = 0 # G (future rewards) of terminal position is 0 although the reward r is high

        # learn the new value for the Value Function
        new_value = old_qsa + (r + GAMMA * max_q_s2a2 - old_qsa)
        set_Q_s_a(s, a, new_value)

        # next state = current state, next action is the "Q-greedy" best action
        s = s2
        a = a2        

    # every PROBEth episode: print status info
    if episode % PROBE == 0:
        if episode == 0:
            probe_divisor = 1
        else:
            probe_divisor = PROBE
        print("%d\t\t\t%0.2f\t\t\t%0.4f" % (episode, steps_last_interval / probe_divisor, eps))
        won_last_interval = 0
        steps_last_interval = 0

# now, after we learned how to balance the pole: test it and use the visual output of Gym
print("\nTest:")
print("=====\n")
print("Episode\tSteps\tResult")

all_steps = 0
for episode in range(TEST_EPISODES):
    observation = env.reset()
    env.render()
    done = False
    won = False
    episode_step_count = 0
    # we are ignoring Gym's "done" function, so that we can run the system longer
    while observation[0] > -2.4 and observation[0] < 2.4 and episode_step_count < 5000:
        episode_step_count += 1
        all_steps += 1
        a, _ = max_Q_s(observation)
        observation, _, done, _ = env.step(a)
        env.render()
    
    print("%d\t\t%d\t\t" % (episode, episode_step_count))

print("Avg. Steps: %0.2f" % (all_steps / TEST_EPISODES))
env.close()
