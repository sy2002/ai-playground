# Solves the CartPole problem using Q-Learning and a RBF network
#
# The meta parameters below are chosen, so that on average, the system learns to
# balance "endlessly", i.e. 5,000 cycles. If this is not working at the first time,
# this might be due to the stochastic nature of the whole thing: Just run again.
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

RBF_EXEMPLARS       = 250           # amount of exemplar per "gamma instance" of the RBF network
RBF_GAMMA_COUNT     = 10            # amount of "gamma instances", i.e. EXEMPLARS x GAMMA_COUNT features
RBF_GAMMA_MIN       = 0.05          # minimum gamma, linear interpolation between min and max
RBF_GAMMA_MAX       = 4.0           # maximum gamma
RBF_SAMPLING        = 30000         # amount of samples to take for sampling the observation space and for init. RBFs

LEARN_EPISODES      = 200           # number of episodes to learn
TEST_EPISODES       = 10            # number of episodes that we use the visual rendering to test what we learned
GAMMA               = 0.999         # discount factor
ALPHA               = 0.75          # initial learning rate
ALPHA_DECAY         = 0.10          # learning rate decay
EPSILON             = 0.5           # epsilon-greedy algorithm
EPSILON_DECAY_t     = 0.1
EPSILON_DECAY_m     = 8             # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

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
# not care about the data you pass to it. This is a thing I did not know, when I
# programmed this in August 2018 and what I now found out in April 2019;
# see also this link: [@TODO add link to explanation repo]
observation_samples = []
for i in range(RBF_SAMPLING):
    env.reset()
    done = False
    while not done:
        observation, _, done, _ = env.step(env.action_space.sample())
        observation_samples.append(observation)
print("Done. Amount of observation samples: ", len(observation_samples))

# create RBF network and scaler
scaler = StandardScaler()
scaler.fit(observation_samples)
gammas = np.linspace(RBF_GAMMA_MIN, RBF_GAMMA_MAX, RBF_GAMMA_COUNT)
models = [RBFSampler(n_components=RBF_EXEMPLARS, gamma=g) for g in gammas]
transformer_list = []

for model in models:
    model.fit([observation_samples[0]]) # RBFSampler just needs the dimensionality, not the data itself
    transformer_list.append((str(model), model))

rbfs    = FeatureUnion(transformer_list)     #union of all RBF exemplar's output
rbf_net = [SGDRegressor(eta0=ALPHA, power_t=ALPHA_DECAY, learning_rate='invscaling', max_iter=5, tol=None)
           for i in range(len(env_actions))] #linear model that will work with the rbfs

def transform_s(s):
    return rbfs.transform(scaler.transform(np.array(s).reshape(1, -1)))

def transform_val(val):
    return np.array([val]).ravel()

# SGDRegressor needs "partial_fit" to be called before the first "predict" is called
for net in rbf_net:
    net.partial_fit(transform_s(env.reset()), transform_val(np.random.choice(env_actions)))

def get_Q_s_a(s, a):
    return rbf_net[a].predict(transform_s(s))

def set_Q_s_a(s, a, val):
    rbf_net[a].partial_fit(transform_s(s), transform_val(val))

def max_Q_s(s):
    max_val = float("-inf")
    max_idx = -1
    for a in env_actions:
        val = get_Q_s_a(s, a)
        if val > max_val:
            max_val = val
            max_idx = a
    return max_idx, max_val

t                   = 1.0
won_last_interval   = 0
steps_last_interval = 0

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
        new_value = old_qsa + (r + GAMMA * max_q_s2a2 - old_qsa)
        set_Q_s_a(s, a, new_value)

        # next state = current state, next action is the "Q-greedy" best action
        s = s2
        a = a2        

    if episode % PROBE == 0:
        if episode == 0:
            probe_divisor = 1
        else:
            probe_divisor = PROBE
        print("%d\t\t\t%0.2f\t\t\t%0.4f" % (episode, steps_last_interval / probe_divisor, eps))
        won_last_interval = 0
        steps_last_interval = 0

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
    while observation[0] > -2.4 and observation[0] < 2.4 and episode_step_count < 5000:
        episode_step_count += 1
        all_steps += 1
        a, _ = max_Q_s(observation)
        observation, _, done, _ = env.step(a)
        env.render()
    
    print("%d\t\t%d\t\t" % (episode, episode_step_count))

print("Avg. Steps: %0.2f" % (all_steps / TEST_EPISODES))
env.close()
