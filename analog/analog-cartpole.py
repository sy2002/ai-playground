
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

LEARN_EPISODES      = 1000          # number of episodes to learn
TEST_EPISODES       = 10            # number of episodes that we use the visual rendering to test what we learned
GAMMA               = 0.999         # discount factor for Q-Learning
ALPHA               = 0.75          # initial learning rate
ALPHA_DECAY         = 0.10          # learning rate decay
EPSILON             = 0.5           # randomness for epsilon-greedy algorithm (explore vs exploit)
EPSILON_DECAY_t     = 0.1
EPSILON_DECAY_m     = 6             # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

IMPULSE_DURATION    = 10            # impulse duration in milliseconds

PROBE               = 20            # after how many episodes we will print the status

from readchar import readchar
import serial
import threading
from time import sleep
import sys

import numpy as np

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

HC_PORT = '/dev/cu.usbserial-DN050L1O'

HC_SIM_X_POS        = "0261"
HC_SIM_X_VEL        = "0260"
HC_SIM_ANGLE        = "0263"
HC_SIM_ANGLE_VEL    = "0160"

HC_SIM_DIRECTION_1  = "D0"
HC_SIM_DIRECTION_0  = "d0"
HC_SIM_IMPULSE_1    = "D1"
HC_SIM_IMPULSE_0    = "d1"


try:
    ser = serial.Serial(    port=HC_PORT,
                            baudrate=115200,
                            bytesize=8, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                            rtscts=False,
                            timeout=0.02)
except:
    print("ERROR: SERIAL PORT CANNOT BE OPENED.")
    sys.exit(1)

def hc_send(cmd):
    ser.write(cmd.encode("ASCII"))

def hc_receive():
    line = ser.readline().decode("ASCII")
    line = line.split("\n")[0]
    return line
#    return ser.readline().decode("ASCII").split("\n")[0]

def res2float(str):
    f = 0
    try:
        f = float(str.split(" ")[0])
    except:
        print("ERROR IN FLOAT CONVERSION:", str)
        exit(1)
    return f

def hc_get_value(addr):
    hc_send("g")
    hc_send(addr)
    return str2float(hc_receive().split(" ")[0])

def hc_get_sim_state():
    hc_send("g")
    hc_send(HC_SIM_X_POS)
    sleep(0.001)
    hc_send("g")
    hc_send(HC_SIM_X_VEL)    
    sleep(0.001)
    hc_send("g")    
    hc_send(HC_SIM_ANGLE)
    sleep(0.001)
    hc_send("g")    
    hc_send(HC_SIM_ANGLE_VEL)
    sleep(0.001)
    return (    res2float(hc_receive()),
                res2float(hc_receive()),
                res2float(hc_receive()),
                res2float(hc_receive()))

def hc_reset_sim():
    hc_send("i")
    sleep(0.05)
    hc_send("o")
    sleep(0.05)        
    ser.flushInput()

def hc_influence_sim(a):
    if (a == 1):
        hc_send(HC_SIM_DIRECTION_1)
    else:
        hc_send(HC_SIM_DIRECTION_0)

    duration = IMPULSE_DURATION / 1000.0

    hc_send(HC_SIM_IMPULSE_1)
    sleep(duration)
    hc_send(HC_SIM_IMPULSE_0)

# Reset HC
received = ""
while (received != "RESET"):
    print("Reset attempt")
    hc_send("x")
    sleep(1)
    received = hc_receive()
    if received != "":
        print("Received:", received)

"""
i = 0
while True:
    if i == 0:
        hc_reset_sim()
    print(hc_get_sim_state())
    hc_send("h")
    readchar()
    hc_send("o")
    sleep(0.03)
    ser.flushInput()
    i = 1
    #i += 1 % 1000
"""


print("\nCartPole-v1 solver by sy2002 on 20th of July 2019")
print("=================================================\n")
print("Sampling observation space by playing", RBF_SAMPLING, "random episodes...")


# create environment
env_actions = [0, 1] # needs to be in ascending order with no gaps, e.g. [0, 1]

# create scaler and fit it to the sampled observation space
#scaler = StandardScaler()
#scaler.fit(observation_samples)

# the RBF network is built like this: create as many RBFSamplers as RBF_GAMMA_COUNT
# and do so by setting the "width" parameter GAMMA of the RBFs as a linear interpolation
# between RBF_GAMMA_MIN and RBF_GAMMA_MAX
gammas = np.linspace(RBF_GAMMA_MIN, RBF_GAMMA_MAX, RBF_GAMMA_COUNT)
models = [RBFSampler(n_components=RBF_EXEMPLARS, gamma=g) for g in gammas]

# we will put all these RBFSamplers into a FeatureUnion, so that our Linear Regression
# can regard them as one single feature space spanning over all "Gammas"
transformer_list = []
for model in models:
    model.fit([[1.0, 1.0, 1.0, 1.0]]) # RBFSampler just needs the dimensionality, not the data itself
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
    return rbfs.transform(np.array(s).reshape(1, -1))

# SGDRegressor expects a vector, so we need to transform our action, which is 0 or 1 into a vector
def transform_val(val):
    return np.array([val]).ravel()

# SGDRegressor needs "partial_fit" to be called before the first "predict" is called
# so let's do that for all actions (aka entries in the rbf_net list) with random (dummy) values
# for the state and the Value Function
for net in rbf_net:
    random_val_func = np.random.rand() * np.random.randint(1, 100) # random float 0 < x < 100
    net.partial_fit(transform_s([0.5, 0.1, -0.5, 0.1]), transform_val(random_val_func))

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
print("Episode\t\t\tAvg. Steps\t\tMax Steps\t\tepsilon")

episode_sim_max = [-1000, -1000, -1000, -1000]
episode_sim_min = [1000, 1000, 1000, 1000]

"""
Max: [0.9283, 1.2565, 1.2565, 1.1949]
Min: [-0.9449, -1.2537, -1.2537, -0.9689]
"""

episode_step_max = 0
episode_step_count = 0

for episode in range(LEARN_EPISODES + 1):
    # let epsilon decay over time
    if episode % EPSILON_DECAY_m == 0:
        t += EPSILON_DECAY_t

    # each episode starts again at the begining
    hc_reset_sim()
    s = hc_get_sim_state()
    a = np.random.choice(env_actions)

    if episode_step_max < episode_step_count:
        episode_step_max = episode_step_count

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

        hc_influence_sim(a)
        observation = hc_get_sim_state()
        r = 1

        for i in range(4):
            if episode_sim_max[i] < observation[i]:
                episode_sim_max[i] = observation[i]
            if episode_sim_min[i] > observation[i]:
                episode_sim_min[i] = observation[i]

        # x-position [0] or angle [1] invalid
        done = abs(observation[0]) > 0.9 or abs(observation[2]) > 1.0  

        s2 = observation

        # Q-Learning
        old_qsa = get_Q_s_a(s, a)
        # if this is not the terminal position (which has no actions), then we can proceed as normal
        if not done:
            # "Q-greedily" grab the gain of the best step forward from here
            a2, max_q_s2a2 = max_Q_s(s2)

        # else change the rewards in the terminal position to have a clearer bias for "longevity"
        else:
            if episode_step_count < 20:
                r = -100
            elif episode_step_count < 50:
                r = -50
            elif episode_step_count < 70:
                r = -20
            elif episode_step_count < 100:
                r = -10
            elif episode_step_count < 120:
                r = 10
            elif episode_step_count < 150:
                r = 100
            elif episode_step_count < 200:
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
            episode_step_max = episode_step_count
        else:
            probe_divisor = PROBE
        print("%d\t\t\t%0.2f\t\t\t%d\t\t\t%0.4f" % (episode, steps_last_interval / probe_divisor, episode_step_max, eps))
        won_last_interval = 0
        steps_last_interval = 0
        episode_step_max = 0
        #print("Max:", episode_sim_max)
        #print("Min:", episode_sim_min)


# now, after we learned how to balance the pole: test it and use the visual output of Gym
print("\nTest:")
print("=====\n")

# use DISTURB_PROB to switch a probabilistic system disturbance:
# if you set it to a value larger then 0.0, the system will be disturbed
# with this probability for an amount of steps given by DISTURB_DURATION
DISTURB_PROB = 0.00
DISTURB_DURATION = 5        # needs to be at least 1; suggestion: try 5 in combination with 0.01 probability

if DISTURB_PROB > 0.0:
    print("Disturb-Mode ON! Probability = %0.4f  Duration = %d\n" % (DISTURB_PROB, DISTURB_DURATION))
print("Episode\tSteps\tResult")

all_steps = 0
for episode in range(TEST_EPISODES):
    observation = env.reset()
    env.render()
    done = False
    won = False
    episode_step_count = 0
    dist_ongoing = 0

    # we are ignoring Gym's "done" function, so that we can run the system longer
    while observation[0] > -2.4 and observation[0] < 2.4 and episode_step_count < 5000:
        episode_step_count += 1
        all_steps += 1

        a, _ = max_Q_s(observation)
        if DISTURB_PROB > 0.0:            
            if dist_ongoing == 0 and np.random.rand() > (1.0 - DISTURB_PROB):
                dist_ongoing = DISTURB_DURATION
            if dist_ongoing > 0:
                print("\tstep #%d:\tdisturbance ongoing: %d" % (episode_step_count, dist_ongoing))
                dist_ongoing -= 1
                a = np.random.choice(env_actions)

        observation, _, done, _ = env.step(a)
        env.render()
    
    print("%d\t\t%d\t\t" % (episode, episode_step_count))

print("Avg. Steps: %0.2f" % (all_steps / TEST_EPISODES))
env.close()
