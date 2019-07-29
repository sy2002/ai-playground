# ANALOG-CARTPOLE - A hybrid analog/digital computing experiment
#
# Use digital Reinforcement Learning to learn to balance an inverse pendulum
# on a cart simulated by a Model-1 by Analog Paradigm (http://analogparadigm.com)
#
# Analog part done by vaxman on 2019-07-27
# Digital part done by sy2002 on 2019-07-27

# if you don't have a Model-1 at hand, set SOFTWARE_ONLY to True
# to use a software based physics simulation by OpenAI Gym
SOFTWARE_ONLY     = True

print("\nAnalog Cartpole - A hybrid analog/digital computing experiment")
print("==============================================================\n")

import numpy as np
import serial
import sys

from readchar import readchar
from time import sleep

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

if (SOFTWARE_ONLY):
    import gym
    import gym.spaces
    print("WARNING: Software-only mode. No analog computer is being used.\n")

# ----------------------------------------------------------------------------
# Analog Setup
# ----------------------------------------------------------------------------

# Hybrid Controller serial setup
HC_PORT             = '/dev/cu.usbserial-DN050L1O'
HC_BAUD             = 115200        
HC_BYTE             = 8
HC_PARITY           = serial.PARITY_NONE
HC_STOP             = serial.STOPBITS_ONE
HC_RTSCTS           = False
HC_TIMEOUT          = 0.02          # increase to e.g. 0.05, if you get error #2

HC_IMPULSE_DURATION = 10            # analog impulse duration in milliseconds

HC_SIM_X_POS        = "0261"        # address of cart's x-position
HC_SIM_X_VEL        = "0260"        # address of cart's x-velocity
HC_SIM_ANGLE        = "0263"        # address of pole's/pendulum's angle
HC_SIM_ANGLE_VEL    = "0160"        # address of pole's/pendulum's angular velocity

HC_SIM_DIRECTION_1  = "D0"          # digital out #0=hi: direction = 1, e.g. right
HC_SIM_DIRECTION_0  = "d0"          # digital out #0=lo: direction = 0, e.g. left
HC_SIM_IMPULSE_1    = "D1"          # digital out #1=hi: apply force
HC_SIM_IMPULSE_0    = "d1"          # digital out #1=lo: stop applying force

HC_CMD_RESET        = "x"           # reset hybrid controller, try multiple times until response received
HC_CMD_INIT         = "i"           # initial condition (i.e. pendulum is upright)
HC_CMD_OP           = "o"           # start to operate
HC_CMD_HALT         = "h"           # halt/pause (can be resumed by HC_CMD_OP)
HC_CMD_GETVAL       = "g"           # set address of computing element and return value and ID

HC_RSP_RESET        = "RESET"       # HC response on HC_CMD_RESET

# ----------------------------------------------------------------------------
# RL Meta Parameters
# ----------------------------------------------------------------------------

RBF_EXEMPLARS       = 250           # amount of exemplars per "gamma instance" of the RBF network
RBF_GAMMA_COUNT     = 10            # amount of "gamma instances", i.e. RBF_EXEMPLARS x RBF_GAMMA_COUNT features
RBF_GAMMA_MIN       = 0.05          # minimum gamma, linear interpolation between min and max
RBF_GAMMA_MAX       = 4.0           # maximum gamma
RBF_SAMPLING        = 100           # amount of episodes to learn for initializing the scaler

LEARN_EPISODES      = 1000          # number of episodes to learn
TEST_EPISODES       = 10            # number of episodes that we use the visual rendering to test what we learned
GAMMA               = 0.999         # discount factor for Q-Learning
ALPHA               = 0.75          # initial learning rate
ALPHA_DECAY         = 0.10          # learning rate decay
EPSILON             = 0.5           # randomness for epsilon-greedy algorithm (explore vs exploit)
EPSILON_DECAY_t     = 0.1
EPSILON_DECAY_m     = 6             # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

PROBE               = 20            # after how many episodes we will print the status

# ----------------------------------------------------------------------------
# Serial protocol functions for Model-1 Hybrid Controller
# ----------------------------------------------------------------------------

if not SOFTWARE_ONLY:
    try:
        hc_ser = serial.Serial( port=HC_PORT,
                                baudrate=HC_BAUD,
                                bytesize=HC_BYTE, parity=HC_PARITY, stopbits=HC_STOP,
                                rtscts=HC_RTSCTS,
                                timeout=HC_TIMEOUT)
    except:
        print("ERROR #1: SERIAL PORT CANNOT BE OPENED.")
        sys.exit(1)

def hc_send(cmd):
    hc_ser.write(cmd.encode("ASCII"))

def hc_receive():
    # HC ends each communication with "\n", so we can conveniently use readline
    return hc_ser.readline().decode("ASCII").split("\n")[0]

# when using HC_CMD_GETVAL, HC returns "<value><space><id/type>\n"
# we ignore <type> but we expect a well formed response
def hc_res2float(str):
    f = 0
    try:
        f = float(str.split(" ")[0])
        return f
    except:
        print("ERROR #2: FLOAT CONVERSION:", str)
        sys.exit(2)

# ask for a value and give the system 1ms to return it
def hc_ask_for_value(addr):
    hc_send(HC_CMD_GETVAL + addr)
    sleep(0.001)

# Ask for the 4 relevant values that the simulation state consists of
# and let the serial buffer be filled with the results. Then read four
# "lines" from the serial buffer that are containing the results and
# convert them to floats.
# We are doing it this way (instead of fetching each value one at a time),
# because for some reason pySerial seems to be slow while reading
def hc_get_sim_state():
    hc_ask_for_value(HC_SIM_X_POS)
    hc_ask_for_value(HC_SIM_X_VEL)    
    hc_ask_for_value(HC_SIM_ANGLE)
    hc_ask_for_value(HC_SIM_ANGLE_VEL)
    return (    hc_res2float(hc_receive()),
                hc_res2float(hc_receive()),
                hc_res2float(hc_receive()),
                hc_res2float(hc_receive()))

# bring the simulation back to the initial condition (pendulum is upright)
def hc_reset_sim():
    hc_send(HC_CMD_INIT)
    sleep(0.05)         #time for condensators to recharge
    hc_send(HC_CMD_OP)
    sleep(0.05)         #time for the feedback string to arrive

    #TODO: instead of just flushing the serial input buffer:
    #read and expect the correct feedback from HC
    hc_ser.flushInput()

# influence simulation by using an impulse to push the cart to the left or to
# the right; it does not matter if "1" means left or right as long as "0" means
# the opposite of "1"
def hc_influence_sim(a):
    if (a == 1):
        hc_send(HC_SIM_DIRECTION_1)
    else:
        hc_send(HC_SIM_DIRECTION_0)
   
    hc_send(HC_SIM_IMPULSE_1)
    sleep(HC_IMPULSE_DURATION / 1000.0)
    hc_send(HC_SIM_IMPULSE_0)

# ----------------------------------------------------------------------------
# Prepare environment / simulation
# ----------------------------------------------------------------------------

# list of possible actions that the RL agent can perform in the environment
env_actions = [0, 1] # needs to be in ascending order with no gaps, e.g. [0, 1]

# prepare OpenAI simulation
if SOFTWARE_ONLY:
    env = gym.make('CartPole-v1')

# prepare analog simulation by resetting the HC
else:
    received = ""
    while (received != HC_RSP_RESET):
        print("Hybrid Controller reset attempt...")
        hc_send(HC_CMD_RESET)
        sleep(1)
        received = hc_receive()
        if received != "":
            print("  received:", received)

    # Trivial "single step debugger" to find out (by looking at the oscilloscope)
    # the right limits for the x position and the pendulum's angle.
    # Comment it in, if you'd like to experiment
    """
    while True:
        if i == 0:
            hc_reset_sim()
        print(hc_get_sim_state())
        hc_send(HC_CMD_HALT)
        readchar()
        hc_send(HC_CMD_OP)
        sleep(0.03)
        hc_ser.flushInput()
    """

# ----------------------------------------------------------------------------
# TODO Continue code cleanup and refactoring to have one switchable
# code base between the analog simulator and OpenAI gym right after here;
# also add some sampling for calibrating the scaler
# ----------------------------------------------------------------------------

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
    a = np.random.choice(env_actions)    
    if SOFTWARE_ONLY:
        s = env.reset()
    else:
        hc_reset_sim()
        s = hc_get_sim_state()
    
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
        if SOFTWARE_ONLY:
            observation, r, done, _ = env.step(a)
        else:
            hc_influence_sim(a)
            observation = hc_get_sim_state()
            r = 1 # reward each "timestep" with a 1 to reward longevity
            # episode done, if x-position [0] or angle [1] invalid
            done = abs(observation[0]) > 0.9 or abs(observation[2]) > 1.0  

        for i in range(4):
            if episode_sim_max[i] < observation[i]:
                episode_sim_max[i] = observation[i]
            if episode_sim_min[i] > observation[i]:
                episode_sim_min[i] = observation[i]

        s2 = observation

        # Q-Learning
        old_qsa = get_Q_s_a(s, a)
        # if this is not the terminal position (which has no actions), then we can proceed as normal
        if not done:
            # "Q-greedily" grab the gain of the best step forward from here
            a2, max_q_s2a2 = max_Q_s(s2)

        # else change the rewards in the terminal position to have a clearer bias for "longevity"
        else:
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


if not SOFTWARE_ONLY:
    sys.exit(0)

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
