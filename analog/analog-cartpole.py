# ANALOG-CARTPOLE - A hybrid analog/digital computing experiment
#
# Use digital Reinforcement Learning (RL) to learn to balance an inverse pendulum
# on a cart simulated by a Model-1 by Analog Paradigm (http://analogparadigm.com)
#
# Analog part done by vaxman on 2019-07-27
# Digital part done by sy2002 on 2019-07-27

# if you don't have a Model-1 at hand, set SOFTWARE_ONLY to True
# to use a software based physics simulation by OpenAI Gym
SOFTWARE_ONLY     = False

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

CLBR_RND_EPISODES   = 500           # during calibration: number of random episodes
CLBR_LEARN_EPISODES = 120           # during calibration: number of learning episodes
LEARN_EPISODES      = 500           # real learning: # of episodes to learn
TEST_EPISODES       = 10            # software only: # of episodes  we use the visual rendering to test what we learned
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
# Environment abstraction
# ----------------------------------------------------------------------------

env = None

# list of possible actions that the RL agent can perform in the environment
# for the algorithm, it doesn't matter if 0 means right and 1 left or vice versa or if
# there are more than two possible actions
env_actions = [0, 1] # needs to be in ascending order with no gaps, e.g. [0, 1]

def env_random_action():
    return np.random.choice(env_actions)

def env_prepare():
    global env

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

# reset environment: each episode starts again at the initial condition
# independent of using software or the analog computer, we have these core variables
# that are describing the state of the RL's environment and possible actions:
#    s = (cart position, cart velocity, pole angle, pole angle velocity)
#    a = <possible actions> = (0 or 1), apply an impulse to the car to the left/right
def env_reset():
    if SOFTWARE_ONLY:
        s = env.reset()
    else:
        hc_reset_sim()
        s = hc_get_sim_state()
    return s, env_random_action()

#perform action and return observation, reward and "done" flag
def env_step(action_to_be_done):
    if SOFTWARE_ONLY:
        observation, r, done, _ = env.step(action_to_be_done)
    else:
        hc_influence_sim(action_to_be_done)
        observation = hc_get_sim_state()
        r = 1 # reward each "timestep" with a 1 to reward longevity
        # episode done, if x-position [0] or angle [1] invalid
        done = abs(observation[0]) > 0.9 or abs(observation[2]) > 1.0  
    return observation, r, done

#visual display of the environment's output on the digital computer's screen
def env_render():
    if SOFTWARE_ONLY:
        env.render()
    else:
        print("ERROR #3: env_render(): NOT IMPLEMENTED, YET.")
        sys.exit(3)
       
# ----------------------------------------------------------------------------
# Reinforcement Learning Core
# ----------------------------------------------------------------------------

scaler      = None
gammas      = None
models      = None
rbfs        = None
rbf_net     = None

# transform the 4 features (Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip)
# into RBF_EXEMPLARS x RBF_GAMMA_COUNT distances from the RBF centers ("Exemplars")
def rl_transform_s(s):
    if scaler == None:  # during calibration, we do not have a scaler, yet
        return rbfs.transform(np.array(s).reshape(1, -1))
    else:
        return rbfs.transform(scaler.transform(np.array(s).reshape(1, -1)))

# SGDRegressor expects a vector, so we need to transform our action, which is 0 or 1 into a vector
def rl_transform_val(val):
    return np.array([val]).ravel()

def rl_init():
    global scaler, gammas, models, rbfs, rbf_net

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

    # SGDRegressor needs "partial_fit" to be called before the first "predict" is called
    # so let's do that for all actions (aka entries in the rbf_net list) with random (dummy) values
    # for the state and the Value Function
    for net in rbf_net:
        random_val_func = np.random.rand() * np.random.randint(1, 100) # random float 0 < x < 100
        net.partial_fit(rl_transform_s([0.5, 0.1, -0.5, 0.1]), rl_transform_val(random_val_func))

# get the result of the Value Function for action a taken in state s
def rl_get_Q_s_a(s, a):
    return rbf_net[a].predict(rl_transform_s(s))

# learn Value Function for action a in state s
def rl_set_Q_s_a(s, a, val):
    rbf_net[a].partial_fit(rl_transform_s(s), rl_transform_val(val))

# find the best action to take in state s
def rl_max_Q_s(s):
    max_val = float("-inf")
    max_idx = -1
    for a in env_actions:
        val = rl_get_Q_s_a(s, a)
        if val > max_val:
            max_val = val
            max_idx = a
    return max_idx, max_val

def rl_learn(learning_duration, record_observations=False):
    if record_observations:
        recorded_obs = []
    else:
        recorded_obs = None

    print("Episode\t\tMean Steps\tMedian Steps\tMin. Steps\tMax. Steps\tEpsilon")

    t = 1.0                             # used to decay epsilon
    probe_episode_step_count = []       # used for probe stats: mean, median, min, max

    for episode in range(learning_duration + 1):
        # let epsilon decay over time
        if episode % EPSILON_DECAY_m == 0:
            t += EPSILON_DECAY_t

        # each episode starts again at the begining
        s, a = env_reset()        
        episode_step_count = 0
        done = False

        while not done:
            episode_step_count += 1

            # epsilon-greedy: do a random move with the decayed EPSILON probability
            p = np.random.random()
            eps = EPSILON / t
            if p > (1 - eps):
                a = np.random.choice(env_actions)

            # exploit or explore and collect reward
            observation, r, done = env_step(a)            
            s2 = observation
            if record_observations:
                recorded_obs.append(observation)

            # Q-Learning
            old_qsa = rl_get_Q_s_a(s, a)
            # if this is not the terminal position (which has no actions), then we can proceed as normal
            if not done:
                # "Q-greedily" grab the gain of the best step forward from here
                a2, max_q_s2a2 = rl_max_Q_s(s2)

            # else change the rewards in the terminal position to have a clearer bias for "longevity"
            else:
                a2 = a
                max_q_s2a2 = 0 # G (future rewards) of terminal position is 0 although the reward r is high

            # learn the new value for the Value Function
            new_value = old_qsa + (r + GAMMA * max_q_s2a2 - old_qsa)
            rl_set_Q_s_a(s, a, new_value)

            # next state = current state, next action is the "Q-greedy" best action
            s = s2
            a = a2        

        # every PROBEth episode: print status info
        probe_episode_step_count.append(episode_step_count)
        if episode % PROBE == 0:
            print("%d\t\t%0.2f\t\t%0.2f\t\t%d\t\t%d\t\t%0.4f" % (   episode, 
                                                                    np.mean(probe_episode_step_count),
                                                                    np.median(probe_episode_step_count),
                                                                    np.min(probe_episode_step_count),
                                                                    np.max(probe_episode_step_count),
                                                                    eps))
            probe_episode_step_count = []

    return recorded_obs

# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

print("Calibrating:")
print("============\n")

env_prepare()   # setup the environment (either analog or digital)
rl_init()       # init and clear the RL "brain"
clbr_res = []   # contains all observation samples taken during calibration

print("Performing %d random episodes..." % CLBR_RND_EPISODES, end="")
episode_counts = []
for i in range(CLBR_RND_EPISODES):
    env_reset()
    episode_count = 0
    done = False
    while not done:
        observation, r, done = env_step(env_random_action())
        clbr_res.append(observation)
        episode_count += 1
    episode_counts.append(episode_count)
print("\b\b\b: Done. %0.2f average steps per episode" % np.mean(episode_counts))

print("Performing %d uncalibrated learning episodes:\n" % CLBR_LEARN_EPISODES)
clbr_res += rl_learn(CLBR_LEARN_EPISODES, record_observations=True)

# create scaler and fit it to the sampled observation space
scaler = StandardScaler() 
scaler.fit(clbr_res)

print("\nCalibration done. Samples taken: %d" % scaler.n_samples_seen_)
print("    Mean:     %+2.8f %+2.8f %+2.8f %+2.8f" % tuple(scaler.mean_))
print("    Variance: %+2.8f %+2.8f %+2.8f %+2.8f" % tuple(scaler.var_))
print("    Scale:    %+2.8f %+2.8f %+2.8f %+2.8f" % tuple(scaler.scale_))

# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------

print("\nLearn:")
print("======\n")
rl_init()
rl_learn(LEARN_EPISODES)

if not SOFTWARE_ONLY:
    print("")
    sys.exit(0)

# ----------------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------------

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
print("Episode\tDstb\tResult")

all_steps = 0
for episode in range(TEST_EPISODES):
    observation, _ = env_reset()
    env_render()
    done = False
    won = False
    episode_step_count = 0
    dist_ongoing = 0

    # we are ignoring Gym's "done" function, so that we can run the system longer
    while observation[0] > -2.4 and observation[0] < 2.4 and episode_step_count < 5000:
        episode_step_count += 1
        all_steps += 1

        a, _ = rl_max_Q_s(observation)
        if DISTURB_PROB > 0.0:            
            if dist_ongoing == 0 and np.random.rand() > (1.0 - DISTURB_PROB):
                dist_ongoing = DISTURB_DURATION
            if dist_ongoing > 0:
                print("\tstep #%d:\tdisturbance ongoing: %d" % (episode_step_count, dist_ongoing))
                dist_ongoing -= 1
                a = env_random_action()

        observation, _, done = env_step(a)
        env_render()
    
    print("%d\t\t%d\t\t" % (episode, episode_step_count))

print("Avg. Steps: %0.2f" % (all_steps / TEST_EPISODES))
env.close()
