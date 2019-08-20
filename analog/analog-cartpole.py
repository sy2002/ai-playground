#!/usr/bin/env python3
#
# ANALOG-CARTPOLE - A hybrid analog/digital computing experiment
#
# Use digital Reinforcement Learning (RL) to learn to balance an inverse pendulum
# on a cart simulated by a Model-1 by Analog Paradigm (http://analogparadigm.com)
#
# Analog part done by vaxman on 2019-07-27, 2019-07-28, 2019-08-03
# Digital part done by sy2002 on 2019-07-27, 2019-07-29, 2019-08-03, 2019-08-20

# ----------------------------------------------------------------------------
# Global Flags
# ----------------------------------------------------------------------------

# If you don't have a Model-1 at hand, set SOFTWARE_ONLY to True
# to use a software based physics simulation by OpenAI Gym.
SOFTWARE_ONLY           = False

# Use this switch to toggle calibration.
# It seems, that without calibration the system runs better on real hardware.
PERFORM_CALIBRATION     = False

# If the digital computer is too slow to be able to follow the speed of
# the Model-1 analog computer in real-time, then set SINGLE_STEP to True.
# It will make sure, that during the calibration and learning phase, the
# analog computer will be halted while the digital computer is doing its math.
# Later, during execution phase, the system will always work in realtime.
SINGLE_STEP_LEARNING    = False

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

print("\nAnalog Cartpole - A hybrid analog/digital computing experiment")
print("==============================================================\n")

import numpy as np
import serial
import sys

from joblib import dump, load
from readchar import readchar
from time import sleep

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

if (SOFTWARE_ONLY):
    import gym
    import gym.spaces
    print("WARNING: Software-only mode. No analog computer is being used.")

# ----------------------------------------------------------------------------
# Analog Setup
# ----------------------------------------------------------------------------

# Analog simulation parameters
HC_IMPULSE_DURATION = 10            # duration [ms] of the impulse, that influences the car
HC_X_MAX            = 0.9           # maximum |x| of car, otherwise episode done
HC_ANGLE_MAX        = 0.5           # maximum |angle| of pole, otherwise episode done

# Hybrid Controller serial setup
HC_PORT             = "/dev/cu.usbserial-DN050L1O"
HC_BAUD             = 115200        
HC_BYTE             = 8
HC_PARITY           = serial.PARITY_NONE
HC_STOP             = serial.STOPBITS_ONE
HC_RTSCTS           = False
HC_TIMEOUT          = 0.02          # increase to e.g. 0.05, if you get error #2
HC_BULK             = False         # use bulk communication in hc_get_sim_state()

# Addresses of the environment/simulation data
HC_SIM_X_POS        = "0223"        # address of cart's x-position
HC_SIM_X_VEL        = "0222"        # address of cart's x-velocity
HC_SIM_ANGLE        = "0161"        # address of pole's/pendulum's angle
HC_SIM_ANGLE_VEL    = "0160"        # address of pole's/pendulum's angular velocity

HC_SIM_DIRECTION_1  = "D0"          # digital out #0=hi: direction = 1, e.g. right
HC_SIM_DIRECTION_0  = "d0"          # digital out #0=lo: direction = 0, e.g. left
HC_SIM_IMPULSE_1    = "D1"          # digital out #1=hi: apply force
HC_SIM_IMPULSE_0    = "d1"          # digital out #1=lo: stop applying force

# Model-1 Hybrid Controller: commands and responses
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
CLBR_LEARN_EPISODES = 200           # during calibration: number of learning episodes
LEARN_EPISODES      = 2000          # real learning: # of episodes to learn
TEST_EPISODES       = 10            # software only: # of episodes  we use the visual rendering to test what we learned
TEST_MAX_STEPS      = 5000          # maximum amount of steps during test/execution phase

GAMMA               = 0.999         # discount factor for Q-Learning
ALPHA               = 0.3           # initial learning rate
ALPHA_DECAY         = 0.2           # learning rate decay
EPSILON             = 0.5           # randomness for epsilon-greedy algorithm (explore vs exploit)
EPSILON_DECAY_t     = 0.1
EPSILON_DECAY_m     = 18            # every episode % EPSILON_DECAY_m == 0, we increase EPSILON_DECAY_t

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
                                dsrdtr=False,
                                timeout=HC_TIMEOUT)
        sleep(1.5) #https://pyserial.readthedocs.io/en/latest/appendix.html: FAQ
        dbg_last_sent = ""
    except:
        print("ERROR #1: SERIAL PORT CANNOT BE OPENED.")
        sys.exit(1)

def hc_send(cmd):
    global dbg_last_sent
    dbg_last_sent = cmd
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
        print("Last command sent:", dbg_last_sent)
        print("Length of received string:", len(str))
        print("Hex output of received string:", ":".join("{:02x}".format(ord(c)) for c in str))
        sys.exit(2)

# ask for a value
def hc_ask_for_value(addr):
    hc_send(HC_CMD_GETVAL + addr)

# query the current state of the simulation, which consists of
# the x-pos and the the x-velocity of the cart, the angle and
# angle velocity of the pole/pendulum
def hc_get_sim_state():
    # bulk transfer: ask for all values that constitue the state in
    # a bulk and read them in a bulk
    if HC_BULK:
        hc_ask_for_value(HC_SIM_X_POS)
        hc_ask_for_value(HC_SIM_X_VEL)    
        hc_ask_for_value(HC_SIM_ANGLE)
        hc_ask_for_value(HC_SIM_ANGLE_VEL)
        return (    hc_res2float(hc_receive()),
                    hc_res2float(hc_receive()),
                    hc_res2float(hc_receive()),
                    hc_res2float(hc_receive()))
    else:
        hc_ask_for_value(HC_SIM_X_POS)
        res_x_pos = hc_res2float(hc_receive())
        hc_ask_for_value(HC_SIM_X_VEL)
        res_x_vel = hc_res2float(hc_receive())    
        hc_ask_for_value(HC_SIM_ANGLE)
        res_angle = hc_res2float(hc_receive())
        hc_ask_for_value(HC_SIM_ANGLE_VEL)
        res_anvel = hc_res2float(hc_receive())
        return (res_x_pos, res_x_vel, res_angle, res_anvel)

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
def hc_influence_sim(a, is_learning):
    if is_learning and SINGLE_STEP_LEARNING:
        hc_send(HC_CMD_OP) #TODO: do this right; instead of flushing in line #200

    if (a == 1):
        hc_send(HC_SIM_DIRECTION_1)
    else:
        hc_send(HC_SIM_DIRECTION_0)
   
    hc_send(HC_SIM_IMPULSE_1)
    sleep(HC_IMPULSE_DURATION / 1000.0)
    hc_send(HC_SIM_IMPULSE_0)

    if is_learning and SINGLE_STEP_LEARNING:
        #TODO: do this the right way, see also line #185
        hc_send(HC_CMD_HALT)
        sleep(0.05)
        hc_ser.flushInput()

# ----------------------------------------------------------------------------
# Environment abstraction
# ----------------------------------------------------------------------------

env = None

# List of possible actions that the RL agent can perform in the environment.
# For the algorithm, it doesn't matter if 0 means right and 1 left or vice versa
# or if there are more than two possible actions
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

# evaluate, if an episode is over
def env_is_done(observation):
    if SOFTWARE_ONLY:
        return abs(observation[0]) > 2.4
    else:
        # episode done, if x-position [0] or angle [2] invalid
        return abs(observation[0]) > HC_X_MAX or abs(observation[2]) > HC_ANGLE_MAX

# perform action and return observation, reward and "done" flag
def env_step(action_to_be_done, is_learning=True):
    if SOFTWARE_ONLY:
        observation, r, done, _ = env.step(action_to_be_done)
    else:
        hc_influence_sim(action_to_be_done, is_learning)
        observation = hc_get_sim_state()
        r = 1 # reward each "timestep" with a 1 to reward longevity
        done = env_is_done(observation)  
    return observation, r, done

# visual display of the environment's output on the digital computer's screen
def env_render():
    if SOFTWARE_ONLY:
        env.render()

def env_close():
    if SOFTWARE_ONLY:
        env.close()
       
# ----------------------------------------------------------------------------
# Reinforcement Learning Core
# ----------------------------------------------------------------------------

scaler      = None
gammas      = None
models      = None
rbfs        = None
rbf_net     = None

# save the model
def rl_save(filename):
    try:
        dump(scaler,    filename + ".scaler")
        dump(rbfs,      filename + ".rbfs")
        dump(rbf_net,   filename + ".rbfnet")
    except:
        print("ERROR #4: Error saving RL model.")
        sys.exit(4)

# load the model
def rl_load(filename):
    global scaler, rbfs, rbf_net
    try:
        scaler  = load(filename + ".scaler")
        rbfs    = load(filename + ".rbfs")
        rbf_net = load(filename + ".rbfnet")
    except:
        print("ERROR #5: Error loading RL model.")
        sys.exit(5)

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

    t = 1.0                             # used to decay epsilon
    probe_episode_step_count = []       # used for probe stats: mean, median, min, max
    tprint("Episode", "Mean Steps", "Median Steps", "Min. Steps", "Max. Steps", "Epsilon")

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
            tprint("%d" % episode,
                   "%0.2f" % np.mean(probe_episode_step_count),
                   "%0.2f" % np.median(probe_episode_step_count),
                   "%d" % np.min(probe_episode_step_count),
                   "%d" % np.max(probe_episode_step_count),
                   "%0.4f" % eps)
            probe_episode_step_count = []

    return recorded_obs

# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

def main_calibrate():
    global scaler

    print("\nCalibrate:")
    print("==========\n")

    rl_init()       # init and clear the RL "brain"
    clbr_res = []   # contains all observation samples taken during calibration

    print("Performing %d random episodes..." % CLBR_RND_EPISODES, end="")
    sys.stdout.flush()  # necessary to make sure we see the printed strint immediatelly
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

def main_learn(save_file):
    print("\nLearn:")
    print("======\n")

    rl_init()
    rl_learn(LEARN_EPISODES)

    if save_file:
        rl_save(save_file)

# ----------------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------------

def main_test(test_duration):
    print("\nTest:")
    print("=====\n")
    tprint("Episode", "Result")

    all_steps = 0
    for episode in range(test_duration):
        observation, _ = env_reset()
        env_render()
        episode_step_count = 0
        dist_ongoing = 0

        while not env_is_done(observation) and episode_step_count < TEST_MAX_STEPS:
            episode_step_count += 1
            all_steps += 1

            a, _ = rl_max_Q_s(observation)          # evaluate Value Function and find next action
            observation, _, _ = env_step(a, False)  # perform next action and observe result
            env_render()                            # (software mode) display visualization

        tprint("%d" % episode, "%d" % episode_step_count)
    print("\nAvg. Steps: %0.2f\n" % (all_steps / test_duration))

# ----------------------------------------------------------------------------
# Command line handling and tools
# ----------------------------------------------------------------------------

MAIN_MODE_STD   = 0     # standard mode: calibrate, learn, test
MAIN_MODE_SAVE  = 1     # like standard mode, but save calibration and learned state
MAIN_MODE_LOAD  = 2     # execute only: load calibration and learned state and run/test it

def tprint(*argv):
    for arg in argv:
        print(arg.ljust(16), end="")
    print("")

def parse_args():
    argc = len(sys.argv)

    # Standard Mode
    if argc == 1:
        return MAIN_MODE_STD, None, None
    
    elif argc in [3, 4]:
        cmd = sys.argv[1].upper()

        # Save / Persistence mode
        if cmd == "-S":
            filename = sys.argv[2]
            if filename:
                print("HINT: Persistence mode active. Save to:", filename + ".scaler, " + 
                                                                filename + ".rbfs and " +
                                                                filename + ".rbfnet")
            return MAIN_MODE_SAVE, filename, None

        # Load / Execution mode
        elif cmd == "-L":
            filename = sys.argv[2]
            duration = None
            if argc == 4:
                try:    
                    duration = int(sys.argv[3])
                except:
                    pass
            print("HINT: Execution mode active. Loading:", filename + ".scaler, " + 
                                                           filename + ".rbfs and " +
                                                           filename + ".rbfnet")
            if duration is None:
                duration = TEST_EPISODES
            return MAIN_MODE_LOAD, filename, duration

    # Invalid arguments
    print("ERROR #3: Invalid command line options.")
    sys.exit(3)

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

mode, filename, duration = parse_args()

env_prepare()

if mode != MAIN_MODE_LOAD:
    if PERFORM_CALIBRATION:
        main_calibrate()
    main_learn(filename)
    main_test(TEST_EPISODES)
else:
    rl_load(filename)
    main_test(duration)

env_close()
