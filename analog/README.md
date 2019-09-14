## ANALOG-CARTPOLE - A hybrid analog/digital computing experiment

Use digital Reinforcement Learning to learn to balance an inverse pendulum
on a cart simulated on a Model-1 by  
Analog Paradigm (http://analogparadigm.com)

Done in July to September 2019. Analog part by [vaxman](http://www.vaxman.de).
Digital part by [sy2002](http://www.sy2002.de).

Watch a short introduction to this experiment on YouTube by clicking the
image below or by using this link: https://youtu.be/jDGLh8YWvNE

[![YouTube Link Image](doc/model-1-play.jpg)](https://youtu.be/jDGLh8YWvNE)

One of the classical "Hello World" examples of Reinforcement Learning is
the inverse pendulum. A pole is mounted on a cart. The cart can move in one
dimension, for example to the left or to the right. The center of mass is
located above the pivot point as shown in the following image:

![Schematic](doc/schematic.png)

A force is applied to move the cart to the right and to the left to keep
the pendulum upright. Important parameters of this system are: the position
of the cart, the velocity of the cart, the angle of the pole and
the angular velocity.

For training your reinforcement learning algorithms,
[OpenAI Gym offers a simulation](http://gym.openai.com/envs/CartPole-v1/)
of a simplified version of this model that delivers exactly these
four parameters for your control algorithm and that expects that you keep
applying impulses from the left or from the right to keep the pole upright.

In our experiment, we used a Model-1 from Analog Paradigm to create an
analog version of OpenAI Gym's simulation. Here is our setup:

![Setup of the Experiment](doc/setup.jpg)

You can see Model-1 on the right side. The algorithm (shown below) is wired on
the Model-1 using the black, blue, yellow and red cables. On top of Model-1,
there is an oscilloscope that shows the output of some training episodes of
the reinforcement learning algorithm. On the left is a Mac, on which a Python
script runs the reinforcement learning and which uses Model-1's Hybrid
Controller to send commands from the digital computer to the analog computer.

The oscilloscope on the right shows the control actions of the Python script:
A bar in the middle means "controls nothing" but "calculates and learns". A
bar on top means push cart to the right and a bar on the bottom means push
cart to the left.

### Analog simulation of the inverse pendulum

Programming an analog computer is quite different from programming a classic
stored-program digital computer as there is no algorithm controlling the 
operation of an analog computer. Instead, an analog computer program is a 
schematic describing how the various computing elements (such as summers,
integrators, multipliers, coeffizient potentiometers etc.) have to be 
interconnected in order to form an electronic model, an analogue, for the 
problem being investigated. 

At first sight this looks (and feels) quite weird but it turns out that it
is not only much more easy to program an analog computer than a digital 
computer as there is no need to think about numerical algorithms when it comes
to tasks like integration etc. Further, an analog computer is much more 
energy efficient than a digital computer since all of its computing elements
work in parallel without any need for slow memory lookups etc. Further, the
signals on which an analog computer simulation is based are continuous 
voltages or currents and not binary signals, so the super-linear increase of
power consumption as is typical for digital computers does not hold for analog
computers.

History background of analog computers can be found in this
[great book by Bernd Ulmann](https://www.amazon.de/Analog-Computing-Bernd-Ulmann/dp/3486728970/ref=sr_1_3).

A modern introduction to the programming of analog and hybrid computers is
described in
[this textbook](https://www.amazon.de/Analog-Computer-Programming-Gruyter-Textbook/dp/3110662078/ref=sr_1_6).

The schematic shown below is the analog computer setup to simulate an inverted
pendulum. The mathematic derivation can be found in
[this paper](http://analogparadigm.com/downloads/alpaca_20.pdf).

![Analog Program](doc/analog_program.png)

The picture below shows the subprogram controlling the moving cart by means of
a hybrid controller which couples the analog computer with a stored-program 
digital computer by means of an USB-interface. D0 and D1 depict two digital 
outputs of this controller and control two electronic switches by which the 
cart can be pushed to the left or to the right.

![Controler Schematics](doc/control.png)

### Digital reinforcement learning using the analog computer

#### System architecture of the experiment

[analog-cartpole.py](analog-cartpole.py) represents the digital part of this
experiment. It is a Python 3 script that uses the good old
[serial communication](https://en.wikipedia.org/wiki/Serial_communication)
to control the analog computer. For being able to do so, the analog computer
[Model 1](http://analogparadigm.com/products.html) offers an analog/digital
interface called [Hybrid Controller](http://analogparadigm.com/downloads/hc_handbook.pdf).

In our experiment, Model 1's Hybrid Controller is connected to a Mac using a
standard USB cable. There is no driver or other additional software necessary
to communicate between the Mac and the Hybrid Controller. The reason why this
works so smoothly is, that the Hybrid Controller uses an FTDI chip to implement
the serial communication over USB. Apple added FTDI support natively from OS X
Mavericks (10.9) on as described in
[Technical Note TN2315](https://developer.apple.com/library/archive/technotes/tn2315/_index.html).
For Windows and Linux, have a look at the 
[installation guides](https://www.ftdichip.com/Support/Documents/InstallGuides.htm).

The bottom line is, that from the perspective of our Python 3 script, the
analog computer can be controlled easily by sending serial commands to the
Hybrid Controller using [pySerial](https://pypi.org/project/pyserial/).

#### Digital-only mode

If you do not have an Analog Paradigm Model 1 analog computer handy, then
you can also run it on any digital computer that supports Python 3
and [OpenAI Gym](http://gym.openai.com/). For doing so, you can toggle the
operation mode of [analog-cartpole.py](analog-cartpole.py) by setting the
flag `SOFTWARE_ONLY` to `True`.

When doing so, we advise you to also set `PERFORM_CALIBRATION` to `True` as
this yields much better results on OpenAI Gym's simulation and you should set
`SINGLE_STEP_LEARNING` to `False` as single step learning is just a means of
compensating the slowness of the Python script in the "real world realm" of
analog computers.

#### Reinforcement Learning

We apply [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
to balance the CartPole. For doing so, the following four features are used to
feed a [Q-learning algorithm](https://en.wikipedia.org/wiki/Q-learning):
cart position, cart velocity, pole angle and the angular velocity of the pole's tip.

As Q-learning is a model-free algorithm, it actually "does not matter for the
algorithm", what the semantics of these four parameters/features mentioned above
are. It "does not know" the "meaning" of a feature such as "cart position" or
"angular velocity". For the Q-learning algorithm, the set of these features
is just what the current `State` within the `Environment` is comprised of.
And the `State` enables the `Agent` to decide, which `Action` to perform next.

![Diagram explaining the basics of RL](https://upload.wikimedia.org/wikipedia/commons/1/1b/Reinforcement_learning_diagram.svg)


* When we use a collection of (aka "network") of Radial Basis Functions (RBFs), then we can transform
  these 4 features into n distances from the centers of the RBFs, where n = RBF_EXEMPLARS x RBF_GAMMA_COUNT
* The "Gamma" parameter controls the "width" of the RBF's bell curve. The idea is, to use multiple RBFSamplers
  with multiple widths to construct a network with a good variety to sample from.
* The RBF transforms the initial 4 features into plenty of features and therefore offers a lot of "variables"
  or something like "resolution", where a Linear Regression algorithm can calcluate the weights for.
  In contrast, the original four features of the observation space would yield a too "low resolution".
* The Reinforcement Learning algorithm used to learn to balance the pole is Q-Learning.
* The "State" "s" of the Cart is obtained by using the above-mentioned RBF Network to transform the four
  original features into the n distances from a randomly chosen amount of RBF centers (aka "Exemplars").
  Note that it is absolutely OK, that the Exemplars are chosen randomly, because each Exemplar defines one
  possible combination of (Cart Position, Cart Velocity, Pole Angle and Pole Velocity At Tip); and therefore
  having lots of those random Exemplars just gives us the granularity ("resolution") we need for our Linear Regression.
* The possible "Actions" "a" of the system are "push from left" or "push from right", aka 0 and 1.
* For each possible action we are calculating one Value Function using Linear Regression. It can be illustrated by
  asking the question: "For the state we are currently in, defined by the distances to the Exemplars, what is the
  Value Function for pushing from left or puhsing from right. The larger one wins."

On RBFSampler:

Despite its name, the RBFSampler is actually not using any RBFs inside. I did some experiments about
this fact. Go to https://github.com/sy2002/ai-playground/blob/master/RBFSampler-Experiment.ipynb 
to learn more.

### Results

At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd
gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum
dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor
invidunt ut labore et dolore magna aliquyam erat, sed diam.

Episode|Mean Steps|Median Steps|Min. Steps|Max. Steps|Epsilon|
-------|----------|------------|----------|----------|-------|
0|82.00|82.00|82|82|0.4545
20|81.75|61.00|34|217|0.4167
40|170.10|138.50|48|420|0.3846
60|351.80|256.50|48|1398|0.3571
80|633.35|477.00|95|2421|0.3333
100|589.35|387.00|39|2945|0.3125
120|4674.80|2825.00|14|19709|0.2941
140|3608.60|1708.00|105|14783|0.2778

Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie
consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et
accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit
augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet.
