# rl_gazebo

Library to perform Deep Reinforcement Learning on Gazebo physics simulator in a ROS environment.

The library is tested with Ubuntu 18.04, ROS Melodic and Python 3.6

## Install

Create a virtual envinronment or conda environemnt

```
virtualenv --pythoon=python3.6 /path/to/virtual/environment
source /path/to/virtual/environment/bin/activate
cd /path/to/virtual/environment/lib/python3.6/site-packages/rl_gazebo
pip install -e .
```
Install manually all the dependecies.

Inside the package there is the ROS metapackage ```rlgazebo``` containing all necessary to necessary files. Create your own ros workspace where you prefere and copy inside of it the all the metapackage and build normally.


## Dependencies
```
numpy
tensorflow
stable-baselines
```

## Usage

Open two different terminals (it is necessary since ROS Melodic and the python packages uses different python versions). 
In the first one source only the ROS worksspace and launch the desired environemnt
```
source /path/to/workspace/devel/setup.bash
roslaunch rl_gazebo ur10_humancyl.launch 
```
In the second terminal source both the ROS workspace and the virtual environment(the order is in which they are sourced is not relevant), then launch the desired script.
```
source /path/to/workspace/devel/setup.bash
source /path/to/virtual/environment/bin/activate
python /path/to/virtual/environment/lib/python3.6/site-packages/rl_gazebo/scripts/test_random_target.py
```
