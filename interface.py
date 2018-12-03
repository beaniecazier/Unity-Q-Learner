env_name = None

train_mode = True

import matplotlib.pyplot as plt
import numpy as np
import sys

from mlagents.envs import UnityEnvironment

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env = UnityEnvironment(file_name=env_name)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

def output(next_action):
    env.step(next_action)

def current_state():
    brainInfo = info['CrawlerBrain']
    return brainInfo.vector_observations

def goal_distance():
    brainInfo = info['CrawlerBrain']
    distanceVector = np.array(brainInfo.vector_observations[0], brainInfo.vector_observations[1], brainInfo.vector_observations[2])
    return np.linalg.norm(distanceVector)
