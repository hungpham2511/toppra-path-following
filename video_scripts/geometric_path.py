import numpy as np
import openravepy as orpy
import toppra as ta
import matplotlib.pyplot as plt
import following_lib as fo
from rave.Rave import inv_dyn

# Setup Logging
import logging
import coloredlogs
logger = logging.getLogger('trajectory')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('traj.log', mode='a')
logger.addHandler(fh)

# Setup logging for toppra
coloredlogs.install(level='DEBUG')
np.set_printoptions(5)

# Load OpenRAVE environment
env = orpy.Environment()
env.Load('/home/hung/git/robotic-CRI/rave/denso_vs060.dae')
robot = env.GetRobots()[0]

# Geometric path
waypoints = np.array([[0., 0., 0., 0., 0., 0.],
                      [-1., 0.5, 0.5, 0., 0.5, 0.9],
                      [1., 0.8, 1.3, 0.4, 1., 1.2],
                      [1., 0.9, 1.7, 0.2, 0.5, 0.8],
                      [0., 0., 0., 0., 0., 0.]])
path = ta.SplineInterpolator(np.linspace(0, 1, 5), waypoints)
