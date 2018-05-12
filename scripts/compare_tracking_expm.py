"""This script compare tracking performances of three controllers

1. OSG: Online Scalling with Guarantee (OSG)
1. TT: Computed Torque Trajectory Tracking (TT)
2. OS: Online Scaling (OS)

The controllers track the same geometric path. There are non-zero
initial joint positions.
"""
import numpy as np
import openravepy as orpy
import toppra as ta
import matplotlib.pyplot as plt
import following_lib as fo
import os
from rave.Rave import inv_dyn

# Setup Logging
import logging
import coloredlogs
logger = logging.getLogger('trajectory')
fh = logging.FileHandler('traj.log', mode='a')
logger.addHandler(fh)

# Setup logging for toppra
coloredlogs.install(level='INFO')
np.set_printoptions(5)

# Load OpenRAVE environment
env = orpy.Environment()
env.Load(os.path.join(os.environ["TOPPRA_FOLLOWING_HOME"],
                      'models/denso_vs060.dae'))
robot = env.GetRobots()[0]

# Geometric path
waypoints = np.array([[0., 0., 0., 0., 0., 0.],
                      [-1., 0.5, 0.5, 0., 0.5, 0.9],
                      [1., 0.8, 1.3, 0.4, 1., 1.2],
                      [1., 0.9, 1.7, 0.2, 0.5, 0.8],
                      [0., 0., 0., 0., 0., 0.]])
path = ta.SplineInterpolator(np.linspace(0, 1, 5), waypoints)

tau_max = np.r_[30., 70., 70., 30., 20., 20.] * 4
robot.SetDOFTorqueLimits(tau_max)
robot.SetDOFVelocityLimits(tau_max)
tau_min = - tau_max

# TOPP
N = 100
ss = np.linspace(0, 1, N+1)
cnst = ta.create_rave_torque_path_constraint(path, np.linspace(0, 1, N+1), robot)
cnst_intp = ta.interpolate_constraint(cnst)
pp = ta.qpOASESPPSolver([cnst_intp])
pp.set_goal_interval([0, 1e-8])
pp.set_start_interval([0, 1e-5])
pp.solve_controllable_sets()
us, xs = pp.solve_topp()
ts, qs, qds, qdds, ss_ref = fo.compute_trajectory_points(
    path, pp.ss, us, xs, dt=1e-3, smooth=True)

# Tracking parameter
lamb = 30

###############################################################################
#                   OSG_tb : Online Scaling w/Robust guarantee               #
###############################################################################

# Robust controllable sets the OSG controller
# TODO: Compute Ps propoerly.
vs = np.zeros((N+1, 12, 3))
Ps = np.zeros((N+1, 12, 3, 3))
vs[:, :, 0] = cnst.a
vs[:, :, 1] = cnst.b
vs[:, :, 2] = cnst.c

# Ps[:, 0] = np.eye(3) * 50
# Ps[:, 6] = np.eye(3) * 50
# Ps[:, 1] = np.eye(3) * 50
# Ps[:, 7] = np.eye(3) * 50
# Ps[:, 2] = np.eye(3) * 35
# Ps[:, 8] = np.eye(3) * 35
# Ps[:, 3] = np.eye(3) * 35
# Ps[:, 9] = np.eye(3) * 35
# Ps[:, 4] = np.eye(3) * 35
# Ps[:, 10] = np.eye(3) *35
# Ps[:, 5] = np.eye(3) * 25
# Ps[:, 11] = np.eye(3) *25


Ps[:, 0] = np.eye(3) * 50
Ps[:, 6] = np.eye(3) * 50
Ps[:, 1] = np.eye(3) * 50
Ps[:, 7] = np.eye(3) * 50
Ps[:, 2] = np.eye(3) * 50
Ps[:, 8] = np.eye(3) * 50
Ps[:, 3] = np.eye(3) * 50
Ps[:, 9] = np.eye(3) * 50
Ps[:, 4] = np.eye(3) * 50
Ps[:, 10] = np.eye(3)* 50
Ps[:, 5] = np.eye(3) * 50
Ps[:, 11] = np.eye(3)* 50


robust_cnst = fo.RobustPathConstraint(vs, Ps, ss)
ws = 0.01 * np.ones(N+1)  # Norm of the tracking error
Ks = np.ones((N+1, 2))  # Robust controllable sets
Ks[N] = [0, 1e-8]
for i in range(N-1, -1, -1):
    logger.info("OSRG: compute robust controllable set i={:d}".format(i))
    Ks[i] = fo.robust_one_step(robust_cnst, i, Ks[i + 1], ws[i], ws[i + 1], method='ECOS')

# def f(method):
#     robust_cnst = fo.RobustPathConstraint(vs, Ps, ss)
#     ws = 0.1 * np.ones(N+1)  # Norm of the tracking error
#     Ks = np.ones((N+1, 2))  # Robust controllable sets
#     Ks[N] = [0, 1e-8]
#     for i in range(N-1, -1, -1):
#         logger.info("OSRG: compute robust controllable set i={:d}".format(i))
#         Ks[i] = fo.robust_one_step(robust_cnst, i, Ks[i + 1], ws[i], ws[i + 1], method=method)

# Define experiments
reload(fo)
OSG_exp = fo.ExperimentOSG(robot, path, Ks, ss,
                           tau_min, tau_max, lamb)
OS_exp = fo.ExperimentOS(robot, path, us, xs, ss,
                         tau_min, tau_max, lamb)
TT_exp = fo.ExperimentTT(robot, path, qs, qds, qdds,
                         tau_min, tau_max, lamb)
TT_notrb_exp = fo.ExperimentTT(robot, path, qs, qds, qdds,
                               10 * tau_min, 10 * tau_max, lamb)
experiments = [OSG_exp, OS_exp, TT_exp]

# Setup
initial_err = np.array([-0.02483, -0.05122,  0.02318,
                        0.09367 , -0.09675,  0.13449,
                        0.      ,  0.     ,  0.     ,
                        0.      ,  0.     ,  0.     ])
initial_err = 0.1 * initial_err / np.linalg.norm(initial_err)

noise_det_sampled = np.random.randn(10000, 6)
def noise_function(t, noise_det_sampled=noise_det_sampled):
    return np.diag([0.3, 0.8, 0.2, 0.1, 0.2, 0.1]).dot(
        noise_det_sampled[int(t / 1e-3)])

for exp in experiments:
    exp.set_unit_noise_function(noise_function)
    # exp.set_noise_level(10.)
    exp.set_noise_level(0.)
    exp.set_initial_error(initial_err)
    exp.lamb = lamb
    exp.reset()

TT_res = TT_exp.run()
OSG_res = OSG_exp.run()
OS_res = OS_exp.run()

fig, axs = plt.subplots(2, 1)
axs[0].plot(TT_res['traj_e'][:, :6], c='blue')
axs[0].plot(OSG_res['traj_e'][:, :6], c='red')
axs[0].plot(OS_res['traj_e'][:, :6], c='purple')
axs[1].plot(np.linalg.norm(TT_res['traj_e'][:, :6], axis=1), c='blue')
axs[1].plot(np.linalg.norm(OSG_res['traj_e'][:, :6], axis=1), c='red')
axs[1].plot(np.linalg.norm(OS_res['traj_e'][:, :6], axis=1), c='purple')
plt.show()

###############################################################################
#                    Fig 1: Compare norm of tracking error                    #
###############################################################################
# PLotting
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['text.usetex'] = True
f, ax = plt.subplots(figsize=[3.2, 1.8])
ax.plot(ss_ref,
        np.linalg.norm(TT_res['traj_e'][:, :6], axis=1), '-',
        label='TT', c='C1')
ax.plot(OS_res['traj_s'],
        np.linalg.norm(OS_res['traj_e'][:, :6], axis=1), '-',
        label='OS', c='C4')
ax.plot(OSG_res['traj_s'],
        np.linalg.norm(OSG_res['traj_e'][:, :6], axis=1),
        label='TOPT', c='C2')
ax.plot((0, 1), (0, 0), '--', lw=0.5, c='gray')
# ax.set_ylim(-0.0015, 0.02)
ax.set_xlim(-0.025, 1.025)
# ax.set_yticks(np.arange(0, 0.021, 0.002))
ax.legend()
plt.savefig('compare_tracking_performance.pdf')
plt.show()

###############################################################################
#                  Fig 2: The online generate Parametrization                 #
###############################################################################
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['text.usetex'] = True
f, ax = plt.subplots(figsize=[3.2, 1.8])
ax.plot(ss, Ks[:, 1], '--', lw=1, c='C6', alpha=0.9, label='Robust controllable sets')
ax.plot(ss, Ks[:, 0], '--', lw=1, c='C6', alpha=0.9)
# ax.plot(ss, pp.K[:, 1], '--', lw=1, c='C3', alpha=0.5)
# ax.plot(ss, pp.K[:, 0], '--', lw=1, c='C3', alpha=0.5)
ax.plot(ss, xs, c='C3')
ax.plot(OS_res['traj_s'], OS_res['traj_sd'] ** 2, c='C4', label='Parameterization OS', alpha=1)
ax.plot(OSG_res['traj_s'], OSG_res['traj_sd'] ** 2, c='C2', label='Parameterization TOPT', zorder=9)
ax.set_xlim(-0.025, 0.6)
ax.set_ylim(-0.25, 3.2)
ax.legend()
plt.savefig('actual_sd_traj.pdf')
plt.show()

import IPython
if IPython.get_ipython() is None:
    IPython.embed()
