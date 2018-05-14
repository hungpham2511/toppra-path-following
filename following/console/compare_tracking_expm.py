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
import following as fo
import os, time, logging


def main(env=None, verbose=True):
    # Setup Logging
    if verbose:
        logging.basicConfig(level="DEBUG")
    else:
        logging.basicConfig(level="WARN")
    # Setup logging for toppra
    np.set_printoptions(5)

    # Load OpenRAVE environment
    print("Loading openrave environment...")
    if env is None:
        env = orpy.Environment()
    else:
        env.Reset()
    fo.try_load_denso(env)
    # env.Load(os.path.join(os.environ["TOPPRA_FOLLOWING_HOME"],
                          # 'models/denso_vs060.dae'))
    env.SetViewer('qtosg')
    robot = env.GetRobots()[0]
    newrobot = orpy.RaveCreateRobot(env, robot.GetXMLId())
    newrobot.Clone(robot, 0)
    I = np.eye(4)
    I[:3, 3] = [-0.008, +0.008, - 0.008]
    newrobot.SetTransform(I)
    robot.SetName("actual_robot")
    env.AddRobot(newrobot)
    env.GetViewer().SetCamera(np.array([[ 0.76186,  0.36365, -0.53604,  1.01322],
                                        [ 0.63826, -0.28039,  0.71694, -1.03112],
                                        [ 0.11041, -0.88834, -0.44572,  1.2291 ],
                                        [ 0.     ,  0.     ,  0.     ,  1.     ]]))

    for l in env.GetRobots()[1].GetLinks():
        for geom in l.GetGeometries():
            geom.SetTransparency(0.1)
            geom.SetDiffuseColor([0.9, 0.3, 0.3])

    # Geometric path
    print("Compute the controllable sets and the time-optimal parametrization.")
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

    ss = np.linspace(0, 1, 100 + 1)
    cnst = ta.create_rave_torque_path_constraint(robot, 1)
    robust_cnst = ta.constraint.RobustCanonicalLinearConstraint(cnst, [10, 10, 10], 1)
    pp = ta.algorithm.TOPPRA([cnst], path, ss)
    robust_pp = ta.algorithm.TOPPRA([robust_cnst], path, ss)
    Ks = robust_pp.compute_controllable_sets(0, 0)
    traj, _, (sdd_vec, sd_vec, _) = pp.compute_trajectory(0, 0, return_profile=True)
    us = sdd_vec
    xs = sd_vec ** 2
    ts = np.arange(0, traj.get_duration(), 1e-3)
    qs = traj.eval(ts)
    qds = traj.evald(ts)
    qdds = traj.evaldd(ts)

    # Compare ss_ref: path positions at referenc time instance.
    ss_traj = ta.SplineInterpolator(traj.ss_waypoints, ss)
    ss_ref = ss_traj.eval(ts)

    # Tracking parameter
    lamb = 30

    # Define experiments
    OSG_exp = fo.ExperimentOSG(robot, path, Ks, ss, tau_min, tau_max, lamb, cloned_robot=newrobot)
    OS_exp = fo.ExperimentOS(robot, path, us, xs, ss, tau_min, tau_max, lamb, cloned_robot=newrobot)
    TT_exp = fo.ExperimentTT(robot, path, qs, qds, qdds, tau_min, tau_max, lamb, cloned_robot=newrobot)
    TT_notrb_exp = fo.ExperimentTT(robot, path, qs, qds, qdds, 10 * tau_min, 10 * tau_max, lamb)
    experiments = [OSG_exp, OS_exp, TT_exp]

    # Setup
    initial_err = np.array([-0.02483, -0.05122,  0.02318,
                            0.09367 , -0.09675,  0.13449,
                            0.      ,  0.     ,  0.     ,
                            0.      ,  0.     ,  0.     ])
    initial_err = 0.3 * initial_err / np.linalg.norm(initial_err)

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

    print("Run Trajectory-Tracking controller in 3 secs")
    TT_exp.set_reference_robot_joint(waypoints[0])
    TT_exp.set_robot_joint(waypoints[0] + initial_err[:6])
    time.sleep(3)
    TT_res = TT_exp.run()
    print("Run Online-Scaling controller in 3 secs")
    TT_exp.set_reference_robot_joint(waypoints[0])
    TT_exp.set_robot_joint(waypoints[0] + initial_err[:6])
    time.sleep(3)
    OS_res = OS_exp.run()
    print("Running Online-Scaling with Guarantee controller in 3 secs")
    TT_exp.set_reference_robot_joint(waypoints[0])
    TT_exp.set_robot_joint(waypoints[0] + initial_err[:6])
    time.sleep(3)
    OSG_res = OSG_exp.run()

    ###############################################################################
    #                    Fig 0:                                                   #
    ###############################################################################
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(TT_res['traj_e'][:, :6], c='blue')
    axs[0].plot(OSG_res['traj_e'][:, :6], c='red')
    axs[0].plot(OS_res['traj_e'][:, :6], c='purple')
    axs[1].plot(np.linalg.norm(TT_res['traj_e'][:, :6], axis=1), c='blue', label="trajectory tracking")
    axs[1].plot(np.linalg.norm(OSG_res['traj_e'][:, :6], axis=1), c='red', label="OSG")
    axs[1].plot(np.linalg.norm(OS_res['traj_e'][:, :6], axis=1), c='purple', label="OS")
    plt.legend()
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
    ax.plot(ss_ref, np.linalg.norm(TT_res['traj_e'][:, :6], axis=1), '-',
            label='TT', c='C1')
    ax.plot(OS_res['traj_s'], np.linalg.norm(OS_res['traj_e'][:, :6], axis=1), '-',
            label='OS', c='C4')
    ax.plot(OSG_res['traj_s'], np.linalg.norm(OSG_res['traj_e'][:, :6], axis=1),
            label='TOPT', c='C2')
    ax.plot((0, 1), (0, 0), '--', lw=0.5, c='gray')
    ax.set_xlim(-0.025, 1.025)
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


if __name__ == '__main__':
    main()
