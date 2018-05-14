import numpy as np
import openravepy as orpy
import toppra as ta
import matplotlib.pyplot as plt
import os


def main():
    # Setup the robot, the geometric path and the torque bounds
    env = orpy.Environment()
    env.Load(os.path.join(os.environ["TOPPRA_FOLLOWING_HOME"], 'models/denso_vs060.dae'))
    robot = env.GetRobots()[0]
    np.random.seed(11)
    waypoints = np.random.randn(5, 6) * 0.4
    path = ta.SplineInterpolator(np.linspace(0, 1, 5), waypoints)
    tau_max = np.r_[30., 50., 20., 20., 10., 10.]
    robot.SetDOFTorqueLimits(tau_max)

    # Nominal case
    cnst = ta.create_rave_torque_path_constraint(robot, 1)
    pp = ta.algorithm.TOPPRA([cnst], path)
    Ks = pp.compute_controllable_sets(0, 0)

    robust_Ks_dict = {}
    ellipsoid_axes = ([1, 1, 1], [3, 3, 3], [7, 7, 7])
    for vs in ellipsoid_axes:
        robust_cnst = ta.constraint.RobustCanonicalLinearConstraint(cnst, vs)
        robust_pp = ta.algorithm.TOPPRA([robust_cnst], path)
        robust_Ks = robust_pp.compute_controllable_sets(0, 1e-3)
        robust_Ks_dict[vs[0]] = robust_Ks

    plt.plot(Ks, '--', c='blue')
    plt.plot(robust_Ks_dict[1], '--', c='green')
    plt.plot(robust_Ks_dict[3], '--', c='orange')
    plt.plot(robust_Ks_dict[7], '--', c='red')
    plt.show()


if __name__ == '__main__':
    main()
