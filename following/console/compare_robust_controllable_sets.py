import numpy as np
import openravepy as orpy
import toppra as ta
import matplotlib.pyplot as plt
import os, argparse
import logging
logger = logging.getLogger(__name__)


def main(env=None, verbose=False, savefig=False):
    if verbose:
        ta.setup_logging("DEBUG")
    else:
        ta.setup_logging("INFO")

    # Setup the robot, the geometric path and the torque bounds
    print("Loading the environment")
    if env is None:
        env = orpy.Environment()
    else:
        env.Reset()
    env.Load(os.path.join(os.environ["TOPPRA_FOLLOWING_HOME"], 'models/denso_vs060.dae'))
    robot = env.GetRobots()[0]
    np.random.seed(11)
    waypoints = np.random.randn(5, 6) * 0.4
    path = ta.SplineInterpolator(np.linspace(0, 1, 5), waypoints)
    tau_max = np.r_[30., 50., 20., 20., 10., 10.]
    robot.SetDOFTorqueLimits(tau_max)

    # Nominal case
    print("Computing controllable sets for the ideal case.")
    cnst = ta.create_rave_torque_path_constraint(robot, 1)
    pp = ta.algorithm.TOPPRA([cnst], path)
    Ks = pp.compute_controllable_sets(0, 0)

    robust_Ks_dict = {}
    ellipsoid_axes = ([1, 1, 1], [3, 3, 3], [7, 7, 7])
    for vs in ellipsoid_axes:
        print("Computing controllable sets for perturbation ellipsoid={:}".format(vs))
        robust_cnst = ta.constraint.RobustCanonicalLinearConstraint(cnst, vs)
        robust_pp = ta.algorithm.TOPPRA([robust_cnst], path)
        robust_Ks = robust_pp.compute_controllable_sets(0, 1e-3)
        robust_Ks_dict[vs[0]] = robust_Ks

    plt.plot([0, 100], [0, 0], '--', c='black')
    plt.plot(Ks[:, 1], '--', c='blue', label="ideal case")
    plt.plot(robust_Ks_dict[1][:, 1], '--', c='green', label="low perturbation level")
    plt.plot(robust_Ks_dict[3][:, 1], '--', c='orange', label="medium perturbation level")
    plt.plot(robust_Ks_dict[7][:, 1], '--', c='red', label="high perturbation level")
    plt.legend()
    plt.title("Controllable set for different levels of perturbations.")
    plt.tight_layout()
    if savefig:
        plt.savefig("icra18-compare-robust-controllable-sets.pdf")
    plt.show()


if __name__ == '__main__':
    main()
