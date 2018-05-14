import numpy as np

def inv_dyn(rave_robot, q, qd, qdd, forceslist=None, returncomponents=True):
    """Compure Torque required to generate desired acceleration.

    This is a simple wrapper around OpenRAVE's ComputeInverseDynamics
    function.

    M(q) qdd + C(q, qd) qd + g(q) = tau
    
    Parameters
    ----------
    rave_robot : OpenRAVE robot
    q : ndarray
    qd : ndarray
    qdd : ndarray
    forceslist : list
    returncomponents : bool

    Returns
    -------
    out :
    """
    if np.isscalar(q):  # Scalar case
        q_ = [q]
        qd_ = [qd]
        qdd_ = [qdd]
    else:
        q_ = q
        qd_ = qd
        qdd_ = qdd

    # Temporary remove velocity Limits
    vlim = rave_robot.GetDOFVelocityLimits()
    alim = rave_robot.GetDOFAccelerationLimits()
    rave_robot.SetDOFVelocityLimits(100 * vlim)
    rave_robot.SetDOFAccelerationLimits(100 * alim)
    with rave_robot:
        rave_robot.SetDOFValues(q_)
        rave_robot.SetDOFVelocities(qd_)
        res = rave_robot.ComputeInverseDynamics(
            qdd_, forceslist, returncomponents=returncomponents)
    
    rave_robot.SetDOFVelocityLimits(vlim)
    rave_robot.SetDOFAccelerationLimits(alim)

    return res


def fw_dyn(rave_robot, q, qd, tau):
    """ compute forward dynamics

    M(q) qdd + h(q, qd) qd + g(q) = tau
    """
    with rave_robot:
        n = rave_robot.GetDOF()
        rave_robot.SetDOFValues(q)
        rave_robot.SetDOFVelocities(qd)

        # 1: compute A = h(q, qd) qd + g(q)
        tm, tc, tg = rave_robot.ComputeInverseDynamics(
            np.zeros(n), None, returncomponents=True)

        # 2: compute M
        M = []
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            tmi, _, _ = rave_robot.ComputeInverseDynamics(
                ei, None, returncomponents=True)
            M.append(tmi)
        M = np.array(M).T

        # print M, tc, tg

        # 3: compute qdd
        qdd = np.linalg.solve(M, tau - tc - tg)

    return qdd

