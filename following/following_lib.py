import numpy as np
import cvxpy as cvx
import ecos
import toppra as ta
from .rave import fw_dyn, inv_dyn
from scipy.integrate import ode
from scipy.sparse import csc_matrix
import quadprog
import coloredlogs
import logging
logger = logging.getLogger(__name__)


def try_load_denso(env):
    if not env.Load("denso.dae"):
        logger.info("Denso model not found. Downloading...")
        import urllib
        urllib.urlretrieve("https://raw.githubusercontent.com/start-jsk/denso/indigo-devel/vs060/model/denso-vs060.mujin.dae", "denso.dae")
        logger.info("Download finished. Reloading.")
        res = env.Load("denso.dae")
        assert res


class ExperimentBase(object):
    """Base class for the experiments done in my ICRA 2018 paper on path tracking.

    Args:
    robot: OpenRAVE robot
    path: SplineInterpolator. The geometric path.
    dt: float. Sample time.

    """
    def __init__(self, robot, path, dt=1e-3, cloned_robot=None):
        self.robot = robot
        self.cloned_robot = cloned_robot
        self.path = path
        self.sys = PathTrackingSimulation(robot)
        self.dof = self.sys.dof
        self.dt = dt

    def set_robot_joint(self, q):
        self.robot.SetActiveDOFValues(q)

    def set_reference_robot_joint(self, q):
        if self.cloned_robot is not None:
            self.cloned_robot.SetActiveDOFValues(q)

    def set_noise_level(self, x):
        """Scalar to multiply with unit noise function.

        Args:
            x: A scalar, used to multiply with the unit noise
            function.
        """
        self.noise_level = x

    def set_initial_error(self, e):
        """ Initial state error for path tracking.

        Args:
            e: A 2n-dimensional vector, the tracking error.
        """
        self.initial_err = e

    def path_controller(self, i, s, sd, q, qd):
        """ The overall Path Controller.

        Should return a tuple (u, tau). u should be a float, tau should be a n-dimensional vector.

        """
        pass

    def set_unit_noise_function(self, func):
        """Unit noise function.

        At each simulation step, compute the unit noise vector, which
        will then added to the torque command.

        Args:
            func: The unit noise function, taking a scalar producing
            the torque

        Returns:
            None

        """
        self.noise_function = func

    def run(self):
        """Run the simulation.

        Run the simulation until path position _s is larger than
        0.99999

        Returns:
            A dictionary containing results of the experiment.
        """
        logger.debug('Starting simulation')
        traj_e = []
        traj_t = []
        traj_s = []
        traj_sd = []
        traj_q = []
        traj_qd = []
        traj_tau = []
        traj_sdd = []
        while self.sys._s < 1 - 1e-3:
            logger.debug('Simulation progress {:f}%'.format(
                100 * self.sys._s / 1))
            # Actual Torque
            sdd, tau = self.path_controller(0, self.sys._s,
                                            self.sys._sd,
                                            self.sys._q,
                                            self.sys._qd)
            tau += self.noise_function(self.sys._t) * self.noise_level
            # Simulate
            self.sys.set_control(tau, sdd)
            self.sys.integrate(self.sys._t + self.dt)
            # Store infos
            traj_t.append(self.sys._t)
            traj_e.append(self.compute_current_error())
            traj_s.append(self.sys._s)
            traj_sd.append(self.sys._sd)
            traj_q.append(np.array(self.sys._q))
            traj_qd.append(self.sys._qd)
            traj_tau.append(tau)
            traj_sdd.append(sdd)
            self.set_robot_joint(self.sys._q)
            self.set_reference_robot_joint(self.path.cspl(self.sys._s))

        return {'traj_t': np.array(traj_t),
                'traj_e': np.array(traj_e),
                'traj_sd': np.array(traj_sd),
                'traj_s': np.array(traj_s),
                'traj_tau': np.array(traj_tau),
                'traj_q': np.array(traj_q),
        }

    def compute_current_error(self):
        """Compute the current tracking error.

        Returns:
            A 2n-dimensional vector, tracking error.
        """
        eq = self.sys._q - self.path.cspl(self.sys._s)
        eqd = self.sys._qd - self.path.cspld(self.sys._s) * self.sys._sd
        return np.r_[eq, eqd]

    def reset(self):
        """Reset the simulation.
        """
        q_initial = self.path.cspl(0) + self.initial_err[:self.dof]
        qd_initial = np.zeros(self.dof) + self.initial_err[self.dof:]
        self.sys.set_initial_value(q_initial, qd_initial)


def _solve_sdd_minmax(E, F, tau_min, tau_max):
    """ Compute online acceleration bounds.

    Given the following inequality
         tau_min <= E sdd + F <= tau_max

    Args:
        E: A vector.
        F: A vector.
        tau_min: A vector.
        tau_max: A vector.
    Returns:
        sdd_min: A scalar.
        sdd_max: A scalar.
    """
    # Transform to A sdd + B <= 0 form
    A = np.hstack((E, -E))
    B = np.hstack((F - tau_max, - F + tau_min))
    sdd_max = 1e5
    sdd_min = -1e5
    for i in range(A.shape[0]):
        if A[i] < 0:
            sdd_min = max(sdd_min, - B[i] / A[i])
        elif A[i] > 0:
            sdd_max = min(sdd_max, - B[i] / A[i])
    return sdd_min, sdd_max


class ExperimentOS(ExperimentBase):
    """Experiment for the OS controller.

    """
    def __init__(self, robot, path, us, xs, ss, tau_min, tau_max, lamb, cloned_robot=None):
        super(ExperimentOS, self).__init__(robot, path, cloned_robot=cloned_robot)
        self.xs = xs
        self.us = us
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.lamb = lamb
        self.ss = ss

    def path_controller(self, i, s, sd, q, qd):
        e = self.compute_current_error()
        eq = e[:self.dof]
        eqd = e[self.dof:]
        E = inv_dyn(self.robot,
                    self.sys._q,
                    self.sys._qd,
                    self.path.cspld(self.sys._s))[0]

        F = inv_dyn(self.robot,
                    self.sys._q,
                    self.sys._qd,
                    self.path.cspldd(self.sys._s) * self.sys._sd ** 2
                    - 2 * self.lamb * eqd - self.lamb ** 2 * eq,
                    returncomponents=False)
        sdd_min, sdd_max = _solve_sdd_minmax(E, F, self.tau_min, self.tau_max)
        # Check for negative path velocity
        assert self.sys._sd > -1e-8
        # Add additional constraint to ensure accelerations are chosen to avoid negative velocities
        sdd_min_zero = - self.sys._sd / 1e-3
        sdd_min = max(sdd_min, sdd_min_zero)
        sd_ref = np.sqrt(np.interp(self.sys._s, self.ss, self.xs))
        sdd_ref = self.us[self.ss[:-1] <= self.sys._s][-1]  # weird code
        sdd_nominal = sdd_ref - 10 * (self.sys._sd ** 2 - sd_ref ** 2)
        # Clipped path acceleration
        if sdd_min < sdd_max:
            sdd = np.clip(sdd_nominal, sdd_min, sdd_max)
            # sdd = sdd_nominal  # Keep sdd_nominal the same
        else:
            sdd = sdd_nominal  # Keep sdd_nominal the same
        tau = E * sdd + F
        tau_clipped = np.clip(tau, self.tau_min, self.tau_max)
        return sdd, tau_clipped


class ExperimentOSG(ExperimentBase):
    """Experiment for the OSG controller.

    Args
        robot: OpenRAVE robot.
        path: toppra geometric path.
        Ks: (N +1, 2) array. Robust controllable sets.
        

    """
    def __init__(self, robot, path, Ks, ss, tau_min, tau_max, lamb, cloned_robot=None):
        super(ExperimentOSG, self).__init__(robot, path, cloned_robot=cloned_robot)
        self.Ks = Ks
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.lamb = lamb
        self.ss = ss

    def path_controller(self, i, s, sd, q, qd):
        e = self.compute_current_error()
        eq = e[:self.dof]
        eqd = e[self.dof:]
        E = inv_dyn(self.robot,
                    self.sys._q,
                    self.sys._qd,
                    self.path.cspld(self.sys._s))[0]

        F = inv_dyn(self.robot,
                    self.sys._q,
                    self.sys._qd,
                    self.path.cspldd(self.sys._s) * self.sys._sd ** 2
                    - 2 * self.lamb * eqd - self.lamb ** 2 * eq,
                    returncomponents=False)
        sdd_min, sdd_max = _solve_sdd_minmax(E, F, self.tau_min, self.tau_max)

        # Bound from robust controllable set
        target_index = np.where(self.ss > self.sys._s)[0][0]
        if target_index < len(self.ss) - 1:
            target_index += 1
        sdd_max_robust = ((self.Ks[target_index][1] - self.sys._sd ** 2)
                          / 2
                          / (self.ss[target_index] - self.sys._s))
        sdd_min_robust = ((self.Ks[target_index][0] - self.sys._sd ** 2)
                          / 2
                          / (self.ss[target_index] - self.sys._s))
        if sdd_min > sdd_max:
            # sdd = np.clip(0, sdd_min_robust, sdd_max_robust)
            sdd = np.clip(sdd_max_robust, sdd_min, sdd_max)
        else:
            sdd = np.clip(sdd_max_robust, sdd_min, sdd_max)
        sdd = max(sdd_min_robust, sdd)
            
        tau = E * sdd + F
        tau_clipped = np.clip(tau, self.tau_min, self.tau_max)
        return sdd, tau_clipped


class ExperimentTT(ExperimentBase):
    """ Experiment for the TT controller.
    """
    def __init__(self, robot, path, qs, qds, qdds, tau_min, tau_max, lamb, cloned_robot=None):
        super(ExperimentTT, self).__init__(robot, path, cloned_robot=cloned_robot)
        self.qs = qs
        self.qds = qds
        self.qdds = qdds
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.lamb = lamb

    def path_controller(self, i, s, sd, q, qd):
        """ Control law.

        Args:
            i: An Int, index of the current state.
            s: A Float, the path position.
            sd: A Float, the path velocity.
            q: A n-dimensional vector, current joint position.
            qd: A n-dimensional vector, current joint velocity.

        Returns:
            A Float, the path acceleration.
            A n-dimensional vector, the joint torque.
        """
        eq = q - self.qs[i]
        eqd = qd - self.qds[i]
        qdd_desired = (self.qdds[i]
                       - 2 * self.lamb * eqd
                       - self.lamb ** 2 * eq)
        tau = inv_dyn(self.robot, q, qd, qdd_desired,
                      returncomponents=False)
        tau_clipped = np.clip(tau, self.tau_min, self.tau_max)
        return 0., tau_clipped

    def run(self):
        """ Run simulation.
        """
        logger.debug('Starting simulation')
        traj_e = []
        traj_t = []
        traj_s = []
        traj_sd = []
        traj_q = []
        traj_qd = []
        traj_tau = []
        traj_sdd = []
        for i in range(self.qs.shape[0]):
            logger.debug('Simulation progress {:f}%'.format(
                100 * float(i) / self.qs.shape[0]))
            traj_e.append(self.compute_current_error(i))
            # Actual Torque
            sdd, tau = self.path_controller(i, self.sys._s,
                                            self.sys._sd,
                                            self.sys._q,
                                            self.sys._qd)
            tau += self.noise_function(self.sys._t) * self.noise_level
            # Simulate
            self.sys.set_control(tau, sdd)
            self.sys.integrate(self.sys._t + self.dt)
            # Store infos
            traj_t.append(self.sys._t)
            traj_s.append(self.sys._s)
            traj_sd.append(self.sys._sd)
            traj_q.append(np.array(self.sys._q))
            traj_qd.append(self.sys._qd)
            traj_tau.append(tau)
            traj_sdd.append(sdd)
            self.set_robot_joint(self.sys._q)
            self.set_reference_robot_joint(self.qs[i])

        return {'traj_t': np.array(traj_t),
                'traj_e': np.array(traj_e),
                'traj_sd': np.array(traj_sd),
                'traj_s': np.array(traj_s),
                'traj_tau': np.array(traj_tau),
                'traj_q': np.array(traj_q),
        }

    def compute_current_error(self, i):
        """Compute the current tracking error.

        Args:
            i: An Int, the index of the current trajectory.

        Returns:
            A 2n-dimensional vector, tracking error.
        """
        eq = self.sys._q - self.qs[i]
        eqd = self.sys._qd - self.qds[i]
        return np.r_[eq, eqd]


class PathTrackingSimulation(object):
    """ A simulator for Path Tracking experiment

    """
    def __init__(self, robot, integrator='dopri5', dt=1e-3):
        self.robot = robot
        self.integrator = integrator
        self.dt = dt
        self.dof = self.robot.GetDOF()

        # Right-hand side of the dynamics equation
        def X(t, y, u):
            """The ODE of a robot.

            Parameters
            ----------
            t : float
                Time.
            y : ndarray, shape (2*dof,)
                State := [position, velocity].
            u : ndarray, shape (dof,)
                Control := torque

            Returns
            -------
            out : ndarray, shape (2*dof,)
                Derivative of State := [velocity, acceleration]
            """
            q = y[:self.dof]
            qd = y[self.dof:]
            qdd = fw_dyn(robot, q, qd, u)
            return np.r_[qd, qdd]

        self.r = ode(X).set_integrator(self.integrator)

    def set_initial_value(self, q, qd, s=0., sd=0., t=0):
        """ Set initial values for the state.

        Parameters
        ----------
        q : ndarray
        qd : ndarray
        s : float
        sd : float
        t : float, optional
            Starting time, default to 0.
        """
        self._q = q
        self._qd = qd
        self._s = s
        self._sd = sd
        self._t = t

    def set_control(self, tau, sdd=0.):
        """
        """
        self._tau = tau
        self._sdd = sdd

    def integrate(self, t_new):
        """
        """
        self.r.set_initial_value(np.r_[self._q, self._qd], t=self._t)
        self.r.set_f_params(self._tau)
        self.r.integrate(t_new)
        dt = t_new - self._t
        self._s = self._s + self._sd * dt + self._sdd * dt ** 2 / 2
        self._sd = self._sd + self._sdd * dt
        # Copy new value
        self._q = self.r.y[:self.dof]
        self._qd = self.r.y[self.dof:]
        self._t = self.r.t


class RobustPathConstraint(object):
    """Discretized robust constraint on a path.

    A robust path contraint has the following form

    ..math::
        (a_i + \Delta a_i) u + (b_i + \Delta b_i) x + (c_i + \Delta c_i) <= 0

    where (\Delta a_i, \Delta b_i, \Delta c_i) are unknown and bounded
    perturbations with the following representation
    
    ..math::
        [\Delta a_ij, \Delta b_ij, \Delta c_ij]^\top = w_{ij} \mathbf P_{ij} u, \|u\|_2 \leq 1.
    

    A method to handle path constraints of this kind is 
    

    Parameters
    ----------
    vs : ndarray, shape (N+1, m, 3)
        Mean of discretized constraints.
    Ps : ndarray, shape (N+1, m, 3, 3)
        Ellipsoidal approximation of the constraints.
    ss : ndarray, shape (N+1,)
        Array of discretized gridpoints.

    Returns
    -------
    out : RobustPathConstraint

    """
    def __init__(self, vs, Ps, ss):
        self.ss = ss
        self.vs = vs
        self.Ps = Ps


def robust_one_step(c, i, I_target, w_i, w_j,
                    solver='ECOS',
                    verbose=False,
                    method='CVX'):
    """Compute the robust one-step set.

    The current implementation uses CVXPY or ECOS. CVXPY should be
    correct, while ECOS would give better running times.

    This function computes the robust one-step using the interpolation
    discretization scheme. This scheme requires (x, u) to satisfy the
    constraints at stage i and (x + 2 Delta u, u) to satisfy the
    constraints at stage i+1.

    The two parameters w_i and w_j are scaling factors on the degree
    of uncertainties. In future it is probably better to keep it to
    one, or a fixed positive number.

    Parameters
    ----------
    c : RobustPathConstraint
        The Robust Path Constraint to consider.
    i : int
        Index of the "safe" squared velocities.
    I_target : ndarray, shape (2,)
        The interval of "target" squared velocities.
    w_i : float
        Expected noise in the i-th stage.
    w_j : float
        Expected noise in the j=(i+1)-th stage.

    Returns
    -------
    I_safe : ndarray, shape (2,)
        The interval of "safe" squared velocities.

    """
    if method == "CVX":  # The most mature method
        x = cvx.Variable(2)  # [u_i, \dot s_i ^ 2]
        m = c.vs[i].shape[0]
        constraints = [x[1] + 2 * (c.ss[i+1] - c.ss[i]) * x[0] <= I_target[1],
                       x[1] + 2 * (c.ss[i+1] - c.ss[i]) * x[0] >= I_target[0],
                       x[1] >= 0]

        # Constraint at the i-th side
        for k in range(m):
            _ = (c.vs[i, k][0:2] * x + c.vs[i, k][2]
                 + w_i *
                 cvx.norm(c.Ps[i, k, :, 0:2] * x + c.Ps[i, k, :, 2])) <= 0
            constraints.append(_)

        # Constraint at the (i+1)-th side
        H = np.array([[1., 0, 0],
                      [2 * (c.ss[i + 1] - c.ss[i]), 1., 0],
                      [0., 0, 1]])
        for k in range(m):
            v_tilde_j = np.dot(c.vs[i + 1, k], H)
            P_tilde_j = np.dot(c.Ps[i + 1, k], H)
            _ = (v_tilde_j[0:2] * x + v_tilde_j[2]
                 + w_j *
                 cvx.norm(P_tilde_j[:, 0:2] * x + P_tilde_j[:, 2])) <= 0
            constraints.append(_)

        # Solve for maximum and minimum
        obj_max = cvx.Maximize(x[1])
        prob_max = cvx.Problem(obj_max, constraints)
        prob_max.solve(solver=solver, verbose=verbose)
        obj_min = cvx.Minimize(x[1])
        prob_min = cvx.Problem(obj_min, constraints)
        prob_min.solve(solver=solver, verbose=verbose)
        return np.r_[prob_min.value, prob_max.value]

    elif method == 'ECOS':
        m = c.vs[i].shape[0]
        dims = {'l': 3, 'q': [4] * 2 * m}
        G = np.zeros((3 + 4 * 2 * m, 2))
        h = np.zeros(3 + 4 * 2 * m)
        G[0, :] = [2 * (c.ss[i+1] - c.ss[i]), 1]
        G[1, :] = [- 2 * (c.ss[i+1] - c.ss[i]), - 1]
        G[2, :] = [0, -1.]
        h[0] = I_target[1]
        h[1] = - I_target[0]
        h[2] = 0
        for k in range(m):
            G[3 + 4 * k, :] = c.vs[i][k, :2]
            G[3 + 4 * k + 1: 3 + 4 * k + 4] = - w_i * c.Ps[i][k][:, :2]
            h[3 + 4 * k] = - c.vs[i][k, 2]
            h[3 + 4 * k + 1: 3 + 4 * k + 4] = w_i * c.Ps[i][k][:, 2]

        # Constraint at the (i+1)-th side
        H = np.array([[1., 0, 0],
                      [2 * (c.ss[i+1] - c.ss[i]), 1., 0],
                      [0., 0, 1]])
        for k in range(m):
            v_tilde_j = np.dot(c.vs[i+1, k], H)
            P_tilde_j = np.dot(c.Ps[i+1, k], H)
            G[3 + 4 * m + 4 * k, :] = v_tilde_j[:2]
            G[3 + 4 * m + 4 * k + 1: 3 + 4 * m + 4 * k + 4] = - w_j * P_tilde_j[:, :2]
            h[3 + 4 * m + 4 * k] = - v_tilde_j[2]
            h[3 + 4 * m + 4 * k + 1: 3 + 4 * m + 4 * k + 4] = w_j * P_tilde_j[:, 2]
        G = csc_matrix(G)
        c_min = np.array([0., 1.])
        c_max = np.array([0., - 1.])
        # return ecos.solve(c_max, G, h, dims, verbose=verbose)
        x_min = ecos.solve(c_min, G, h, dims, verbose=verbose)['x'][1]
        x_max = ecos.solve(c_max, G, h, dims, verbose=verbose)['x'][1]
        return x_min, x_max


def generate_random_torque_constraints_samples(robot, path, s, sdmin, sdmax, w_max, N_sample, K):
    """Generate torque constraint samples randomly.

    Using parameter set (s, sdmin, sdmax, w_max), generate N_sample
    of (sd^2, E, F) such that
    - sdmin < sd < sdmax
    - |e| <= w_max
    - q = p(s) + e[:6]
    - qdot = p(s) sd + e[6:]

    Parameters
    ----------
    robot : OpenRAVE robot
    path : interpolator
    s : float
    sdmin : float
        Random velocities are generated in (sdmin, sdmax).
    sdmax : float
        Random velocities are generated in (sdmin, sdmax).
    w_max : float
        Random state is at most w_max away from (p(s), p_s(s) * sd).
    N_sample : int
        Number of samples.
    K : ndarray, shape (12, 6)
        Error Gain Matrix.

    Returns
    -------
    data : list
        List contains N_sample of samples.
    """
    p = path.cspl(s)
    p_s = path.cspld(s)
    p_ss = path.cspldd(s)
    data = []
    for i in range(N_sample):
        e = np.random.randn(12)
        e = w_max * e / np.linalg.norm(e)
        sdot = np.random.rand() * (sdmax - sdmin) + sdmin
        q = p + e[:6]
        qdot = p_s * sdot + e[6:]

        # tau = _t1 sddot + _t2
        _t1, _, _ = ta.inv_dyn(robot, q, p_s, p_s)
        _t2 = ta.inv_dyn(robot, q, qdot, p_ss * sdot ** 2 + np.dot(K, e),
                         returncomponents=False)

        tau_bnd = robot.GetDOFTorqueLimits()
        E = np.hstack((_t1, -_t1))
        F = np.hstack((_t2 - tau_bnd, - _t2 - tau_bnd))
        data.append([sdot ** 2, E, F])
    return data


def compute_trajectory_points(path, sgrid,
                              ugrid, xgrid,
                              dt=1e-2, smooth=True,
                              smooth_eps=1e-4):
    """Compute trajectory with uniform sampling time.

    Note
    ----
    Additionally, if `smooth` is True, the return trajectory is smooth
    using least-square technique. The return trajectory, also
    satisfies the discrete transition relation. That is

    q[i+1] = q[i] + qd[i] * dt + qdd[i] * dt ^ 2 / 2
    qd[i+1] = qd[i] + qdd[i] * dt

    If one finds that the function takes too much time to terminate,
    then it is very likely that the most time-consuming part is
    least-square. In this case, there are several options that one
    might take.
    1. Set `smooth` to False. This might return badly conditioned
    trajectory.
    2. Reduce `dt`. This is the recommended option.

    Parameters
    ----------
    path : interpolator
    sgrid : ndarray, shape (N+1,)
        Array of gridpoints.
    ugrid : ndarray, shape (N,)
        Array of controls.
    xgrid : ndarray, shape (N+1,)
        Array of squared velocities.
    dt : float, optional
        Sampling time step.
    smooth : bool, optional
        If True, do least-square smoothing. See above for more details.
    smooth_eps : float, optional
        Relative gain of minimizing variations of joint accelerations.

    Returns
    -------
    tgrid : ndarray, shape (M)
        Time at each gridpoints.
    q : ndarray, shape (M, dof)
        Joint positions at each gridpoints.
    qd : ndarray, shape (M, dof)
        Joint velocities at each gridpoints.
    qdd : ndarray, shape (M, dof)
        Joint accelerations at each gridpoints.

    """
    tgrid = np.zeros_like(sgrid)  # Array of time at each gridpoint
    N = sgrid.shape[0] - 1
    sdgrid = np.sqrt(xgrid)
    for i in range(N):
        tgrid[i+1] = ((sgrid[i+1] - sgrid[i]) / (sdgrid[i] + sdgrid[i+1]) * 2
                      + tgrid[i])
    # shape (M+1,) array of sampled time
    tsample = np.arange(tgrid[0], tgrid[-1], dt)
    ssample = np.zeros_like(tsample)  # sampled position
    xsample = np.zeros_like(tsample)  # sampled velocity squared
    sdsample = np.zeros_like(tsample)  # sampled velocity
    usample = np.zeros_like(tsample)  # sampled path acceleration
    igrid = 0
    for i, t in enumerate(tsample):
        while t > tgrid[igrid + 1]:
            igrid += 1
        usample[i] = ugrid[igrid]
        sdsample[i] = sdgrid[igrid] + (t - tgrid[igrid]) * usample[i]
        xsample[i] = sdsample[i] ** 2
        ssample[i] = (sgrid[igrid] +
                      (xsample[i] - xgrid[igrid]) / 2 / usample[i])

    q = path.eval(ssample)
    qs = path.evald(ssample)  # derivative w.r.t [path position] s
    qss = path.evaldd(ssample)

    def array_mul(vectors, scalars):
        # given array of vectors and array of scalars
        # multiply each vector with each scalar
        res = np.zeros_like(vectors)
        for i in range(scalars.shape[0]):
            res[i] = vectors[i] * scalars[i]
        return res

    qd = array_mul(qs, sdsample)
    qdd = array_mul(qs, usample) + array_mul(qss, sdsample ** 2)

    # Smoothing
    if not smooth:
        return tsample, q, qd, qdd, ssample
    else:
        dof = q.shape[1]
        # Still slow, I will now try QP with quadprog
        A = np.array([[1., dt], [0, 1.]])
        B = np.array([dt ** 2 / 2, dt])
        M = tsample.shape[0] - 1
        Phi = np.zeros((2 * M, M))
        for i in range(M):  # Block diagonal
            Phi[2 * i: 2 * i + 2, i] = B
        for i in range(1, M):  # First column
            Phi[2 * i: 2 * i + 2, 0] = np.dot(A, Phi[2 * i - 2: 2 * i, 0])
        for i in range(1, M):  # Next column
            Phi[2 * i:, i] = Phi[2 * i - 2: 2 * M - 2, i - 1]

        Beta = np.zeros((2 * M, 2))
        Beta[0: 2, :] = A
        for i in range(1, M):
            Beta[2 * i: 2 * i + 2, :] = np.dot(A, Beta[2 * i - 2: 2 * i, :])

        Delta = np.zeros((M - 1, M))
        for i in range(M-1):
            Delta[i, i] = 1
            Delta[i, i + 1] = - 1

        for k in range(dof):
            Xd = np.vstack((q[1:, k], qd[1:, k])).T.flatten()  # numpy magic
            x0 = np.r_[q[0, k], qd[0, k]]
            xM = np.r_[q[-1, k], qd[-1, k]]

            G = np.dot(Phi.T, Phi) + np.dot(Delta.T, Delta) * smooth_eps
            a = - np.dot(Phi.T, Beta.dot(x0) - Xd)
            C = Phi[2 * M - 2:].T
            b = xM - Beta[2 * M - 2:].dot(x0)
            sol = quadprog.solve_qp(G, a, C, b, meq=2)[0]
            Xsol = np.dot(Phi, sol) + np.dot(Beta, x0)
            Xsol = Xsol.reshape(-1, 2)
            q[1:, k] = Xsol[:, 0]
            qd[1:, k] = Xsol[:, 1]
            qdd[:-1, k] = sol
            qdd[-1, k] = sol[-1]

        return tsample, q, qd, qdd, ssample
