#!/usr/local/bin/python3

from math import atan2

""" Definitions:

configuration (q): a 2D planar rigid body configuration, specified using (x, y)
                   of the origin of the local frame w.r.t. global frame, and
                   an angle theta. q = (x, y, theta)

control (u): Specified using u = (ux, uy, uth), expressed w.r.t. local frame.
             u = (1, 0, 1) means drive forwards (current local x), omega = 1

configuration velocity (qd): The control, transformed into the global frame.
                             qd = (xd, yd, thd)

transform (T): A homogeneous transform matrix expressing a configuration,
               coordinate transform, or rigid body motion

Things we want to compute:

  -- compute transform from from configuration: transform_from_config
  -- compute configuration from transform: config_from_transform
  -- compute tranform from control + duration: transform_from_control
  -- compute transform from sequence of controls and durations:
         transform_from_sequence

"""

from numpy import pi, cos, sin, sinc, array, sqrt, arctan, \
    round
import sys
print(sys.version)


controls_rs = array([
            [1, 0, 0],
            [-1, 0, 0],
            [1, 0, -1],
            [1, 0, 1],
            [-1, 0, -1],
            [-1, 0, 1]])


def config_from_transform(T):
    x = T[0][2]
    y = T[1][2]
    theta = atan2(T[1][0], T[0][0])
    return x, y, theta


def verc(x, epsilon=.0001):
    # the cardinal versine fuction, (1 - cos x) / x, and 0 at 0
    if(x < epsilon):
        # compute Taylor series approximation near zero
        return x / 2 + x * x * x / 24 + x ** 5 / 720
    else:
        return (1 - cos(x)) / x


def transform_from_config(q):

    x = q[0]
    y = q[1]
    th = q[2]
    c = cos(th)
    s = sin(th)

    T = array([[c, -s, x],
               [s,  c, y],
               [0, 0, 1]])

    return T


def transform_from_control(u, t):
    # equation 3.14 from Furtuna's thesis.
    # compute the homogeneous transformation matrix
    # that moves points of the rigid body through a time t,
    # using a control u of the form (xdot, ydot, thetadot)

    ux = u[0]
    uy = u[1]

    th = t * u[2]  # total angle rotated through
    c = cos(th)
    s = sin(th)
    sinc_th = sinc(th / pi)  # Furtuna uses un-normalized cardinal sine
    verc_th = verc(th)

    T = array([[c, -s, ux * t * sinc_th - uy * t * verc_th],
               [s,  c, ux * t * verc_th + uy * t * sinc_th],
               [0, 0, 1]])

    return(T)


def transform_from_sequence(sequ, seqt):
    # sequ is a list of control inputs, and seqt is a list of durations.
    # output: a single transform representing the final configuration

    resultT = transform_from_config([0, 0, 0])  # identity transform

    for i in range(len(seqt)):
        u = sequ[i]
        duration = seqt[i]

        currentT = transform_from_control(u, duration)
        resultT = resultT @ currentT

    return resultT


def sample_trajectory(sequ, seqt, final_t, n, epsilon=.00001):
    # Compute a list of n + 1 transforms

    # n is number of timesteps. number of samples = n + 1, since first
    # and last configuration are both in returned list

    timestep = (final_t / n)
    t = 0
    current_action = 0

    # the list of transforms we intend to build and return
    T_list = []

    # transform up through the previous complete action
    T_prev = transform_from_config([0, 0, 0])
    t_prev_total = 0  # accumulated time for completed controls

    while t < final_t + epsilon and current_action < len(seqt):
        if t <= t_prev_total + seqt[current_action]:
            # still working on the current action

            u = sequ[current_action]
            duration = t - t_prev_total

            currentT = transform_from_control(u, duration)
            T_list.append(T_prev @ currentT)
            t += timestep

        else:
            # finished the current action, switch controls

            # update T_prev
            u = sequ[current_action]
            duration = seqt[current_action]
            T_prev = T_prev @ transform_from_control(u, duration)

            t_prev_total += duration  # accumulated time for completed controls
            current_action += 1

    return T_list


if __name__ == '__main__':

    # T = transform_from_control(controls_rs[0], .2)

    T = transform_from_sequence([controls_rs[0]], [.2])
    print(T)

    T_list = sample_trajectory([controls_rs[0], controls_rs[3]], [.2, .2], .4, 10)
    print(T_list)
