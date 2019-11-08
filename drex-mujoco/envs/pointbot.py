"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise and air resistance
proportional to its velocity. State representation is (x, vx, y, vy). Action
representation is (fx, fy), and mass is assumed to be 1.
"""

import os
import pickle

import os.path as osp
import numpy as np
import tensorflow as tf
from gym import Env
from gym import utils
from gym.spaces import Box

from .pointbot_const import *

def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)

def lqr_gains(A, B, Q, R, T):
    Ps = [Q]
    Ks = []
    for t in range(T):
        P = Ps[-1]
        Ps.append(Q + A.T.dot(P).dot(A) - A.T.dot(P).dot(B)
            .dot(np.linalg.inv(R + B.T.dot(P).dot(B))).dot(B.T).dot(P).dot(A))
    Ps.reverse()
    for t in range(T):
        Ks.append(-np.linalg.inv(R + B.T.dot(Ps[t+1]).dot(B)).dot(B.T).dot(P).dot(A))
    return Ks, Ps


class PointBot(Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(4)
        self.A[2,3] = self.A[0,1] = 1
        self.A[1,1] = self.A[3,3] = 1 - AIR_RESIST
        self.B = np.array([[0,0], [1,0], [0,0], [0,1]])
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE, np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        # self.obstacle = ComplexObstacle([[-30, -20], [-20, 20]])
        self.start_state = START_STATE
        # TODO: remove this later or find clean way to set mode in gym env
        self.mode = 1
        self.obstacle = OBSTACLE[self.mode]
        self.start_state = [-100, 0, 0, 0]
        # DONE TODO

    def set_mode(self, mode):
        self.mode = mode
        self.obstacle = OBSTACLE[mode]
        if self.mode == 1:
            self.start_state = [-100, 0, 0, 0]

    def get_random_action(self):
        print(self.action_space)

    def step(self, a):
        a = process_action(a)
        next_state = self._next_state(self.state, a)
        cur_cost = self.step_cost(self.state, a)
        self.cost.append(cur_cost)
        if not self.obstacle(self.state):
            self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.done = HORIZON <= self.time
        if self.done and not self.is_stable(self.state):
            self.cost[-1] += FAILURE_COST
            cur_cost += FAILURE_COST
        return self.state, cur_cost, self.done, {}

    def reset(self):
        self.state = self.start_state + np.random.randn(4)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        return self.state

    def _next_state(self, s, a):
        return self.A.dot(s) + self.B.dot(a) + NOISE_SCALE * np.random.randn(len(s))

    def step_cost(self, s, a):
        if HARD_MODE:
            return int(np.linalg.norm(np.subtract(GOAL_STATE, s)) > GOAL_THRESH) + self.obstacle(s)
        return np.linalg.norm(np.subtract(GOAL_STATE, s))

    def collision_cost(self, obs):
        return self.obstacle(obs)


    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectory(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:,0], states[:,2])
        plt.show()

    # Returns whether a state is stable or not
    def is_stable(self, s):
        return np.linalg.norm(np.subtract(GOAL_STATE, s)) <= GOAL_THRESH

    def teacher(self, sess=None):
        return PointBotTeacher()

    def get_features(self, state):
        return state

class PointBotTeacher(object):

    def __init__(self):
        self.env = PointBot()
        self.Ks, self.Ps = lqr_gains(self.env.A, self.env.B, np.eye(4), 50 * np.eye(2), HORIZON)
        self.demonstrations = []
        self.default_noise = 0.2

    # all get_rollout functions for all envs should have a noise parameter
    def get_rollout(self, noise_param_in=None, mode="eps_greedy"):
        if mode == "eps_greedy":
            if noise_param_in is None:
                noise_param = 0
            else:
                noise_param = noise_param_in

        elif mode == "gaussian_noise":
            if noise_param_in is None:
                noise_param = 0
            else:
                noise_param = noise_param_in

        obs = self.env.reset()
        O, features, A, cost_sum, costs = [obs], [self.env.get_features(obs)], [], 0, []
        for i in range(HORIZON):
            if self.env.mode == 1:
                noise_idx = np.random.randint(int(2 * HORIZON /3))
                # if i < HORIZON / 2:
                #     action = [0.1, 0.1]
                # else:
                action = self._expert_control(obs, i)
            else:
                noise_idx = np.random.randint(int(HORIZON))
                if i < HORIZON / 4:
                    action = [0.1, 0.25]
                elif i < HORIZON / 2:
                    action = [0.4, 0.]
                elif i < HORIZON / 3 * 2:
                    action = [0, -0.5]
                else:
                    action = self._expert_control(obs, i)

            if i < noise_idx:
                if mode == "eps_greedy":
                    assert(noise_param <= 1)
                    if np.random.random() < noise_param:
                        action = self.env.action_space.sample()
                    else: # Do what expert would do
                        if np.random.random() < self.default_noise:
                            action = self.env.action_space.sample()

                elif mode == "gaussian_noise":
                    action = (np.array(action) +  np.random.normal(0, noise_param + self.default_noise, self.env.action_space.shape[0])).tolist()
                else:
                    print("Invalid Mode!")
                    assert(False)

            A.append(action)
            obs, cost, done, info = self.env.step(action)
            O.append(obs)
            features.append(self.env.get_features(obs))
            cost_sum += cost
            costs.append(cost)
            if done:
                break

        return {
            "obs": np.array(O),
            "features": np.array(features),
            "noise": noise_param,
            "actions": np.array(A),
            "cost_sum": cost_sum,
            "costs": np.array(costs),
        }

    def _get_gain(self, t):
        return self.Ks[t]

    def _expert_control(self, s, t):
        return self._get_gain(t).dot(s)

if __name__ == "__main__":
    teacher = PointBotTeacher()
    teacher.env.set_mode(1)
    num_per_noise = 10
    rollouts = np.array([ [teacher.get_rollout(noise) for _ in range(num_per_noise)] for noise in 0.15*np.arange(5)])
    print(rollouts.shape)
    pickle.dump( rollouts, open( "rollouts.p", "wb" ) )

    
