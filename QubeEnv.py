import gym
import numpy as np
from gym import spaces
from scipy.integrate import odeint
from gym_brt.envs.rendering import QubeRenderer

from gym_brt.control import flip_and_hold_policy


MAX_MOTOR_VOLTAGE = 3
ACT_MAX = np.asarray([MAX_MOTOR_VOLTAGE], dtype=np.float64)
# OBS_MAX = [theta, alpha, theta_dot, alpha_dot]
OBS_MAX = np.asarray([np.pi / 2, np.pi, np.inf, np.inf], dtype=np.float64)


def next_state_fn(dt=0.004):
    # Physical constants from system
    Rm, kt, km = 8.4, 0.042, 0.042  # Motor
    mr, Lr, Dr = 0.095, 0.045, 0.00027  # Rotary arm
    mp, Lp, Dp = 0.024, 0.129, 0.00005  # Pendulum arm
    g = 9.81

    Rm = np.random.normal(loc=Rm, scale=Rm / 10)
    kt = np.random.normal(loc=kt, scale=kt / 10)
    km = np.random.normal(loc=km, scale=km / 10)
    mr = np.random.normal(loc=mr, scale=mr / 10)
    Lr = np.random.normal(loc=Lr, scale=Lr / 10)
    Dr = np.random.normal(loc=Dr, scale=Dr / 10)
    mp = np.random.normal(loc=mp, scale=mp / 10)
    Lp = np.random.normal(loc=Lp, scale=Lp / 10)
    Dp = np.random.normal(loc=Dp, scale=Dp / 10)
    g = np.random.normal(loc=g, scale=g / 10)

    Jr = mr * Lr ** 2 / 12
    Jp = mp * Lp ** 2 / 12

    def next_state(
        state,
        action,
        dt=dt,
        Rm=Rm,
        kt=kt,
        km=km,
        mr=mr,
        Lr=Lr,
        Dr=Dr,
        mp=mp,
        Lp=Lp,
        Dp=Dp,
        g=g,
    ):
        theta, alpha, theta_dot, alpha_dot = state
        # Calculate the derivative of the state
        # fmt: off
        theta_dot_dot = float((-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*(-(km * (action - km * theta_dot)) / Rm)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        alpha_dot_dot = float((2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*(-(km * (action - km * theta_dot)) / Rm))*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)))
        # fmt: on

        theta_dot += dt * theta_dot_dot
        alpha_dot += dt * alpha_dot_dot
        theta += dt * theta_dot
        alpha += dt * alpha_dot

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

        return theta, alpha, theta_dot, alpha_dot

    return next_state


class QubeSwingupEnv(gym.Env):
    def __init__(self, frequency=250, **kwargs):
        self.state = [0, 0, 0, 0]
        self.frequency = frequency
        self._viewer = None
        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX)
        self.action_space = spaces.Box(-ACT_MAX, ACT_MAX)
        self.next_state = next_state_fn(1/self.frequency)

    def reward(self):
        theta = self.state[0]
        alpha = self.state[1]
        reward = 1 - (0.8 * np.abs(alpha) + 0.2 * np.abs(theta)) / np.pi
        return reward

    def isdone(self):
        theta = self.state[0]
        done = False
        done |= abs(theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        # Start the pendulum stationary at the top (stable point)
        self.state = [0, np.pi, 0, 0] + np.random.randn(4) * 0.1
        self.next_state = next_state_fn(1/self.frequency)
        return self.state

    def step(self, action):
        self.state = self.next_state(self.state, action)
        reward = self.reward()
        done = self.isdone()
        return self.state, reward, done, {}

    def render(self, rtype):
        theta = self.state[0]
        alpha = self.state[1]
        if self._viewer is None:
            self._viewer = QubeRenderer(0.0, 0.0, 250)
        self._viewer.render(theta, alpha)


def main():
    num_episodes = 1200
    num_steps = 2500  # 10 seconds if frequency is 250Hz/period is 0.004s
    env = QubeSwingupEnv()
    # rand_policy = lambda x: np.clip(np.random.randn(), -3, 3)

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(num_steps):
            action = flip_and_hold_policy(state)
            state, reward, done, info = env.step(action)
            print(state)
            env.render()
            # if done:
            #     break


if __name__ == "__main__":
    main()
