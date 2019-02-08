#! /usr/bin/env python
import json
import time
from pathlib import Path
from typing import Dict

import gym
import numpy as np
from gridworld_env.gridworld import GridWorld
from gridworld_env.random_walk import run
from gym import spaces


class RandomGridWorld(GridWorld):
    def __init__(
            self,
            random: Dict[str, int] = None,
            *args, **kwargs
    ):
        self.random = random
        self.potential_new = None
        super().__init__(*args, **kwargs)
        self.potential_new = np.ravel_multi_index(
            np.where(
                np.logical_not(
                    np.logical_or(
                        np.isin(self.desc, self.blocked),
                        np.isin(self.desc, self.start)))),
            dims=self.desc.shape)
        self.observation_space = spaces.Tuple(
            [self.observation_space] * (1 + len(random))
        )

    def append_randoms(self, state):
        return (state,) + tuple(self.random_states)

    def set_randoms(self):
        n_choices = sum(self.random.values())
        choices = np.random.choice(
            self.potential_new, size=n_choices, replace=False)
        *self.random_states, _ = np.split(choices,
                                          np.cumsum(list(self.random.values())))

        self.assign(**dict(zip(self.random.keys(), self.random_states)))
        self.set_desc(self.desc)

    def reset(self):
        if self.potential_new is None:
            self.random_states = (None,) * len(self.random)
        else:
            self.set_randoms()
        self.last_transition = None
        return self.append_randoms(super().reset())

    def assign(self, **assignments):
        new_desc = self.original_desc.copy()
        for letter, new_states in assignments.items():
            new_rows, new_cols = zip(*[self.decode(i) for i in new_states])
            new_desc[new_rows, new_cols] = letter
        self.desc = new_desc

    def set_desc(self, desc):
        self.P = self.get_transitions(desc)
        self.isd = self.get_isd(desc)
        self.last_transition = None  # for rendering
        self.nS = desc.size

    def step(self, a):
        s, r, t, i = super().step(a)
        return self.append_randoms(s), r, t, i


if __name__ == '__main__':
    run(gym.make('1x3RandomGridWorld-v0'))
    # env.reset()
    # while True:
    #     s, r, t, i = env.step(env.action_space.sample())
    #     env.render()
    #     print('reward', r)
    #     time.sleep(.5)
    #     if t:
    #         print('reset')
    #         time.sleep(1)
    #         env.reset()
