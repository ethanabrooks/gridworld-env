#! /usr/bin/env python
# stdlib
import sys
from collections import namedtuple
from typing import Container, Dict, Iterable, List

# third party
import numpy as np
from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv
from six import StringIO

Transition = namedtuple('Transition', 'probability new_state reward terminal')


class GridWorld(DiscreteEnv):
    transition_strings = {
        (0, 0): '🛑',
        (0, 1): '👉',
        (1, 0): '👇',
        (0, -1): '👈',
        (-1, 0): '👆'
    }

    def __init__(
            self,
            text_map: Iterable[Iterable[str]],
            terminal: Container[str],
            reward: Dict[str, float],
            transitions: List[np.ndarray] = None,
            probabilities: List[np.ndarray] = None,
            start: Iterable[str] = '',
            blocked: Container[str] = '',
    ):

        if transitions is None:
            transitions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        # because every action technically corresponds to a _list_ of transitions (to
        # permit for stochasticity, we add an additional level to the nested list
        # if necessary
        transitions = [
            t if isinstance(t[0], list) else [t] for t in transitions
        ]

        if probabilities is None:
            probabilities = [[1]] * len(transitions)
        self.actions = list(range(len(transitions)))
        assert len(transitions) == len(probabilities)
        for i in range(len(transitions)):
            assert len(transitions[i]) == len(probabilities[i])
            assert sum(probabilities[i]) == 1
        self.transitions = transitions
        self.probabilities = probabilities
        self.terminal = np.array(list(terminal))
        self.blocked = np.array(list(blocked))
        self.start = np.array(list(start))
        self.reward = reward

        self.random_states = None
        self.last_transition = None
        self._transition_matrix = None
        self._reward_matrix = None

        self.desc = text_map = np.array(
            [list(r) for r in text_map])  # type: np.ndarray

        self.original_desc = self.desc.copy()
        super().__init__(
            nS=text_map.size,
            nA=len(self.actions),
            P=self.get_transitions(desc=text_map),
            isd=self.get_isd(desc=text_map),
        )

    def assign(self, **assignments):
        new_desc = self.original_desc.copy()
        for letter, new_states in assignments.items():
            new_rows, new_cols = zip(*[self.decode(i) for i in new_states])
            new_desc[new_rows, new_cols] = letter
        self.desc = new_desc

    def get_isd(self, desc):
        isd = np.isin(desc, tuple(self.start))
        if isd.sum():
            return np.reshape(isd / isd.sum(), -1)
        return np.arange(self.desc.size)

    def get_transitions(self, desc):
        nrows, ncols = desc.shape

        def get_state_transitions():
            for i in range(nrows):
                for j in range(ncols):
                    state = self.encode(i, j)
                    yield state, dict(get_action_transitions_from(state))

        def get_action_transitions_from(state: int):
            for action in self.actions:
                yield action, list(get_transition_tuples_from(state, action))

        def get_transition_tuples_from(state, action):
            coord = self.decode(state)
            for transition, probability in zip(self.transitions[action],
                                               self.probabilities[action]):
                new_coord = np.clip(
                    np.array(coord) + transition,
                    a_min=np.zeros(2, dtype=int),
                    a_max=np.array(desc.shape, dtype=int) - 1,
                )
                new_char = self.desc[tuple(new_coord)]

                if np.all(np.isin(new_char, self.blocked)):
                    new_coord = coord
                yield Transition(
                    probability=probability,
                    new_state=self.encode(*new_coord),
                    reward=self.reward.get(new_char, 0),
                    terminal=new_char in self.terminal)

        return dict(get_state_transitions())

    def step(self, a):
        prev = self.decode(self.s)
        s, r, t, i = super().step(a)
        self.last_transition = np.array(self.decode(s)) - np.array(prev)
        return s, r, t, i

    def render(self, mode='human'):
        if self.last_transition is not None:
            transition_string = GridWorld.transition_strings[tuple(
                self.last_transition)]

            print(transition_string)

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = self.desc.copy().tolist()
        i, j = self.decode(self.s)

        out[i][j] = utils.colorize(out[i][j], 'blue', highlight=True)

        print('#' * (len(out[0]) + 2))
        for row in out:
            print('#' + "".join(row) + '#')
        print('#' * (len(out[0]) + 2))
        # No need to return anything for human
        if mode != 'human':
            return outfile
        out[i][j] = self.desc[i, j]

    def encode(self, i, j):
        return np.ravel_multi_index((i, j), self.desc.shape)

    def decode(self, s):
        return np.unravel_index(s, self.desc.shape)

    def generate_matrices(self):
        self._transition_matrix = np.zeros((self.nS, self.nA, self.nS))
        self._reward_matrix = np.zeros((self.nS, self.nA, self.nS))
        for s1, action_P in self.P.items():
            for a, transitions in action_P.items():
                trans: Transition
                for trans in transitions:
                    self._transition_matrix[s1, a, trans.
                                            new_state] = trans.probability
                    self._reward_matrix[s1, a] = trans.reward
                    if trans.terminal:
                        for a in range(self.nA):
                            self._transition_matrix[trans.new_state, a, trans.
                                                    new_state] = 1
                            self._reward_matrix[trans.new_state, a] = 0
                            assert not np.any(self._transition_matrix > 1)

    @property
    def transition_matrix(self) -> np.ndarray:
        if self._transition_matrix is None:
            self.generate_matrices()
        return self._transition_matrix

    @property
    def reward_matrix(self) -> np.ndarray:
        if self._reward_matrix is None:
            self.generate_matrices()
        return self._reward_matrix


if __name__ == '__main__':
    import gym
    from gridworld_env.random_walk import run
    run(gym.make('BookGridGridWorld-v0'))
