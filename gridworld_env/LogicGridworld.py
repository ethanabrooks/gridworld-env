from copy import deepcopy

from gridworld_env import RandomGridWorld
import numpy as np

from gridworld_env.gridworld import Transition


class LogicGridWorld(RandomGridWorld):
    def __init__(self, objects, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objects = objects
        self.flattened_objects = [o for l in objects for o in l]
        self.object_type = None
        self.task_color = None
        self.task_objects = None
        self.task_types = ['move', 'touch']
        self.task_type_idx = None
        self.task_type = None
        self.target_color = None
        self.touched = set()

        self.transitions = [[-1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 0, -1]]

    def get_transitions(self, desc):
        nrows, ncols = desc.shape

        def get_state_transitions():
            for i in range(nrows):
                for j in range(ncols):
                    for k in range(2):
                        state = self.encode(i, j, k)
                        yield state, dict(get_action_transitions_from(state))

        def get_action_transitions_from(state: int):
            for action in self.actions + [len(self.actions)]:
                yield action, list(get_transition_tuples_from(state, action))

        def get_transition_tuples_from(state, action):
            coord = self.decode(state)
            for transition, probability in zip(self.transitions[action],
                                               self.probabilities[action]):
                new_coord = np.clip(
                    np.array(coord) + transition,
                    a_min=np.zeros(3, dtype=int),
                    a_max=np.array([nrows, ncols, 2], dtype=int) - 1,
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
        P = deepcopy(super().get_transitions(desc))
        a = len(self.actions)  # last idx for interact
        for s1 in P:
            i, j, k = self.decode(s1)
            assert k == 0
            self.P[s1][a] = [Transition(
                probability=1,
                new_state=self.encode(i, j, 1),  # interact toggles third dim
                reward=0,
                terminal=False,
            )]
            s2 = self.encode(i, j, 1)
            self.P[s2] = {a: [Transition(
                probability=1,
                new_state=self.encode(i, j, 0),  # interact toggles third dim
                reward=0,
                terminal=False,
            )]}

    def encode(self, i, j, k):
        raise NotImplementedError

    def decode(self, s):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def step(self, a):
        s, r, t, i = super().step(a)  # TODO: move action
        maybe_object = self.desc[self.decode(s)]
        if maybe_object in self.flattened_objects:
            self.touched.add(maybe_object)
        if not t:
            success = self.check_success()
            r = float(success)
            t = bool(success)
        # TODO: modify observation
        return s, r, t, i

    def get_colors_for(self, objects):
        contains_object = objects.reshape(1, 1, -1) == np.expand_dims(self.desc, 2)
        return np.broadcast_to(self.original_desc, contains_object.shape)[contains_object]

    def reset(self):
        o = super().reset()
        # task objects
        self.task_color = np.random.choice(self.original_desc)
        self.object_type = np.random.choice(len(self.objects))
        objects = self.objects[self.object_type]
        colors = self.get_colors_for(objects)
        self.task_objects = objects[colors == self.task_color]
        self.task_objects.sort()

        # task type
        self.task_type_idx = np.random.choice(len(self.task_types))
        self.task_type = self.task_types[self.task_type_idx]
        self.target_color = np.random.choice(self.original_desc)
        self.touched = []
        return o

    def check_success(self):
        if self.task_type == 'move':
            return np.all(self.get_colors_for(self.task_objects) == self.target_color)
        if self.task_type == 'touch':
            return np.all(np.isin(self.task_objects, self.touched))
        raise RuntimeError
