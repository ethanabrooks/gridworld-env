import numpy as np
import six
from gym.utils import colorize
from gym.utils.colorize import color2num

from gridworld_env import RandomGridWorld


class LogicGridWorld(RandomGridWorld):
    def __init__(self, objects, text_map, random=None, *args, **kwargs):
        transitions = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, -1],
                [0, 0, 1],
            ])

        if random is None:
            random = dict()
        self.colors_map = np.array(
            [list(r) for r in text_map])  # type: np.ndarray
        blank = ' '
        self.state_char = '🚡'
        text_map = np.full_like(self.colors_map, blank)
        text_map = np.stack([text_map, text_map], axis=2)
        super().__init__(*args, **kwargs,
                         random=random,
                         start=blank,
                         reward=dict(),
                         text_map=text_map,
                         transitions=transitions,
                         terminal=[], )

        self.objects = np.array(objects)
        self.task_types = ['move', 'touch']
        self.object_type = None
        self.task_color = None
        self.task_objects = None
        self.task_type_idx = None
        self.task_type = None
        self.target_color = None
        self.touched = set()

        self.colors = np.unique(self.colors_map)
        self.colors_observation = np.expand_dims(self.colors_map, 2) == \
                                  self.colors.reshape((1, 1, -1))
        self.unique = np.append(self.objects.flatten(), self.state_char)

    @property
    def objects_level(self):
        return self.desc[:, :, 0]

    @property
    def transition_strings(self):
        return np.array(list('🛑👇👆👉👈✋👊'))

    def render_map(self, mode='human'):
        print('touched:', self.touched)
        colors = dict(r='red', g='green', b='blue', y='yellow')
        desc = self.desc.copy()
        i, j, k = tuple(self.decode(self.s))
        desc[i, j, 0] = '*'
        levels = zip(self.colors_map, desc[:, :, 0], desc[:, :, 1])
        for row0, row1, row2 in levels:
            print(six.u('\x1b[30m'), end='')
            last = None
            for color, s1, s2 in zip(row0, row1, row2):
                if color != last:
                    color = colors[color]
                    num = color2num[color] + 10
                    highlight = six.u(str(num))
                    print(six.u(f'\x1b[{highlight}m'), end='')
                print(s1, end='')  # TODO
                last = color
            print(six.u('\x1b[0m'))
        print(six.u('\x1b[39m'), end='')

    def step(self, a):
        s, r, t, i = super().step(a)
        i, j, k = self.decode(s)
        maybe_object = self.desc[i, j]
        touching = np.isin(maybe_object, self.objects.flatten())
        if np.any(touching):
            self.touched.add(maybe_object[touching].item())
        if not t:
            success = self.check_success()
            r = float(success)
            t = bool(success)
        return self.get_observation(s), r, t, i

    def get_observation(self, s):
        desc = self.desc.copy()
        desc[tuple(self.decode(s))] = self.state_char
        one_hots = np.expand_dims(desc, 3) == self.unique.reshape((1, 1, 1, -1))
        h, w, _ = desc.shape
        objects_observation = one_hots.reshape(h, w, -1).astype(int)
        return np.dstack([objects_observation, self.colors_observation])

    def get_colors_for(self, objects):
        return self.colors_map[np.isin(self.objects_level, objects)]

    def reset(self):
        o = super().reset()
        # randomize objects
        h, w, _ = self.desc.shape
        random_coords = [
            self.np_random.randint(low=0, high=h, size=self.objects.size)
            for h in [h, w, 1]
        ]
        random_states = [self.encode(*c) for c in zip(*random_coords)]
        self.assign(**{o: [s] for o, s in zip(self.objects.flatten(), random_states)})

        # task objects
        self.task_color = self.np_random.choice(self.colors)
        self.object_type = self.np_random.choice(len(self.objects))
        objects = self.objects[self.object_type]
        object_colors = self.get_colors_for(objects)
        self.task_objects = objects[object_colors == self.task_color]
        self.task_objects.sort()

        # task type
        self.task_type_idx = self.np_random.choice(len(self.task_types))
        self.task_type = self.task_types[self.task_type_idx]
        self.target_color = self.np_random.choice(self.colors)
        self.touched = []
        return o

    def check_success(self):
        if self.task_type == 'move':
            return np.all(self.get_colors_for(self.task_objects) == self.target_color)
        if self.task_type == 'touch':
            return np.all(np.isin(self.task_objects, self.touched))
        raise RuntimeError


if __name__ == '__main__':
    import gym
    from gridworld_env.keyboard_control import run

    env = gym.make('10x10FourSquareGridWorld-v0')
    actions = dict(w=[-1, 0, 0],
                   s=[1, 0, 0],
                   a=[0, -1, 0],
                   d=[0, 1, 0],
                   q=[0, 0, -1],
                   e=[0, 0, 1], )
    run(env, actions=actions)
