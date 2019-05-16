import numpy as np

from gridworld_env import RandomGridWorld


class LogicGridWorld(RandomGridWorld):
    def __init__(self, objects, text_map, *args, **kwargs):
        transitions = [[-1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, -1, 0],
                       [0, 0, 1],
                       [0, 0, -1]]

        super().__init__(*args, **kwargs,
                         text_map=np.dstack([np.zeros_like(text_map), text_map]),
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

        colors = np.unique(self.desc[1])
        self.observation_tensor = np.expand_dims(self.desc[1], 2) == colors.reshape((1, 1, -1))

    def transition_strings(self):
        return {
            (0, 0, 0):  '🛑',
            (0, 1, 0):  '👉',
            (1, 0, 0):  '👇',
            (0, -1, 0): '👈',
            (-1, 0, 0): '👆',
            (0, 0, 1):  '👊',
            (0, 0, -1): '✋',
        }

    def render_map(self, mode='human'):
        if self.last_transition is not None:
            transition_string = LogicGridWorld.transition_strings[tuple(
                self.last_transition)]

            print(transition_string)
        if self.last_reward is not None:
            print('Reward:', self.last_reward)

        colors = dict(
            r='\e[41m',
            g='\e[49m',
            b='\e[44m',
            y='\e[43m'
        )

        def get_string():
            nrows, ncols = self.desc[1].shape
            last_val = None
            for i in range(nrows):
                for j in range(ncols):
                    val = self.desc[1][i, j]
                    if val != last_val:
                        yield colors[val]
                        last_val = val
                    yield self.desc[0][i, j]
                yield '\n'

        print(''.join(get_string()))

    def step(self, a):
        prev = self.decode(self.s)
        s, r, t, i = super().step(a)
        maybe_object = self.desc[self.decode(s)]
        if maybe_object in self.objects.flatten():
            self.touched.add(maybe_object)
        if not t:
            success = self.check_success()
            r = float(success)
            t = bool(success)

        self.last_action = a
        self.last_reward = r
        self.last_transition = np.array(self.decode(s)) - np.array(prev)
        return self.get_observation(s), r, t, i

    def get_observation(self, s):
        things = np.append(np.unique(self.desc), s)
        things_tensor = np.expand_dims(self.desc, 3) == things.reshape((1, 1, 1, -1))
        import ipdb;
        ipdb.set_trace()
        return np.dstack([things_tensor, self.observation_tensor])

    def get_colors_for(self, objects):
        contains_object = objects.reshape(1, 1, -1) == self.desc
        import ipdb;
        ipdb.set_trace()
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
