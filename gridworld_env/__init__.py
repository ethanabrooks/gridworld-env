import json
from pathlib import Path

from gridworld_env.random_gridworld import RandomGridWorld
from gym.envs import register

from gridworld_env.gridworld import GridWorld

SUFFIX = 'GridWorld-v0'
JSON_PATH = Path(__file__).parent.joinpath('json')


def register_from_string(env_id, **kwargs):
    if 'random' in kwargs:
        class_ = RandomGridWorld
    else:
        class_ = GridWorld

    register(
        id=env_id,
        entry_point=f'{class_.__module__}:{class_.__name__}',
        trials=kwargs.pop('trials', 1),
        reward_threshold=kwargs.pop('reward_threshold', None),
        max_episode_steps=kwargs.pop('max_episode_steps', None),
        max_episode_seconds=kwargs.pop('max_episode_seconds', None),
        local_only=False,
        nondeterministic=False,
        kwargs=kwargs,
    )


def get_args(env_id):
    path = Path(JSON_PATH, env_id.rstrip(SUFFIX)).with_suffix('.json')
    with path.open('rb') as f:
        return json.load(f)


def register_envs():
    for path in JSON_PATH.iterdir():
        with path.open() as f:
            register_from_string(f'{path.stem}{SUFFIX}', **json.load(f))

register_envs()