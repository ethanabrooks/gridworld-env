from pathlib import Path
import json

from gym.envs import register
from gridworld.gridworld import GridWorld


def register_from_string(id, string):
    obj = json.loads(string)
    register(
        id=id,
        entry_point=f'{GridWorld.__module__}:{GridWorld.__name__}',
        trials=obj.pop('trials', 1),
        reward_threshold=obj.pop('reward_threshold', None),
        max_episode_steps=obj.pop('max_episode_steps', None),
        max_episode_seconds=obj.pop('max_episode_seconds', None),
        local_only=False,
        nondeterministic=False,
        kwargs=obj,
    )


def get_id(path: Path):
    return ''.join([word.capitalize() for word in path.stem.split('-')]) \
           + 'GridWorld-v0'


def get_paths():
    yield from Path(__file__).parent.joinpath('json').iterdir()


for path in get_paths():
    with path.open() as f:
        register_from_string(get_id(path), f.read())
