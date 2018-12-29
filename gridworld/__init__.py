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


def register_from_path(path: Path):
    with path.open() as f:
        id = ''.join([word.capitalize() for word in path.stem.split('-')])
        register_from_string(id + 'GridWorld-v0', f.read())


for json_file in Path(__file__).parent.joinpath('json').iterdir():
    register_from_path(json_file)
