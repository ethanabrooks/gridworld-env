from pathlib import Path
import json

from gym.envs import register
from gridworld.gridworld import GridWorld

SUFFIX = 'GridWorld-v0'
JSON_PATH = Path(__file__).parent.joinpath('json')


def register_from_string(env_id, **kwargs):
    register(
        id=env_id,
        entry_point=f'{GridWorld.__module__}:{GridWorld.__name__}',
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


def get_id(path: Path):
    return ''.join([word.capitalize() for word in path.stem.split('-')]) \
           + 'GridWorld-v0'


for path in JSON_PATH.iterdir():
    with path.open() as f:
        register_from_string(get_id(path), **json.load(f))
