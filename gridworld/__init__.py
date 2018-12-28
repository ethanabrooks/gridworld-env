from pathlib import Path
import json

from gym.envs import register
from gridworld.gridworld import Gridworld, parse_json


def register_from_string(id, string):
    obj = json.loads(string)
    register(
        id=id,
        entry_point=f'{Gridworld.__module__}:{Gridworld.__name__}',
        trials=obj.pop('trials', 1),
        reward_threshold=obj.pop('reward_threshold', None),
        local_only=False,
        nondeterministic=False,
        kwargs=obj,
    )


def register_from_path(path: Path):
    with path.open() as f:
        register_from_string(path.name, f.read())


for json_file in Path('json').iterdir():
    register_from_path(json_file)
