from gridworld import get_id, get_paths


def cli():
    for path in get_paths():
        print(get_id(path))
