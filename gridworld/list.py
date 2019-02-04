from gridworld import get_id, JSON_PATH


def cli():
    for path in JSON_PATH.iterdir():
        print(get_id(path))
