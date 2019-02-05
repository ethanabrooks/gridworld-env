from gridworld import SUFFIX, JSON_PATH


def cli():
    for path in JSON_PATH.iterdir():
        print(f'{path.stem}{SUFFIX}')
