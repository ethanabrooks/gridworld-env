import argparse
import time
import numpy as np
import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=gym.make)
    run(**vars(parser.parse_args()))


def run(env, actions=None):
    env.seed(1)
    if actions is None:
        actions = 'wsadx'

    env.reset()
    while True:
        env.render()
        unwrapped = env.unwrapped
        import ipdb; ipdb.set_trace()
        unwrapped.get_observation()
        s, r, t, i = env.step(actions.index(input('act:')))
        print('reward', r)
        if t:
            env.render()
            print('resetting')
            time.sleep(.5)
            env.reset()
            print()


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import gridworld_env
    cli()
