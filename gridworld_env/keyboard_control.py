import argparse
import time
import numpy as np
import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=gym.make)
    run(**vars(parser.parse_args()))


def run(env, actions=None):
    env.seed(0)
    action_map = dict()
    if actions is None:
        actions = dict(
            w=[-1, 0],
            s=[1, 0],
            a=[0, -1],
            d=[0, 1],
            x=[0, 0],
        )
    gridworld = env.unwrapped
    transitions = np.stack([t[0] for t in gridworld.transitions])
    for letter, transition in actions.items():
        idx, = np.all(transitions == transition, axis=1).nonzero()
        action_map[letter] = idx.item()

    env.reset()
    while True:
        env.render()
        s, r, t, i = env.step(action_map[input('act:')])
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
