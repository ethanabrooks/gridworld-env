import argparse
import time

import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=gym.make)
    run(**vars(parser.parse_args()))


def run(env, actions=None):
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
    transitions = [t[0] for t in gridworld.transitions]
    for letter, transition in actions.items():
        try:
            action_map[letter] = transitions.index(transition)
        except ValueError:
            pass

    env.reset()
    while True:
        env.render()
        s, r, t, i = env.step(action_map[input('act:')])
        print(s)
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
