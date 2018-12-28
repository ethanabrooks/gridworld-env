import argparse
import time

import gym


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=gym.make)
    run(**vars(parser.parse_args()))


def run(env):
    env.reset()
    while True:
        env.render()
        time.sleep(1)
        s, r, t, i = env.step(env.action_space.sample())
        print('reward', r)
        if t:
            print('resetting')
            env.reset()


if __name__ == '__main__':
    cli()
