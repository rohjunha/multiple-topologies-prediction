import argparse
import glob
import os

import numpy as np
import pandas as pd


def generate_stats(args):
    logs_dir = os.path.join(args.base_path, args.method + '/' + str(args.num_agent) + 'agent/' + args.type)
    logs_path = glob.glob(logs_dir + "/*.csv")[-1]
    print (logs_path)
    logs = pd.read_csv(logs_path)

    collisions = logs['collision']
    time_taken = logs['time']
    ego_time_taken = logs['ego_time']

    coll_mean, coll_std = np.mean(collisions), np.std(collisions)
    total_t = 0.
    ego_total_t = 0.

    t_s = []
    ego_t_s = []
    for i, col in enumerate(collisions):
        if col == 0:
            t_s.append(time_taken[i])
            ego_t_s.append(ego_time_taken[i])
        else:
            t_s.append(30.0)
            ego_t_s.append(30.0)

    time_mean = np.mean(t_s)
    time_std = np.std(t_s)

    ego_time_mean = np.mean(ego_t_s)
    ego_time_std = np.std(ego_t_s)

    print ("Collision: ", coll_mean * 100)
    print ("Time taken: ", time_mean, " +/- ", time_std)
    print ("Ego Time taken: ", ego_time_mean, " +/- ", ego_time_std)


def main():
    argparser = argparse.ArgumentParser(description='Generate mean/percentage numbers from experiment logs')
    argparser.add_argument(
        '--base-path',
        help='base path of logs')
    argparser.add_argument(
        '--type', type=str,
        choices=["easy", "hard"],
        help="select case for scenario",
        default="easy")
    argparser.add_argument('--num-agent', type=int, default=2)
    argparser.add_argument('--method', type=str,
                           choices=["autopilot", "mfp", "gn"],
                           default="autopilot")
    args = argparser.parse_args()
    generate_stats(args)


if __name__ == '__main__':
    main()
