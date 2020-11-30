from __future__ import division, print_function

import argparse
import codecs
import datetime
import glob
import json
import logging
import math
import os
import random
# System level imports
import sys
import time
from collections import deque
from functools import partial

import carla
import pandas as pd
import scipy
import torch
import tqdm

from controller import controller_eval as controller2d
from controller.cost_function import CostFunction
from custom_classes import World, Logger, GraphNetPredictor
from custom_classes import get_behavior_pairs, generate_start_pos_combinations, get_current_pose
from mtp.argument import fetch_argument
from mtp.config import get_config_list
from mtp.data.data_loader import get_trajectory_data_loader
from mtp.networks import fetch_model_iterator
from mtp.train import Trainer
from scenario_configs import SCENARIOS

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
TARGET_THRESHOLD = 10.0
# lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES = 0.5  # distance between interpolated points


def get_world(client, args, scene_id, start_spawn_ids, behavior_pairs):
    world = World(client.get_world(), args.filter, scene_id, start_spawn_ids, behavior_pairs)
    return world


def initialize_client(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(4.0)
    return client


def initialize_world(args, client, scene_id, start_spawn_ids, behavior_pairs):
    return get_world(client, args, scene_id, start_spawn_ids, behavior_pairs)


def generate_logs_directory(logs_path):
    num_agents = [2, 3, 4]
    df = pd.DataFrame(columns=["idx", "collision", "time"])
    for n in num_agents:
        agents_path = os.path.join(logs_path, "agents_{}".format(n))
        os.makedirs(agents_path)
        for scene_id in SCENARIOS[n]:
            csv_path = os.path.join(
                agents_path, "agents_{}_scenario_{}".format(n, scene_id))
            df.to_csv(csv_path)


def interpolate_waypoints(waypoints_np, wp_distance):
    # Linearly interpolate between waypoints and store in a list
    wp_interp = []    # interpolated values
    # (rows = waypoints, columns = [x, y, v])
    wp_interp_hash = []    # hash table which indexes waypoints_np
    # to the index of the waypoint in wp_interp
    interp_counter = 0     # counter for current interpolated point index
    for i in range(waypoints_np.shape[0] - 1):
        # Add original waypoint to interpolated waypoints list (and append
        # it to the hash table)
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        # Interpolate to the next waypoint. First compute the number of
        # points to interpolate based on the desired resolution and
        # incrementally add interpolated points until the next waypoint
        # is about to be reached.
        num_pts_to_interp = int(np.floor(wp_distance[i] /
                                         float(INTERP_DISTANCE_RES)) - 1)
        wp_vector = waypoints_np[i+1] - waypoints_np[i]
        wp_uvector = wp_vector / np.linalg.norm(wp_vector)
        for j in range(num_pts_to_interp):
            next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
            wp_interp.append(list(waypoints_np[i] + next_wp_vector))
            interp_counter += 1
    # add last waypoint at the end
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)
    interp_counter += 1
    return wp_interp, wp_interp_hash


all_traj_data = torch.load('individual_trajs.pth')


def instantiate_agent_data(world, past_hist_size, intention):
    world.player.reference_traj = np.array(all_traj_data[intention])
    world.player.reference_traj[:, 1] = -1 * world.player.reference_traj[:, 1]
    # plt.plot(world.player.reference_traj[:, 0], world.player.reference_traj[:, 1])
    # plt.show()
    waypoints_np = world.player.reference_traj
    wp_distance = []   # distance array
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(
            np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                    (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
    wp_distance.append(0)  # last distance is 0 because it is the distance
    # from the last waypoint to the last waypoint
    wp_interp, wp_interp_hash = interpolate_waypoints(
        waypoints_np, wp_distance)
    waypoints_np = waypoints_np[:, [0, 1, 3]]
    controller = controller2d.Controller2D(waypoints_np, world.player.carla_agent)
    controller.set_path(wp_interp)
    world.player.set_controller(controller)
    world.player.set_agent_info(waypoints_np, wp_interp, wp_interp_hash,
                          wp_distance, deque(maxlen=past_hist_size))
    return waypoints_np

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================


class Waypoint:
    def __init__(self, transform, speed):
        self.transform = transform
        self.target_speed = speed


def run_experiments_for_n_agents(args, trainer, num_agents, logs_path):
    world = None
    if args.no_rendering:
        no_rendering = True
    else:
        no_rendering = False

    T = 15
    PREDICTION_STEP = 1
    TOTAL_TRIALS = args.num_exp
    rollout_size = 25
    num_min_waypoints = 5
    center_coordinate = [31., -30.]
    method = 'mfp'
    predictor = GraphNetPredictor(trainer, u_dim=4, B=1, N=num_agents, T=T, rollout_size=25,
                                  d=4)

    exp_start = time.time()
    scenarios = json.loads(codecs.open(args.scenarios, 'r', encoding='utf-8').read())['split']
    exp_start = time.time()
    config_file = args.scenarios
    config_name = config_file.split('.')[0]
    exp_configs = json.loads(codecs.open(config_file, 'r', encoding='utf-8').read())
    scenarios = exp_configs['split']
    behavior_pairs = get_behavior_pairs(num_agents)
    behavior_cases = exp_configs['behavior_id']
    num_trials_per_behavior = TOTAL_TRIALS // len(behavior_cases)
    data_logger = Logger(num_agents, logs_path, scenarios)
    intersection_center = [257.0, -248.0]

    client = initialize_client(args)
    all_trajs_data = dict()
    for num_scene, scene_id in enumerate(scenarios):
        scene_steps = 0
        csv_log_path = os.path.join(
            logs_path, "agents_{}_scenario_{}_{}.csv".format(num_agents, scene_id, config_name))

        logs = []
        for b_idx, behavior_id in enumerate(behavior_cases):
            behavior_pair = behavior_pairs[behavior_id]
            start_spawn_ids = generate_start_pos_combinations(scene_id)
            for spawn_id in start_spawn_ids:
                print("SCENE ID: ", scene_id, " BEHAVIOR: ", behavior_pair, " SPAWN_ID: ", spawn_id)
                for param_id in tqdm.tqdm(range(num_trials_per_behavior)):
                    num_runs = 0
                    ego_end_time = 0.
                    ego_target_reached = False
                    collision = False
                    time_complete = False
                    data = []
                    predictor.set_src_dst_tensor(scene_id)
                    start_predictor = False
                    debug_log_gt = []
                    debug_log_pred = []
                    mpc_start = False
                    # data_logger.reset_episode()
                    try:
                        world = initialize_world(args, client, scene_id, spawn_id, behavior_pair)
                        world.get_intersection_distance(intersection_center)
                        tot_target_reached = 0
                        print("Rollout Num: ", num_runs)
                        settings = world.world.get_settings()
                        settings.synchronous_mode = True  # Enables synchronous mode
                        settings.fixed_delta_seconds = 0.1
                        settings.no_rendering_mode = no_rendering
                        world.world.apply_settings(settings)

                        frame = 0
                        steps = 0
                        world_snapshot = world.world.get_snapshot()
                        prev_timestamp = 0.0
                        world.start_time = world_snapshot.timestamp.elapsed_seconds
                        steps = 0
                        for player in world.agents:
                            player_snapshot = world_snapshot.find(
                                player.carla_agent.id)
                            player.curr_wp = get_current_pose(player_snapshot)
                        if scene_id[0] == "0" or scene_id[0] == "2":
                            dist_from_center = math.fabs(world.player.curr_wp[0] - intersection_center[0])
                        else:
                            dist_from_center = math.fabs(world.player.curr_wp[1] - intersection_center[1])

                        world_snapshot = world.world.get_snapshot()
                        for player in world._agents:
                            player_snapshot = world_snapshot.find(
                                player.carla_agent.id)
                            player.curr_wp = get_current_pose(player_snapshot)
                            player.prev_wp = get_current_pose(player_snapshot)

                        ego_waypoints = instantiate_agent_data(world, 15, scene_id[:2])
                        pred_traj = None
                        world.update_behavior_params()
                        # plt.plot(world.player.reference_traj[:, 0], -world.player.reference_traj[:, 1])
                        cost_function = CostFunction(rollout_size)
                        world.player.controller.set_path(world.player.reference_traj)
                        while not time_complete:
                            steps += 1
                            frame += 1
                            if steps > 300:
                                time_complete = True

                            world.world.tick()
                            world.tick(frame, method)
                            if scene_id[0] == "0" or scene_id[0] == "2":
                                dist_from_center = math.fabs(world.player.curr_wp[0] - intersection_center[0])
                            else:
                                dist_from_center = math.fabs(world.player.curr_wp[1] - intersection_center[1])

                            collision = world.check_collision(world_snapshot.timestamp)
                            if collision and (not world.player.target_reached):
                                print("Collision detected!")
                                break

                            restart = True
                            for player in world.agents:
                                if not player.target_reached:
                                    restart = False

                            if restart:
                                time_complete = True

                            world.convert_agents_to_box()
                            dist = scipy.spatial.distance.cdist([intersection_center], [[world.player.curr_wp[0], world.player.curr_wp[1]]])

                            if world.player.curr_wp[1] < -270.:
                                world.player.target_reached = True

                            if dist < 25:
                                start_predictor = True
                            else:
                                start_predictor = False

                            world.wp_traversed.append(world.player.curr_wp[:2])
                            world_snapshot = world.world.get_snapshot()
                            game_timestamp = world_snapshot.timestamp.elapsed_seconds

                            current_timestamp = float(game_timestamp) #/ 1000.0
                            dt = current_timestamp - prev_timestamp

                            debug_log_gt.append(world.agents[1].curr_wp + [steps])
                            # plt.plot([world.agents[1].curr_wp[0]], [world.agents[1].curr_wp[1]], [steps * dt], "ro")

                            np_controls = []
                            for player, player_ctrl in zip(world.agents, world.controllers):
                                if player.target_reached == True:
                                    continue

                                if player.carla_agent.id == world.player.carla_agent.id:
                                    if start_predictor and steps % PREDICTION_STEP == 0 and world.player.target_reached:
                                        world.player.update_waypoints()
                                        world.player.controller.update_waypoints(world.player.new_waypoints)
                                        # world.player.controller.update_waypoints(world.player.new_waypoints)
                                        world.player.controller.update_values(*world.player.curr_wp, world.player.get_speed(), current_timestamp, frame)
                                        world.player.controller.update_controls_2(world)
                                        cmd_throttle, cmd_steer, cmd_brake = player.controller.get_commands()
                                        player.control = carla.VehicleControl(throttle=cmd_throttle.item(), steer=cmd_steer.item(), brake=cmd_brake)
                                        player_ctrl.update_information(world)
                                        speed_limit = world.player.carla_agent.get_speed_limit()
                                        player_ctrl.get_local_planner().set_speed(speed_limit)
                                        control = player_ctrl.run_step()
                                        continue

                                dist = player_ctrl.vehicle.get_location().distance(
                                    player_ctrl.end_waypoint.transform.location)
                                if dist < TARGET_THRESHOLD:
                                    player.target_reached = True
                                    player.control = carla.VehicleControl(throttle=0., steer=0., brake=1.0)

                                    print("Target accomplished")
                                    continue
                                    # world.init()
                                    tot_target_reached += 1

                                player_ctrl.update_information(world)
                                speed_limit = world.player.carla_agent.get_speed_limit()
                                # print ("Speed Lim: ", speed_limit)
                                player_ctrl.get_local_planner().set_speed(speed_limit)

                                control = player_ctrl.run_step()
                                player.control = control
                                # player.control = carla.VehicleControl(throttle=0., steer=0., brake=1.0)

                            for i, player in enumerate(world.agents):
                                player.carla_agent.apply_control(player.control)
                            # world.player.carla_agent.apply_control(player.control)
                            # data_logger.step(world)
                            if world.player.target_reached and not ego_target_reached:
                                ego_end_time = world.world.get_snapshot().timestamp.elapsed_seconds
                                ego_target_reached = True
                            prev_timestamp = current_timestamp
                            data.append(world.player.curr_wp + [world.player.get_speed()])
                    finally:
                        if not collision:
                            scene_steps += 1
                            # data_logger.save(scene_steps)
                        else:
                            if world.player.target_reached:
                                collision = 0.
                            a = 1
                            # data_logger.remove_collision_data()

                        num_runs += 1
                        all_trajs_data[scene_id] = data
                        predictor.stop()
                        torch.save(debug_log_gt, f'debug_gt_scene_{scene_id}_b{behavior_id}.pth')
                        torch.save(debug_log_pred, f'debug_pred_scene_{scene_id}_b{behavior_id}.pth')

                        end_time = world.world.get_snapshot().timestamp.elapsed_seconds
                        if end_time is None:
                            end_time = 0
                        ego_time_taken = ego_end_time - world.start_time
                        time_taken = end_time - world.start_time
                        time_taken = end_time - world.start_time
                        wp_traversed = np.array(world.wp_traversed)
                        trajectory_length = 0.
                        for m in range(wp_traversed.shape[0] - 1):
                            trajectory_length += np.linalg.norm(
                                [wp_traversed[m+1][0] - wp_traversed[m][0], wp_traversed[m+1][1] - wp_traversed[m][1]])
                        logs.append({'collision': int(
                            collision), 'time': time_taken, 'ego_time': ego_time_taken, 'distance': trajectory_length})
                        df = pd.DataFrame(
                            logs, columns=['collision', 'time', 'distance', 'ego_time'])
                        df.to_csv(csv_log_path, encoding='utf-8', index=False)
                        settings = world.world.get_settings()
                        settings.synchronous_mode = False
                        settings.no_rendering_mode = no_rendering
                        settings.fixed_delta_seconds = 0.1
                        world.world.apply_settings(settings)
                        if world is not None:
                            world.world.tick()
                            world.destroy()

    # torch.save(all_trajs_data, "individual_trajs.pth")
    print("TIME TAKEN: ", time.time() - exp_start)
    settings = world.world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = no_rendering
    settings.fixed_delta_seconds = 0.1
    world.world.apply_settings(settings)
    world.destroy()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='no rendering for server')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-s', '--scenarios',
        metavar='S',
        default=2000,
        type=str,
        help='scenarios to collect for')
    argparser.add_argument(
        '-n', '--num-agents',
        default=2,
        type=int,
        help='Number of agents in a scenario')
    argparser.add_argument(
        '--num-exp',
        default=20,
        type=int,
        help='Total number of experiments')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Roaming", "Basic"],
                           help="select which agent to run",
                           default="Basic")
    argparser.add_argument('--num_agent', type=int, default=2)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--beta', type=float, default=0.5)
    argparser.add_argument(
        '--use_winding', action='store_true', dest='use_winding')
    argparser.add_argument('--gpu_index', type=int, default=0)
    args = argparser.parse_args()

    def test_arg_modifier(args, user_args):
        args.CUDA_VISIBLE_DEVICES = -1
        args.OMP_NUM_THREADS = 1
        args.num_agent = user_args.num_agent
        args.seed = user_args.seed
        # args.model_type = user_args.model_type
        args.beta = user_args.beta
        if user_args.num_agent == 4:
            args.bsize = 20
            args.num_history = 5
            args.num_rollout = 15
        return args

    args_pred = fetch_argument(partial(test_arg_modifier, user_args=args))
    args_pred.bsize = 1
    dc, lc, tc, model_dir = get_config_list(args_pred)
    modes = ['test']
    dataloader = {'test': get_trajectory_data_loader(
        dc,
        test=True,
        batch_size=args_pred.bsize,
        num_workers=args_pred.num_workers,
        shuffle=True)}
    run_every = {'test': 1}
    gn_wrapper = fetch_model_iterator(lc, args_pred)
    trainer = Trainer(gn_wrapper, modes, dataloader, run_every, tc)
    trainer.model_wrapper.eval_mode()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    base_path = "./logs_vmpc"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    logs_path = os.path.join(
        base_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_path)

    # methods = ["vanilla_mpc", "ours"]
    # generate_logs_directory(logs_path)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        # trainer = None
        run_experiments_for_n_agents(args, trainer, args.num_agents, logs_path)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
