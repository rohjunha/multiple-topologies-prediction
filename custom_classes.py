import collections
import math
import os
import random
import re
import sys
import weakref
from collections import deque
from itertools import product, permutations
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any

import carla
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import pandas as pd
import torch

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla_spawn_points import START_POS_DICT, END_POS_DICT

# logger = get_logger(__name__)
random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)

INTERP_LOOKAHEAD_DISTANCE = 20.0
k = 0.05  # look forward gain
Lfc = 4.25  # [m] look-ahead distance
Kp = 1.25  # speed proportional gain
Ki = 0.5
Kd = 0.8
dt = 0.1  # [s] time tick

LENGTH = 4.7  # [m]
WIDTH = 1.85  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.87528  # [m]


def onehot_from_index(index_vector: torch.Tensor, num_cls: int):
    onehot = torch.zeros((index_vector.numel(), num_cls), dtype=torch.float32)
    for i in range(index_vector.numel()):
        onehot[i, index_vector[i]] = 1.
    return onehot


def fetch_winding_constraints(num_agent: int):
    edge_index = list(permutations(range(num_agent), 2))
    sorted_edge_index = np.array([sorted(p) for p in edge_index])
    unique_edge_index = np.unique(sorted_edge_index, axis=0)
    constraints = [tuple(int(v) for v in np.where((sorted_edge_index == unique_edge_index[r]).all(axis=1))[0])
                   for r in range(unique_edge_index.shape[0])]
    return constraints


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    transform = measurement.get_transform()
    x = transform.location.x
    y = transform.location.y
    yaw = math.radians(transform.rotation.yaw)

    return [x, y, yaw]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def convert_to_box(transform):
    pts = []
    x, y = transform.location.x, transform.location.y
    yaw = transform.rotation.yaw
    xrad = 4.85 / 2.
    yrad = 1.9 / 2.
    cpos = np.array([
        [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
        [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
    rotyaw = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])
    cpos_shift = np.array([
        [x, x, x, x, x, x, x, x],
        [y, y, y, y, y, y, y, y]])
    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

    for j in range(cpos.shape[1]):
        pts.append([cpos[0, j], cpos[1, j]])
    return pts


def get_behavior_pairs(num_agents):
    behaviors = ["cautious", "normal", "aggressive"]
    pairs = [["cautious"], ["normal"], ["aggressive"]]
    all_pairs = []
    if num_agents == 2:
        for agent_1 in pairs:
            for agent_2 in behaviors:
                behavior = agent_1 + [agent_2]
                all_pairs.append(behavior)
    if num_agents == 3:
        for agent_1 in pairs:
            for agent_2 in behaviors:
                for agent_3 in behaviors:
                    behavior = agent_1 + [agent_2] + [agent_3]
                    all_pairs.append(behavior)
    if num_agents == 4:
        for agent_1 in pairs:
            for agent_2 in behaviors:
                for agent_3 in behaviors:
                    for agent_4 in behaviors:
                        behavior = agent_1 + [agent_2] + [agent_3] + [agent_4]
                        all_pairs.append(behavior)
    return all_pairs


def generate_start_pos_combinations(scenario):
    start_pose_id = []
    for i in range(0, len(scenario), 2):
        start_pose_id.append(int(scenario[i]))
    start_poses = []
    for i in range(len(start_pose_id)):
        start_poses.append(START_POS_DICT[start_pose_id[i]])

    combinations = list(product(*start_poses))
    return combinations


def mkdir_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
    if not path.exists():
        raise FileNotFoundError('the directory was not found: {}'.format(path))
    return path


def visualize_single_row_in_single_figure_mfp(out_dir: Path, step, src_traj, preds):
    out_path = out_dir / 'row_step_{:03d}.png'.format(step)
    src_trajectory = src_traj
    # num_plot = len(item_dict['prd']) + 1
    fig, axl = plt.subplots(nrows=1, ncols=2, figsize=(2 * 3, 3))
    de_list = []
    num_agent, num_rollout = preds.shape[1], preds.shape[2]

    for ax in axl:
        ax.grid()
        ax.axis('equal')
        ax.set_xlim(left=200, right=300)
        ax.set_ylim(bottom=-200, top=-300)
    for n in range(num_agent):
        axl[0].scatter(src_trajectory[:, n, 0], src_trajectory[:, n, 1])
        axl[1].scatter(src_trajectory[:, n, 0], src_trajectory[:, n, 1])
    for prd in range(len(preds)):
        for n in range(num_agent):
            axl[1].scatter(preds[prd, n, :, 0], preds[prd, n, :, 1])

    plt.savefig(str(out_path))


def visualize_single_row(out_dir: Path, index: int, step, item_dict: Dict[str, Any]):
    out_path = out_dir / 'row_step{}_{:03d}.png'.format(step, index)
    src_trajectory = item_dict['src'].squeeze().cpu().detach().numpy()
    tar_winding = item_dict['tar']['winding'].cpu().detach().numpy()
    tar_trajectory = item_dict['tar']['trajectory'].cpu().detach().numpy()

    num_plot = len(item_dict['prd']) + 1
    fig, axl = plt.subplots(nrows=1, ncols=num_plot, figsize=(num_plot * 3, 3))

    # create a list of data
    data_list = [(tar_winding, np.concatenate((src_trajectory, tar_trajectory), axis=1), -1)]
    src_list = [(tar_winding, src_trajectory, -1)]
    prd_list = [(tar_winding, tar_trajectory, -1)]
    for item in item_dict['prd']:
        frequency = item['frequency']
        prd_winding = item['winding'].squeeze().cpu().detach().numpy()
        prd_trajectory = item['trajectory'].squeeze().cpu().detach().numpy()
        data_list.append((prd_winding, np.concatenate((src_trajectory, prd_trajectory), axis=1), frequency))
        src_list.append((prd_winding, src_trajectory, frequency))
        prd_list.append((prd_winding, prd_trajectory, frequency))

    # compute value range
    num_agent, num_rollout, _ = tar_trajectory.shape
    num_frame = num_rollout + src_trajectory.shape[1]

    traj_list = list(map(itemgetter(1), data_list))
    min_values = np.min(np.array([np.min(t.reshape(num_agent * num_frame, -1), axis=0) for t in traj_list]), axis=0)
    max_values = np.max(np.array([np.max(t.reshape(num_agent * num_frame, -1), axis=0) for t in traj_list]), axis=0)
    x1, y1 = min_values[0], min_values[1]
    x2, y2 = max_values[0], max_values[1]

    def fetch_title(frequency: int, condition, num_agent: int):
        goal = condition[:num_agent]
        winding = condition[num_agent:]
        str_goal = ''.join([str(v) for v in goal])
        str_winding = ''.join([str(v) for v in winding])
        if frequency < 0:
            return 'gt, {}/{}'.format(str_goal, str_winding)
        else:
            return '{}, {}/{}'.format(str(frequency), str_goal, str_winding)

    if not isinstance(axl, np.ndarray):
        return

    for ax, data, src, prd in zip(axl, data_list, src_list, prd_list):
        ax.grid()
        ax.axis('equal')
        ax.set_xlim(left=-30, right=30)
        ax.set_ylim(bottom=-30, top=30)
        for n in range(num_agent):
            ax.scatter(src[1][n, :, 0], src[1][n, :, 1])
        for n in range(num_agent):
            ax.scatter(prd[1][n, :, 0], prd[1][n, :, 1])
        ax.set_title(fetch_title(data[2], data[0], num_agent))
    plt.savefig(str(out_path))


def transform(data, theta, offset):
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([[ct, st], [-st, ct]])
    if offset is None:
        nd = data
    else:
        nd = data - offset
    return np.transpose(R @ np.transpose(nd))


class Agent:
    def __init__(self, player):
        self.carla_agent = player
        self.closest_index = 0
        self.closest_distance = 0
        self.new_waypoints = None
        self.waypoints_np = None
        self.wp_interp = None
        self.hash_wp = None
        self.wp_distance = None
        self.controller = None
        self.past_hist = deque(maxlen=15)
        self.prev_wp = None
        self.curr_wp = None
        self.curr_velocity = None
        self.target_reached = False
        self.control = None
        self.reference_traj = None

    def set_controller(self, controller):
        self.controller = controller

    def set_agent_info(self, waypoints_np, wp_interp, hash_wp, wp_distance, past_hist):
        self.waypoints_np = waypoints_np
        # print (self.waypoints_np)
        self.wp_interp = wp_interp
        self.hash_wp = hash_wp
        self.wp_distance = wp_distance
        self.past_hist = past_hist

    def set_velocity(self, velocity):
        self.curr_velocity = [velocity.x, velocity.y, velocity.z]

    def get_speed(self):
        return np.sqrt(np.sum(np.square(np.array(self.curr_velocity))))

    def update_waypoints(self):
        self.closest_distance = np.linalg.norm(np.array([
            self.waypoints_np[self.closest_index, 0] - self.curr_wp[0],
            self.waypoints_np[self.closest_index, 1] - self.curr_wp[1]]))

        new_distance = self.closest_distance
        new_index = self.closest_index

        while new_distance <= self.closest_distance:
            self.closest_distance = new_distance
            self.closest_index = new_index
            new_index += 1
            if new_index >= self.waypoints_np.shape[0]:  # End of path
                break
            new_distance = np.linalg.norm(np.array([
                self.waypoints_np[new_index, 0] - self.curr_wp[0],
                self.waypoints_np[new_index, 1] - self.curr_wp[1]]))

        new_distance = self.closest_distance
        new_index = self.closest_index

        while new_distance <= self.closest_distance:
            self.closest_distance = new_distance
            self.closest_index = new_index
            new_index -= 1
            if new_index < 0:  # Beginning of path
                break
            new_distance = np.linalg.norm(np.array([
                self.waypoints_np[new_index, 0] - self.curr_wp[0],
                self.waypoints_np[new_index, 1] - self.curr_wp[1]]))

        # Once the closest index is found, return the path that has 1
        # waypoint behind and X waypoints ahead, where X is the index
        # that has a lookahead distance specified by
        # INTERP_LOOKAHEAD_DISTANCE
        waypoint_subset_first_index = self.closest_index - 1
        if waypoint_subset_first_index < 0:
            waypoint_subset_first_index = 0

        waypoint_subset_last_index = self.closest_index
        total_distance_ahead = 0
        while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
            total_distance_ahead += self.wp_distance[waypoint_subset_last_index]
            waypoint_subset_last_index += 1
            if waypoint_subset_last_index >= self.waypoints_np.shape[0]:
                waypoint_subset_last_index = self.waypoints_np.shape[0] - 1
                break

        # Use the first and last waypoint subset indices into the hash
        # table to obtain the first and last indicies for the interpolated
        # list. Update the interpolated waypoints to the controller
        # for the next controller update.
        self.new_waypoints = \
            self.wp_interp[self.hash_wp[waypoint_subset_first_index]:
                           self.hash_wp[waypoint_subset_last_index] + 1]


class World:
    def __init__(self, carla_world, actor_filter, global_intent, start_spawn_ids, behavior_pairs):
        self.world = carla_world
        self.world.tick()
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.collision_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self._spawn_points = self.map.get_spawn_points()
        self.waypts = self.map.generate_waypoints(5.0)

        self.global_intent = None
        self.start_spawn_ids = None
        self.behavior_pairs = None
        self._players = []
        self._agents = []
        self._controllers = []
        self.restart(global_intent, start_spawn_ids, behavior_pairs)

        self.recording_enabled = False
        self.start_time = None
        self.wp_traversed = []
        self.recording_start = 0
        self.frame = 0

    @property
    def agents(self):
        return self._agents

    @property
    def controllers(self):
        return self._controllers

    @property
    def player(self):
        return self._agents[0]

    @property
    def player_ctrl(self):
        return self._controllers[0]

    def fetch_npc(self, index: int):
        return self._agents[index + 1]

    def fetch_npc_ctrl(self, index: int):
        return self._controllers[index + 1]

    def get_intersection_distance(self, intersection_center):
        distances = []
        for k in START_POS_DICT.keys():
            idx = START_POS_DICT[k]
            dist = []
            for val in idx:
                if k == 0 or k == 2:
                    dist.append(math.fabs(self.waypts[val].transform.location.x - intersection_center[0]))
                else:
                    dist.append(math.fabs(self.waypts[val].transform.location.y - intersection_center[1]))
            distances.append(dist)

    def update_behavior_params(self):
        for i in range(len(self.controllers)):
            if self.controllers[i].behavior_str == "cautious":
                self.controllers[i].behavior.max_speed += np.random.uniform(-5, 15)
            elif self.controllers[i].behavior_str == "normal":
                self.controllers[i].behavior.max_speed += np.random.uniform(-10, 10)
            else:
                self.controllers[i].behavior.max_speed += np.random.uniform(-15, 5)

    def set_spawn_points(self):
        spawn_points = []
        spawn_points.append(carla.Transform(carla.Location(
            4.12, -28.4, 0.5), carla.Rotation(0, 0, 0)))
        spawn_points.append(carla.Transform(carla.Location(
            28.4, -55.88, 0.5), carla.Rotation(0, 90, 0)))
        spawn_points.append(carla.Transform(carla.Location(
            55.88, -31.6, 0.5), carla.Rotation(0, 180, 0)))
        spawn_points.append(carla.Transform(carla.Location(
            31.6, -4.12, 0.5), carla.Rotation(0, 270, 0)))
        return spawn_points

    def init(self):
        for player in self.agents:
            player.target_reached = False
        tr = self.waypts[2266].transform
        tr.location.z = 0.5
        self.player.carla_agent.set_transform(tr)
        destination = self.waypts[END_POS_DICT[int(self.global_intent[1])]].transform.location
        self.player_ctrl.set_destination(self.player_ctrl.vehicle.get_location(), destination, clean=True)
        # self.player.carla_agent.set_transform(spawn_points[0])
        for i, (player, ctrl) in enumerate(zip(self.agents[1:], self.controllers[1:])):
            tr = self.waypts[2181].transform
            tr.location.z = 0.5
            player.carla_agent.set_transform(tr)
            destination = self.waypts[END_POS_DICT[int(self.global_intent[i + 1])]].transform.location
            ctrl.set_destination(ctrl.vehicle.get_location(), destination, clean=True)
        self.world.tick()

    def __fetch_spawn_points(self):
        random.shuffle(self._spawn_points)
        return self._spawn_points

    def __spawn_carla_player(self):
        blueprint = self.world.get_blueprint_library().filter(
            'vehicle.tesla.*')[1]
        blueprint.set_attribute('role_name', 'hero1')
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        player = None
        while player is None:
            player = self.world.try_spawn_actor(
                blueprint, self.__fetch_spawn_points()[0])
        return player

    def __reset_agent_and_ctrl(self, player, player_index: int):
        agent = Agent(player)
        tr = self.waypts[self.start_spawn_ids[player_index]].transform
        tr.location.z = 0.5
        agent.carla_agent.set_transform(tr)
        ctrl = BehaviorAgent(agent.carla_agent, behavior=self.behavior_pairs[player_index])
        intent_index = 2 * player_index + 1
        dest = self.waypts[END_POS_DICT[int(self.global_intent[intent_index])]].transform.location
        ctrl.set_destination(ctrl.vehicle.get_location(), dest, clean=True)
        return agent, ctrl

    def __spawn_agent_and_ctrl(self, player_index: int):
        return self.__reset_agent_and_ctrl(self.__spawn_carla_player(), player_index)

    def restart(self, global_intent, start_spawn_ids, behavior_pairs):
        self.global_intent = global_intent
        self.start_spawn_ids = start_spawn_ids
        self.behavior_pairs = behavior_pairs

        num_players = len(self.global_intent) // 2
        if not self._players:
            self._players = [self.__spawn_carla_player() for _ in range(num_players)]

        agents, ctrls = zip(*[self.__reset_agent_and_ctrl(p, i) for i, p in enumerate(self._players)])
        self._agents = list(agents)
        self._controllers = list(ctrls)

        if self.collision_sensor is None:
            self.collision_sensor = CollisionSensor(self.player)

    def tick(self, frame, method='gn'):
        world_snapshot = self.world.get_snapshot()
        for player in self.agents:
            player_snapshot = world_snapshot.find(
                player.carla_agent.id)
            player.prev_wp = player.curr_wp
            player.curr_wp = get_current_pose(player_snapshot)
            player.set_velocity(player_snapshot.get_velocity())
            feat = player.curr_wp[:2]
            prev_feat = player.prev_wp[:2]
            if method == 'mfp':
                feat[1] = -feat[1]
            else:
                feat[0] -= 257.5
                feat[1] = -feat[1] - 247.5
            prev_feat[0] -= 257.5
            prev_feat[1] = -prev_feat[1] - 247.5
            feat = feat + [feat[0] - prev_feat[0]] + [feat[1] - prev_feat[1]]
            player.past_hist.append(feat)

    def check_collision(self, timestamp):
        colhist = self.collision_sensor.get_collision_history()
        collision = [colhist[x + timestamp.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        if max_col > 1000.0:
            return True
        return False

    def convert_agents_to_box(self):
        self.box_pts = []
        for player in self.agents:
            transform = player.carla_agent.get_transform()
            self.box_pts.append(convert_to_box(transform))

    def destroy(self):
        print("Attempting to destroy")
        actors = [self.collision_sensor.sensor, self.player.carla_agent]
        for non_player in self.agents[1:]:
            actors.append(non_player.carla_agent)
        for actor in actors:
            if actor is not None and actor.is_alive:
                actor.destroy()


class CollisionSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.collision = False
        world = self._parent.carla_agent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent.carla_agent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        self.collision = True
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class Logger:
    def __init__(self, num_agents, log_path, scenarios):
        self.num_agents = num_agents
        self.log_path = log_path
        self.scenarios = scenarios
        self.current_scenario = None
        self.current_scene_tuples = []
        self.column_header = None
        self.current_save_path = None
        self.create_csv_files()
        self.episode_buffer = []
        for i in range(num_agents):
            self.episode_buffer.append([])
        self.scene_buffer = []

    def create_csv_files(self):
        column_header = []
        for i in range(self.num_agents):
            column_header.extend([f"intention_{i}", f"x{i}_data", f"y{i}_data", f"yaw{i}_data", f"speed{i}_data"])
        self.column_header = column_header
        df = pd.DataFrame(columns=column_header)
        for scene_id in self.scenarios:
            csv_path = os.path.join(
                self.log_path, "scenario_{}.pth".format(scene_id))
            df.to_csv(csv_path)

    def update_episode_buffer(self, world):
        for i, player in enumerate(world.agents):
            agent_data = player.curr_wp + [player.get_speed()]
            self.episode_buffer[i].append(agent_data)

    def reset_episode(self):
        # update scene buffer and then clear episode_buffer
        if len(self.episode_buffer[0]) > 0:
            ep_buffer = np.array(self.episode_buffer)
            data = {}
            for i in range(self.num_agents):
                data[f"intention_{i}"] = self.current_scene_tuples[i]
                data[f"x{i}_data"] = ep_buffer[i, :, 0]
                data[f"y{i}_data"] = ep_buffer[i, :, 1]
                data[f"yaw{i}_data"] = ep_buffer[i, :, 2]
                data[f"speed{i}_data"] = ep_buffer[i, :, 3]

            self.scene_buffer.append(data)
            self.episode_buffer.clear()
            for i in range(self.num_agents):
                self.episode_buffer.append([])

    def remove_collision_data(self):
        self.episode_buffer.clear()
        for i in range(self.num_agents):
            self.episode_buffer.append([])

    def update_current_scenario(self, scenario):
        self.save_scene()
        self.current_scenario = scenario
        self.current_scene_tuples.clear()
        self.scene_buffer.clear()
        for i in range(0, len(scenario), 2):
            start, end = int(scenario[i]), int(scenario[i + 1])
            self.current_scene_tuples.append((start, end))
        self.current_save_path = os.path.join(
            self.log_path, "scenario_{}.pth".format(scenario))

    def save_scene(self):
        if len(self.scene_buffer) > 0:
            torch.save(self.scene_buffer, self.current_save_path)

    def step(self, world):
        self.update_episode_buffer(world)

    def save(self, step):
        if step % 3:
            self.save_scene()


class GraphNetPredictor:
    def __init__(self, trainer, u_dim, B, N, T, rollout_size, d):
        self.trainer = trainer
        self.u_dim = u_dim
        self.B = 1
        self.N = N
        self.T = T
        self.rn = rollout_size
        self.d = d
        self.winding_constraints = fetch_winding_constraints(N)
        self.inputs = []
        self.prev_pred = None
        self.prev_probs = None
        fig = plt.figure()
        # ax = plt.gca()
        ax = p3.Axes3D(fig)
        ax.view_init(90, -90)
        ax.set_xlim((200, 300))
        ax.set_ylim((-200, -300))
        # # ax.set_zlim((0, 20))
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.pause(1.)
        self.eval_root_dir = Path.cwd() / '.gnn/eval'

    def set_src_dst_tensor(self, global_intent):
        self.global_intent = global_intent
        src_index_tensor = [3]
        for i in range(2, len(global_intent), 2):
            src_index_tensor.append(int(global_intent[i]))
        src_index_tensor = torch.tensor(
            [src_index_tensor]).to(self.trainer.device)
        self.src_index_tensor = src_index_tensor
        # dst_index = []
        # for i in range(1, len(global_intent), 2):
        #     dst_index.append(int(global_intent[i]))
        # self.dst_index = torch.tensor(
        #     [dst_index]).to(self.trainer.device)
        # self.dst_list = self.trainer.generate_dst_combinations(src_index_tensor)

        # self.winding = torch.zeros([self.B, self.N*(self.N-1), 2]).to(self.trainer.device)
        # self.dest = torch.zeros([self.B, self.N, 4]).to(self.trainer.device)

    def get_input(self, world):
        curr_state = np.empty((self.B, self.N, self.T, self.d))
        for i, player in enumerate(world.agents):
            past_hist_np = np.expand_dims(np.asarray(player.past_hist), axis=0)
            curr_state[0][i] = past_hist_np
        # plt.plot(curr_state[0, 0, :, 0], -1 * curr_state[0, 0, :, 1])
        curr_state = torch.from_numpy(curr_state).float().to(self.trainer.device)
        return curr_state

    def stop(self):
        plt.savefig("rollouts.png")
        plt.clf()
        if len(self.inputs) > 0:
            output = torch.stack(self.inputs)
            torch.save(output, "./inputs.pt")
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        # ax = plt.gca()
        ax.view_init(90, -90)
        ax.set_xlim((200, 300))
        ax.set_ylim((-200, -300))
        # ax.set_zlim((0, 20))
        self.prev_pred = None

    def predict(self, world, step, dt, scene_id, behavior_id, param_id):
        threshold = 0.0
        output_prd = []
        probs = []
        output = []
        with torch.no_grad():
            curr_state = self.get_input(world)  # Shape: [B x n x T x d]
            # print (curr_state.size())
            self.inputs.append(curr_state)
            next_state = torch.zeros((self.B, self.N, self.rn, self.d)).to(
                self.trainer.device)  # Shape: [B x n x rollout_num x d]
            tar_winding = torch.zeros((self.B, self.N * (self.N - 1))).to(
                self.trainer.device)  # B, E
            src_index_tensor = self.src_index_tensor.to(
                self.trainer.device)
            tar_goal = torch.zeros((self.B, self.N)).to(
                self.trainer.device)  # B, n
            winding_onehot = torch.zeros((self.B, self.N * (self.N - 1), 2)).to(
                self.trainer.device)  # B, E, 2
            goal_onehot = torch.zeros((self.B, self.N, 4)).to(
                self.trainer.device)  # B, n, 4

            num_goal = tar_goal.shape[-1]
            num_winding = tar_winding.shape[-1]
            tar = torch.cat((tar_goal, tar_winding), dim=-1)
            prd_cond = self.trainer.model_wrapper.eval_winding(curr_state, tar_winding.shape[0], tar_goal.shape[0])

            for r, rows in enumerate(prd_cond):
                item = {
                    'src': curr_state[r, :].squeeze(),
                    'tar': {'winding': tar[r, :], 'trajectory': next_state[r, :].squeeze()},
                    'prd': []
                }
                # for prd_goal_onehot, prd_winding_onehot, row in zip(dests, winds, rows):
                for freq, row in rows:
                    prd_goal = row[:num_goal]
                    prd_goal[0] = 1
                    prd_winding = row[num_goal:]
                    # skip if the goal position label is same with the starting position label
                    valid = True
                    if int(prd_goal[0]) != int(self.global_intent[1]):
                        valid = False
                    # # if int(prd_goal[1]) != int(self.global_intent[3]):
                    # #     continue
                    for g, s in zip(prd_goal, src_index_tensor[r, :]):
                        if g.item() == s.item():
                            valid = False

                    # skip if the winding numbers are inconsistent
                    for i1, i2 in self.winding_constraints:
                        if prd_winding[i1] != prd_winding[i2]:
                            valid = False
                    if not valid:
                        continue

                    prd_goal_onehot = onehot_from_index(prd_goal, 4).unsqueeze(0)
                    prd_winding_onehot = onehot_from_index(prd_winding, 2).unsqueeze(0)
                    # print ("wind: ", prd_winding_onehot)
                    # print ("dest: ", prd_goal_onehot)
                    curr_input = curr_state[r, ...].squeeze().unsqueeze(0)
                    B, prd_traj = self.trainer.model_wrapper.eval_trajectory(
                        curr_input, next_state.shape[2], prd_winding_onehot, prd_goal_onehot)
                    if float(freq) / 100 > threshold:
                        output_prd.append(prd_traj[0].cpu().numpy())
                        probs.append(float(freq) / 100)
                    item['prd'].append({
                        'frequency': freq,
                        'winding': row,
                        'trajectory': prd_traj,
                    })
                    # print ("TRAJ SIZE: ", prd_traj.size())
                output.append(item)

            eval_dir = self.eval_root_dir / f'scenario_{scene_id}' / f'behavior_{behavior_id}_{param_id}'
            mkdir_if_not_exists(eval_dir)
            if len(output) > 0:
                for item_id, item in enumerate(output):
                    visualize_single_row(eval_dir, step, item_id, item)
            # output_prd = output_prd
            output_prd = np.array(output_prd)
            # print("PROBS: ", probs)
            probs = np.array(probs)
            current_time = step * dt
            t_arr = np.arange(25) * 0.1 + current_time
            if len(probs) > 0:
                for i in range(output_prd.shape[0]):
                    for k in range(self.N):
                        output_prd[i, k, :, 0] += 257.5
                        output_prd[i, k, :, 1] += 247.5
                        output_prd[i, k, :, 1] = -1 * output_prd[i, k, :, 1]
                        output_prd[i, k, :, 3] = -1 * output_prd[i, k, :, 3]
                        output_prd[i, k, :self.rn - 1, 2] = output_prd[i, k, 1:, 0] - output_prd[i, k, :self.rn - 1, 0]
                        output_prd[i, k, :self.rn - 1, 3] = output_prd[i, k, 1:, 1] - output_prd[i, k, :self.rn - 1, 1]
                        dx, dy = output_prd[i, k, :, 2], output_prd[i, k, :, 3]
                        output_prd[i, k, :, 3] = np.sqrt(output_prd[i, k, :, 2] ** 2 + output_prd[i, k, :, 3] ** 2) / (
                            0.1)
                        output_prd[i, k, :, 2] = (-(np.arctan2(dy, dx) * 180 / np.pi) + 360.) % 360.
                self.prev_pred = output_prd
                self.prev_probs = probs
                return output_prd, probs
            else:
                return self.prev_pred, self.prev_probs
