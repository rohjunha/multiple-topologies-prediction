import threading

import torch

from controller import collision_checker


class Tracking:
    NPOS = 3  # x, y, theta
    TASK_TYPE = "PATH"

    def __init__(self, logger, dtype, map, world_rep, value_fn):
        self.logger = logger
        self.dtype = dtype
        self.map = map

        self.world_rep = world_rep
        self.value_fn = value_fn

        # self.params.get_bool("debug/viz_waypoint", True)
        self.viz_waypoint = False
        self.do_log_cte = False  # self.params.get_bool("debug/log_cte", True)
        # self.params.get_bool("debug/flag/viz_rollouts", False)
        self.viz_rollouts = False
        self.n_viz = -1  # self.params.get_int("debug/viz_rollouts/n", -1)
        # self.params.get_bool("debug/viz_rollouts/print_stats", False)
        self.print_stats = False
        self.circle_offsets = [-1.1, 1.1, 3.1]
        self.circle_radius = [2.0, 2.0, 2.0]
        self.path_select_weight = 10
        self.collision_checker = collision_checker.CollisionChecker(
            self.circle_offsets, self.circle_radius, self.path_select_weight)

        self.path = None
        self.reset()

    def reset(self):
        self.T = 25  # self.params.get_int("T", default=15)
        self.K = 64 * 3  # self.params.get_int("K", default=62)
        self.P = 1  # self.params.get_int("P", default=1)
        controller = 'mpc'  # self.params.get_str("controller", default="mpc")
        if controller == 'umpc':
            self.nR = self.K * self.P  # num rollouts
        else:
            self.nR = self.K  # num rollouts

        # self.params.get_float("cost_fn/finish_threshold", 0.5)
        self.finish_threshold = 0.5
        # self.params.get_float("cost_fn/exceed_threshold", 4.0)
        self.exceed_threshold = 4.0
        # TODO: collision check
        # self.params.get_bool("cost_fn/collision_check", True)
        self.collision_check = True

        self.lookahead = 12.0  # self.params.get_float("cost_fn/lookahead", 1.0)

        # self.params.get_float("cost_fn/dist_w", default=1.0)
        self.dist_w = 1.0
        # self.params.get_float("cost_fn/obs_dist_w", default=5.0)
        self.obs_dist_w = 83.0
        # self.params.get_float("cost_fn/error_w", default=1.0)
        self.error_w = 20.0
        # self.params.get_float("cost_fn/smoothing_discount_rate", default=0.04)
        self.smoothing_discount_rate = 0.03
        # self.params.get_float("cost_fn/smooth_w", default=3.0)
        self.smooth_w = 10.0
        self.prox_w = 100.0
        # self.params.get_float("cost_fn/bounds_cost", default=100.0)
        self.bounds_cost = 10e5

        self.obs_dist_cooloff = torch.arange(
            1, self.T + 1).mul_(2).type(self.dtype)

        self.discount = self.dtype(self.T - 1)
        self.discount_p = self.dtype(self.T)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount_p[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.T - 1).type(self.dtype) * -1)
        self.discount_p.pow_(torch.arange(0, self.T).type(self.dtype) * -1)

        self._prev_pose = None
        self._prev_index = -1
        self._cache_thresh = 0.01

        self.path_lock = threading.Lock()
        with self.path_lock:
            self.path = None

        # if self.collision_check:
        #     self.world_rep.reset()

    def apply(self, poses, ip, world, player_id=None, preds=None):
        """
        Args:
        poses [(nR, T, 3) tensor] -- Rollout of T positions
        ip    [(3,) tensor]: Inferred pose of car in "world" mode

        Returns:
        [(nR,) tensor] costs for each nR paths
        """
        # print ("Poses: ", poses.size())
        index, waypoint = self._get_reference_index(ip)
        assert poses.size() == (self.nR, self.T, self.NPOS)
        assert self.path.size()[1] == 4

        all_poses = poses.view(self.nR * self.T, self.NPOS)

        # use terminal distance (nR, tensor)
        # TODO: should we look for CTE for terminal distance?
        if preds is not None:
            poses = torch.from_numpy(preds[:, 0, :, :])

        errorcost = poses[:, self.T - 1,
                          :2].sub(waypoint[:2]).norm(dim=1).mul(self.error_w)

        # reward smoothness by taking the integral over the rate of change in poses,
        # with time-based discounting factor
        smoothness = (
            ((poses[:, 1:, 2] - poses[:, : self.T - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        ).mul(self.smooth_w)

        result = errorcost.add(smoothness)

        all_collision = False
        # get all collisions (nR, T, tensor)
        if self.collision_check:
            collisions = self.collision_checker.collision_check(
                poses, world, player_id)
            collision_cost = torch.from_numpy(collisions).mul(self.bounds_cost)
            result = result.add(collision_cost)  # .add(obs_dist_cost)
            all_collision = False
            if torch.sum(collision_cost) == self.K * self.bounds_cost:
                all_collision = True

        if all_collision:
            print("Collision: ", all_collision)
        return result, all_collision

    def set_task(self, pathmsg):
        """
        Args:
        path [(x,y,h,v),...] -- list of xyhv named tuple
        """
        self._prev_pose = None
        self._prev_index = None
        path = self.dtype([[pathmsg[i][0], pathmsg[i][1], pathmsg[i]
                            [2], pathmsg[i][3]] for i in range(len(pathmsg))])
        assert path.size() == (len(pathmsg), 4)
        # TODO: any other tests to check validity of path?

        with self.path_lock:
            self.path = path
            self.waypoint_diff = torch.mean(torch.norm(
                self.path[1:, :2] - self.path[:-1, :2], dim=1))
            # TODO: could add value fn that checks
            # viability of path
            return True

    def task_complete(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        with self.path_lock:
            if self.path is None:
                return False
        # print ("Check complete")
        return self._path_complete(state)

    def _path_complete(self, pose):
        '''
        path_complete computes whether the vehicle has completed the path
            based on whether the reference index refers to the final point
            in the path and whether e_x is below the finish_threshold
            or e_y exceeds an 'exceed threshold'.
        input:
            pose - current pose of the vehicle [x, y, heading]
            error - error vector [e_x, e_y]
        output:
            is_path_complete - boolean stating whether the vehicle has
                reached the end of the path
        '''
        index, waypoint = self._get_reference_index(pose)
        if index == (len(self.path) - 1):
            error = self._get_error(pose, index)
            result = (error[0] < self.finish_threshold) or (
                abs(error[1]) > self.exceed_threshold)
            result = True if result == 1 else False
            print("Path complete: ", result)
            return result
        return False

    def _get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        with self.path_lock:
            if ((self._prev_pose is not None) and
                    (torch.norm(self._prev_pose[:2] - pose[:2]) < self._cache_thresh)):
                return (self._prev_index, self.path[self._prev_index])
            diff = self.path[:, :3] - pose
            dist = diff[:, :2].norm(dim=1)
            index = dist.argmin()
            index += int(self.lookahead / self.waypoint_diff)
            index = min(index, len(self.path)-1)
            self._prev_pose = pose
            self._prev_index = index
            return (index, self.path[index])

    def _get_error(self, pose, index=-1):
        '''
        Computes the error vector for a given pose and reference index.
        input:
            pose - pose of the car [x, y, heading]
            index - integer corresponding to the reference index into the
                reference path
        output:
            e_p - error vector [e_x, e_y]
        '''
        index, waypoint = self._get_reference_index(pose)
        theta = pose[2]
        c, s = torch.cos(theta), torch.sin(theta)
        R = self.dtype([(c, s), (-s, c)])
        return torch.matmul(R, waypoint[:2] - pose[:2])
