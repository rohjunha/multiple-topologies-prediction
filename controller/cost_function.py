import threading

import torch

from controller import collision_checker


class CostFunction:
    NPOS = 3  # x, y, theta

    def __init__(
            self,
            rollout_size: int,
            check_collision: bool = False):
        self.rollout_size = rollout_size
        self.circle_offsets = [-1.1, 1.1, 3.1]
        self.circle_radius = [1.4, 1.4, 1.4]
        self.path_select_weight = 10
        self.collision_checker = collision_checker.CollisionChecker(self.circle_offsets, self.circle_radius, self.path_select_weight)
        self.check_collision = check_collision
        self.dtype = torch.FloatTensor

        self.prox_w = 100.0 # proximity weight
        self.smoothing_discount_rate = 0.03
        self.smooth_w = 50.0 # speed smoothness cost weight
        self.collision_cost = 10e4 # collision_cost weights
        self.prob_cost = 1.0

        self.discount = self.dtype(self.rollout_size - 1)
        self.discount_p = self.dtype(self.rollout_size)

        self.discount[:] = 1 + self.smoothing_discount_rate
        self.discount_p[:] = 1 + self.smoothing_discount_rate
        self.discount.pow_(torch.arange(0, self.rollout_size - 1).type(self.dtype) * -1)
        self.discount_p.pow_(torch.arange(0, self.rollout_size).type(self.dtype) * -1)

        self.NPOS = 4
        self.lookahead = 14.0
        self._prev_pose = None
        self._prev_index = -1
        self._cache_thresh = 0.01

        self.path_lock = threading.Lock()
        with self.path_lock:
            self.path = None

    def apply(self, world, preds, waypoint, probs):
        index, waypoint = self._get_reference_index(world.player.curr_wp)
        probs = torch.Tensor(1. / probs)
        preds = torch.from_numpy(preds)
        ego_preds = preds[:, 0, :]
        # poses = ego_preds.view(-1, self.NPOS)

        errorcost = ego_preds[:, self.rollout_size - 1, :2].sub(torch.Tensor(waypoint[:2])).norm(dim=1).mul(self.prox_w)

        smoothness = (
            ((ego_preds[:, 1:, 3] - ego_preds[:, : self.rollout_size - 1, 3]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        ).mul(self.smooth_w)

        result = errorcost.add(smoothness)

        smoothness2 = (
            ((ego_preds[:, 1:, 2] - ego_preds[:, : self.rollout_size - 1, 2]))
            .abs()
            .mul(self.discount)
            .sum(dim=1)
        ).mul(self.smooth_w)
        result = errorcost.add(smoothness2)

        if self.check_collision:
            collisions = self.collision_checker.collision_check(ego_preds, world)
            # collision_cost = torch.from_numpy(collisions).sum(dim=1).mul(self.collision_cost)
            collision_cost = torch.from_numpy(collisions).mul(self.collision_cost)
            result = result.add(collision_cost)

        prob_cost = 0.1 * probs
        result = result.add(prob_cost)
        min_cost_id = torch.argmin(result).item()
        return min_cost_id

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

    def _get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        pose = torch.Tensor(pose)
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