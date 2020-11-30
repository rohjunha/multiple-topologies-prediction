class UMPC:
    # Number of elements in the position vector
    NPOS = 3

    def __init__(self, dtype, mvmt_model, trajgen, cost):
        self.dtype = dtype

        self.trajgen = trajgen
        self.kinematics = mvmt_model
        self.cost = cost

        self.reset(init=True)

    def reset(self, init=False):
        """
        Args:
        init [bool] -- whether this is being called by the init function
        """
        # TODO: Use fixed values
        self.T = 25  # self.params.get_int("T", default=15)
        self.K = 64 * 3  # self.params.get_int("K", default=62)
        self.P = 1  # self.params.get_int("P", default=128)
        # self.nR = self.K * self.P  # num rollouts
        self.nR = self.K

        # Update number of rollouts required by kinematics model
        self.kinematics.set_k(self.nR)

        # Rollouts buffer, the main engine of our computation
        self.rollouts = self.dtype(self.nR, self.T, self.NPOS)

        self.desired_speed = [0.1, 7.0, 4.0]  # self.params.get_float(
        # "trajgen/desired_speed", default=1.0)

        if not init:
            self.trajgen.reset()
            self.kinematics.reset()
            self.cost.reset()

    def step(self, state, ip, world, player_id=None, preds=None):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        assert ip.size() == (3,)

        if self.task_complete(ip):
            return None, None

        self.state = state
        self.ip = ip
        # For each K trial, the first position is at the current position

        v = self.desired_speed  # self.cost.get_desired_speed(ip, self.desired_speed)
        trajs = self.trajgen.get_control_trajectories_2(v)

        costs, all_collision = self._rollout2cost(trajs, world, player_id, preds)

        if not self.trajgen.use_trajopt():
            result, idx = self.trajgen.generate_control(trajs, costs)
        else:
            result, idx = self.trajgen.generate_control(trajs,
                                                        costs, self._rollout2cost)

        if all_collision:
            result[:, :] = 0.1
        # idx = None

        if idx is None:
            # TODO: optimize so we only roll out single canonical control
            idx = 0
            k_prev = self.kinematics.K
            self.kinematics.set_k(1)
            self.rollouts[idx, 0] = ip
            for t in range(1, self.T):
                cur_x = self.rollouts[idx, t - 1].resize(1, self.NPOS)
                self.rollouts[idx, t] = self.kinematics.apply(cur_x,
                                                              result[t].resize(1, 2))
            self.kinematics.set_k(k_prev)

        return result, self.rollouts[idx]

    def set_task(self, task):
        return self.cost.set_task(task)

    def task_complete(self, state):
        """
        Args:
        state [(3,) tensor] -- Current position in "world" coordinates
        """
        # print ("TASK COMPLETE!")
        return self.cost.task_complete(state)

    def _perform_rollout(self, trajs):
        self.rollouts.zero_()
        self.rollouts[:, 0] = self.state.expand_as(self.rollouts[:, 0])
        assert trajs.size() == (self.K, self.T, 2)
        for t in range(1, self.T):
            cur_x = self.rollouts[:, t - 1]
            self.rollouts[:, t] = self.kinematics.apply(cur_x, trajs[:, t - 1])

    def _rollout2cost(self, trajs, world, player_id=None, preds=None):
        self._perform_rollout(trajs)
        return self.cost.apply(self.rollouts, self.ip, world, player_id, preds)

    # def _rollout2cost(self, trajs):
    #     self._perform_rollout(trajs)
    #     costs = self.cost.apply(self.rollouts, self.ip)
    #     return costs.resize(self.P, self.K).mean(0)
