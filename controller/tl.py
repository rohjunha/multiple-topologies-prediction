import torch

# Uniform steer across rollout
class TL:
    # Size of control vector
    NCTRL = 2

    def __init__(self, dtype, _model):
        self.dtype = dtype
        self.max_delta = 1.21
        self.reset()

    def use_trajopt(self):
        return False

    def reset(self):
        self.K = 64 #self.params.get_int("K", default=62)
        self.T = 25 # self.params.get_int("T", default=15)

        min_delta = -1.21 # self.params.get_float("trajgen/min_delta", default=-0.34)
        max_delta = 1.21 # self.params.get_float("trajgen/max_delta", default=0.34)

        desired_speed = 4. # self.params.get_float("trajgen/desired_speed", default=1.0)
        step_size = (max_delta - min_delta) / (self.K - 1)
        deltas = torch.arange(min_delta, max_delta + step_size, step_size)

        # The controls for TL are precomputed, and don't change
        self.ctrls = self.dtype(self.K, self.T, self.NCTRL)
        self.ctrls[:, :, 0] = desired_speed
        for t in range(self.T):
            self.ctrls[:, t, 1] = deltas
        self.ctrls = torch.repeat_interleave(self.ctrls, 3, dim=0)

    def get_control_trajectories(self, velocity):
        """
        Returns:
        [(K, T, NCTRL) tensor] -- of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        """
        self.ctrls[:, :, 0] = velocity
        return self.ctrls

    def get_control_trajectories_2(self, velocities):
        """
        Returns:
        [(K, T, NCTRL) tensor] -- of controls
            ([:, :, 0] is the desired speed, [:, :, 1] is the control delta)
        """
        for i, velocity in enumerate(velocities):
            self.ctrls[i * self.K: self.K * (i + 1), :, 0] = velocity
        return self.ctrls

    def generate_control(self, controls, costs):
        """
        Args:
        controls [(K, T, NCTRL) tensor] -- Returned by get_control_trajectories
        costs [(K, 1) tensor] -- Cost to take a path

        Returns:
        [(T, NCTRL) tensor] -- The lowest cost trajectory to take
        """
        # assert controls.size() == (self.K, self.T, 2)
        # assert costs.size() == (self.K,)
        _, idx = torch.min(costs, 0)
        return controls[idx], idx
