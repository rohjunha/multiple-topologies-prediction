class Kinematics:
    EPSILON = 1e-5
    NCTRL = 2
    NPOS = 3

    def __init__(self, logger, dtype):
        self.logger = logger
        self.dtype = dtype

        self.reset()

    def reset(self):
        k = 256  # self.params.get_int("K", default=62)
        p = 1  # self.params.get_int("P", default=1)
        controller = 'mpc'  # self.params.get_str("controller", default="mpc")
        if controller == 'umpc':
            self.set_k(k*p)
        else:
            self.set_k(k)

    def set_k(self, k):
        """
        In some instances the internal buffer size needs to be changed. This easily facilitates this change

        Args:
        k [int] -- Number of rollouts
        """
        self.K = k
        self.wheel_base = 2.875 # self.params.get_float("model/wheel_base", default=0.29)

        # TODO: import utils
        time_horizon = 0.08 * 3 # utils.get_time_horizon() # (40, 0.015), (20, 0.03)
        T = 25  # self.params.get_int("T", default=15)
        self.dt = 0.1  #0.15 # 0.07 # time_horizon / T

        self.sin2beta = self.dtype(self.K)
        self.deltaTheta = self.dtype(self.K)
        self.deltaX = self.dtype(self.K)
        self.deltaY = self.dtype(self.K)
        self.sin = self.dtype(self.K)
        self.cos = self.dtype(self.K)

    def apply(self, pose, ctrl):
        """
        Args:
        pose [(K, NPOS) tensor] -- The current position in "world" coordinates
        ctrl [(K, NCTRL) tensor] -- Control to apply to the current position
        Return:
        [(K, NCTRL) tensor] The next position given the current control
        """
        # print ("ctr: ", ctrl.size(), " rhs: ", (self.K, self.NCTRL))
        assert pose.size() == (self.K, self.NPOS)
        assert ctrl.size() == (self.K, self.NCTRL)

        self.sin2beta.copy_(ctrl[:, 1]).tan_().mul_(0.5).atan_().mul_(2.0).sin_().add_(
            self.EPSILON
        )

        self.deltaTheta.copy_(ctrl[:, 0]).div_(self.wheel_base).mul_(
            self.sin2beta
        ).mul_(self.dt)

        self.sin.copy_(pose[:, 2]).sin_()
        self.cos.copy_(pose[:, 2]).cos_()

        self.deltaX.copy_(pose[:, 2]).add_(self.deltaTheta).sin_().sub_(self.sin).mul_(
            self.wheel_base
        ).div_(self.sin2beta)

        self.deltaY.copy_(pose[:, 2]).add_(self.deltaTheta).cos_().neg_().add_(
            self.cos
        ).mul_(self.wheel_base).div_(self.sin2beta)

        nextpos = self.dtype(self.K, 3)
        nextpos.copy_(pose)
        nextpos[:, 0].add_(self.deltaX)
        nextpos[:, 1].add_(self.deltaY)
        nextpos[:, 2].add_(self.deltaTheta)

        return nextpos
