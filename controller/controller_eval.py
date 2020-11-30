import numpy as np

from controller import cutils
from utils import rhctensor
from agents.navigation.controller import PIDLongitudinalController, PIDLateralController
from controller.cost import Tracking
from controller.kinematics import Kinematics
from controller.pid import PID
from controller.pure_pursuit import PP
from controller.tl import TL
from controller.umpc import UMPC
from controller.world_rep import Simple


class Controller2D:
    def __init__(self, waypoints, player):
        self.vars = cutils.CUtils()
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._prev_timestamp = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70. / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

        # MPC
        self.logger = None #logger.RosLog()
        self.dtype = rhctensor.float_tensor()
        self.model = Kinematics(self.logger, self.dtype)
        self.trajgen = TL(self.dtype, self.model)
        # TODO: generate map data wrt carla
        self.map_data = None

        world_rep = Simple(self.logger, self.dtype, self.map_data)
        value_fn = None #SimpleKNN(self.logger, self.dtype, self.map_data)
        self.cost_fn = Tracking(self.logger, self.dtype, self.map_data, world_rep, value_fn)
        self.mpc = UMPC(self.dtype, self.model, self.trajgen, self.cost_fn)

        # PIDs
        self.steering_pid = PID(P=0.5, I=0.0, D=0.)
        # self.steering_pid = PID(P=0.34611, I=0.0370736, D=3.5349)
        self.steering_pid.setSampleTime = 0.1

        self.throttle_brake_pid = PID(P=7.0, I=1.0, D=1.026185)
        # self.throttle_brake_pid = PID(P=0.37, I=0.032, D=0.024)
        # self.throttle_brake_pid  = PID(P=1.90, I=0.05, D=0.80)
        self.throttle_brake_pid.setSampleTime = 0.1

        self.pid_throttle = PIDLongitudinalController(player, 0.37, 0.024, 0.032, dt=0.1)
        self.pid_steer = PIDLateralController(player, 0.05, 0., 0., 0.1)
        # Pure Pursuit
        # self.pp = PP(L=0.5, k=1.00, k_Ld=1.5)
        # self.pp = PP(L=4.6, k=1.00, k_Ld=1.7)
        self.pp = PP(L=2.87, k=1.00, k_Ld=0.7)

    # def set_path(self, traj):
    #     path_msg = []
    #     for i in range(len(traj)):
    #         path_msg.append(XYHV(traj[i][0], traj[i][1], traj[i][2], traj[i][3]))
    #     self.cost_fn.set_task(path_msg)

    def set_path(self, traj):
        self.cost_fn.set_task(traj)

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_timestamp = timestamp
        self._current_frame = frame
        # if frame < 60:
        #     self.mpc.cost.collision_check = False
        # else:
        #     self.mpc.cost.collision_check = True
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0
        # for i in range(len(self._waypoints)):
        #     dist = np.linalg.norm(np.array([
        #         self._waypoints[i][0] - self._current_x,
        #         self._waypoints[i][1] - self._current_y]))
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_idx = i
        ind = 0
        distance_this_index = np.linalg.norm(np.array([
                self._waypoints[ind][0] - self._current_x,
                self._waypoints[ind][1] - self._current_y]))
        while True:
            distance_next_index = np.linalg.norm(np.array([self._waypoints[ind + 1][0] - self._current_x, self._waypoints[ind + 1][1] - self._current_y]))
            if distance_this_index < distance_next_index:
                break
            ind = ind + 1 if (ind + 1) < len(self._waypoints) else ind
            distance_this_index = distance_next_index
        # old_nearest_point_index = ind
        k = 0.5
        Lfc = 8.5
        Lf = k * self._current_speed + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > np.linalg.norm(np.array([self._waypoints[ind][0] - self._current_x, self._waypoints[ind][1] - self._current_y])):
            if (ind + 1) >= len(self._waypoints):
                break  # not exceed goal
            ind += 1
        # if min_idx < len(self._waypoints)-1:
        #     if self._current_frame < 10:
        #         desired_speed = self._waypoints[min_idx][3]
        #     else:
        #         desired_speed = self._waypoints[min_idx][3]
        #     self.target_wp = self._waypoints[min_idx]
        # else:
        print ("IDX: ", ind)
        desired_speed = self._waypoints[ind][3]
        self.target_wp = self._waypoints[ind]

        self._desired_speed = desired_speed

    def update_desired_speed_mpc(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            if self._current_frame < 10:
                desired_speed = self._waypoints[min_idx][3]
            else:
                desired_speed = self._waypoints[min_idx][3]
            self.target_wp = self._waypoints[min_idx]
        else:
            desired_speed = self._waypoints[-1][3]
            self.target_wp = self._waypoints[-1]

        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints, world=None):
        if world is None:
            self._waypoints = new_waypoints
        else:
            x = self._current_x
            y = self._current_y
            yaw = self._current_yaw
            ip = self.dtype(3)
            ip[0], ip[1], ip[2] = x, y, yaw
            state = ip
            # next_traj, rollout = self.mpc.step(state, ip, world, None, new_waypoints)
            self._waypoints = new_waypoints[0, 0, :]# next_traj

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def map_coord_2_Car_coord(self, x, y, yaw, waypoints):

        wps = np.squeeze(waypoints)
        wps_x = wps[:, 0]
        wps_y = wps[:, 1]

        num_wp = wps.shape[0]

        # create the Matrix with 3 vectors for the waypoint x and y coordinates w.r.t. car
        wp_vehRef = np.zeros(shape=(3, num_wp))
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        wp_vehRef[0, :] = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef[1, :] = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)

        return wp_vehRef

    def update_controls(self, world, player_id=None, preds=None):
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        ip = self.dtype(3)
        ip[0], ip[1], ip[2] = x, y, yaw
        state = ip

        v = self._current_speed
        # print ("CURRENT SPEED: ", v)
        self.update_desired_speed()
        v_desired = self._desired_speed
        t = self._current_timestamp
        waypoints = self._waypoints
        next_waypt = waypoints[0]
        throttle_output = 0.
        steer_output = 0.
        brake_output = 0.

        self.vars.create_var('v_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            wps_vehRef = self.map_coord_2_Car_coord(x, y, yaw, waypoints)
            wps_vehRef_x = wps_vehRef[0, :]
            wps_vehRef_y = wps_vehRef[1, :]

            # fit a 3rd order polynomial to the waypoints
            coeffs = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)

            CarRef_x = CarRef_y = CarRef_yaw = 0.0
            cte = np.polyval(coeffs, CarRef_x) - CarRef_y
            yaw_err = CarRef_yaw - np.arctan(coeffs[1])

            speed_err = v_desired - v

            #### MPC ####
            self.throttle_brake_pid.update(speed_err, output_limits = [-1.0, 1.00])
            if self.throttle_brake_pid.output > 0.0:
                throttle_output = self.throttle_brake_pid.output
                brake_output = 0.
            elif self.throttle_brake_pid.output >= -0.85:
                throttle_output = 0.
                brake_output = 0.
            else:
                throttle_output = 0.
                brake_output = (-self.throttle_brake_pid.output + 1)/(1-0.85)

            steer_output = float(self.pp.update(coeffs, v))


            if (type(throttle_output) == float):
                self.set_throttle(throttle_output)
            else:
                self.set_throttle(throttle_output.item())  # in percent (0 to 1)
            if (type(steer_output) == float):
                self.set_steer(steer_output)
            else:
                self.set_steer(steer_output.item())        # in rad (-1.22 to 1.22)
            if (type(brake_output) == float):
                self.set_brake(brake_output)        # in percent (0 to 1)
            else:
                self.set_brake(brake_output.item())

        self._prev_timestamp = self._current_timestamp
        self.vars.v_previous = v  # Store forward speed to be used in next step


    def update_controls_2(self, world, player_id=None, preds=None):
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        ip = self.dtype(3)
        ip[0], ip[1], ip[2] = x, y, yaw
        state = ip

        v = self._current_speed
        # print ("CURRENT SPEED: ", v)
        self.update_desired_speed_mpc()
        v_desired = self._desired_speed
        t = self._current_timestamp
        waypoints = self._waypoints
        next_waypt = waypoints[0]
        throttle_output = 0.
        steer_output = 0.
        brake_output = 0.

        self.vars.create_var('v_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            wps_vehRef = self.map_coord_2_Car_coord(x, y, yaw, waypoints)
            wps_vehRef_x = wps_vehRef[0, :]
            wps_vehRef_y = wps_vehRef[1, :]

            # fit a 3rd order polynomial to the waypoints
            coeffs = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)

            CarRef_x = CarRef_y = CarRef_yaw = 0.0
            cte = np.polyval(coeffs, CarRef_x) - CarRef_y
            yaw_err = CarRef_yaw - np.arctan(coeffs[1])

            speed_err = v_desired - v

            #### MPC ####

            next_traj, rollout = self.mpc.step(state, ip, world)
            if (next_traj is None):
                self.set_throttle(0)
                self.set_steer(0)
                self.set_brake(1)
                return
            # print ("nt: ", next_traj)
            # print ("speed: ", next_traj[0][0])
            throttle_output = (next_traj[0][0] - v) / 0.1
            if throttle_output < 0:
                brake_output = min(1.0, -throttle_output)

            steer_output = next_traj[0][1]# (next_traj[0][1] / self.trajgen.max_delta)

            # # compute the optimal trajectory
            # mpc_solution = self.mpc.Solve(state, coeffs)

            # steer_output = mpc_solution[0] # This should be in dregrees since I used degrees before I sent it to the MPC
            # throttle_output = mpc_solution[1]
            # brake_output = mpc_solution[2]

            if (type(throttle_output) == float):
                self.set_throttle(throttle_output)
            else:
                self.set_throttle(throttle_output.item())  # in percent (0 to 1)
            if (type(steer_output) == float):
                self.set_steer(steer_output)
            else:
                self.set_steer(steer_output.item())        # in rad (-1.22 to 1.22)
            if (type(brake_output) == float):
                self.set_brake(brake_output)        # in percent (0 to 1)
            else:
                self.set_brake(brake_output.item())

        self._prev_timestamp = self._current_timestamp
        self.vars.v_previous = v  # Store forward speed to be used in next step
