import numpy as np
import scipy.spatial


class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets #
        self._circle_radii = circle_radii # 1 meter
        self._weight = weight

    def convert_wp_to_obstacles(self, world, player_id=None):
        if player_id is None:
            return
        else:
            raise NotImplementedError

    def generate_obstacles(self, world, player_id=None):
        obstacles = []
        if player_id is None:
            for i in range(1, len(world.agents)):
                obstacles.append(world.box_pts[i])
        else:
            obstacles.append(world.box_pts[0])
            for i in range(1, len(world.agents)):
                if player_id != (i - 1):
                    obstacles.append(world.box_pts[i])
        return obstacles

    # Takes in a set of paths and obstacles, and returns an array
    # of bools that says whether or not each path is collision free.
    # Input: paths - a list of paths, each path of the form [[x1, x2, ...], [y1, y2, ...], [theta1, theta2, ...]].
    #        obstacles - a list of obstacles, each obstacle represented by a list of occupied points of the form [[x1, y1], [x2, y2], ...].
    def collision_check(self, paths, world, player_id=None):
        collision_check_array = np.zeros((paths.shape[0]))
        obstacles = self.generate_obstacles(world, player_id)
        for i in range(len(paths)):
            collision_free = True
            path = paths[i]
            # Iterate over the points in the path.
            for j in range(len(path)):
                circle_locations = np.zeros((len(self._circle_offsets), 2))
                for c in range(len(self._circle_offsets)):
                    circle_locations[c, 0] = path[j][0] + \
                        self._circle_offsets[c] * np.cos(path[j][2])
                    circle_locations[c, 1] = path[j][1] + \
                        self._circle_offsets[c] * np.sin(path[j][2])

                # Assumes each obstacle is approximated by a collection of points
                # of the form [x, y].
                # Here, we will iterate through the obstacle points, and check if any of
                # the obstacle points lies within any of our circles.
                # print ("CIRCLE: ", circle_locations.shape)
                for k in range(len(obstacles)):
                    collision_dists = scipy.spatial.distance.cdist(
                        obstacles[k], circle_locations)
                    collision_dists = np.subtract(
                        collision_dists, self._circle_radii)
                    collision_free = collision_free and not np.any(
                        collision_dists < 0)

                    if not collision_free:
                        break
                if not collision_free:
                    break

            if collision_free:
                collision_check_array[i] = 0
            else:
                collision_check_array[i] = 1

        return collision_check_array

    def logistic(self, x):
        '''
        A function that returns a value between 0 and 1 for x in the
        range[0, infinity] and -1 to 1 for x in the range[-infinity, infinity].
        Useful for cost functions.
        '''
        return 2.0 / (1 + np.exp(np.abs(x/10)))

    def calculate_path_score(self, path, goal_state):
        # So we have a path (multiple points) and a goal.
        # At this point the on;y thing i can think off is to compare how close you are to the GOAL at the end of the path.
        last_pt_idx = len(path[0]) - 1
        last_pt = np.array([path[0][last_pt_idx], path[1][last_pt_idx]])

        error = np.linalg.norm(np.subtract(last_pt, goal_state[:2]))

        score = self.logistic(error)

        return score

    def proximity_to_colliding_paths(self, path, colliding_path):
        assert len(path[0]) == len(colliding_path[0])
        error = np.linalg.norm(np.subtract(path, colliding_path))

        score = -1 * self.logistic(error)
        return score

    # Selects the best path in the path set, according to how closely
    # it follows the lane centerline, and how far away it is from other
    # paths that are in collision.
    # Disqualifies paths that collide with obstacles from the selection
    # process.
    # collision_check_array contains True at index i if paths[i] is collision-free,
    # otherwise it contains False.
    # TODO Implement this function.
    # Input: paths - a list of paths, each path of the form [[x1, x2, ...], [y1, y2, ...], [theta1, theta2, ...]].
    #        collision_check_array - a list of booleans that denote whether or not the corresponding path index is collision free.
    #        goal_state - a list denoting the centerline goal, in the form [x, y].

    def select_best_path_index(self, paths, collision_check_array, goal_state):
        best_index = None
        best_score = -float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.

            if collision_check_array[i]:
                # Compute the "distance from centerline" score.
                # The centerline goal is given by goal_state.
                # The exact choice of objective function is up to you.
                score = self.calculate_path_score(paths[i], goal_state)

                # Compute the "proximity to other colliding paths" score and
                # add it to the "distance from centerline" score.
                # The exact choice of objective function is up to you.
                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not collision_check_array[j]:
                            score += self._weight * \
                                self.proximity_to_colliding_paths(
                                    paths[i], paths[j])

            # Handle the case of colliding paths.
            else:
                score = -float('Inf')

            if score > best_score:
                best_score = score
                best_index = i

        return best_index
