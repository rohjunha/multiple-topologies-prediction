import numpy as np


def get_time_horizon():
    time_horizon = -1 # params.get_float("horizon/time", default=-1.0)
    if time_horizon > 0.0:
        return time_horizon
    else:
        dist_horizon = 2.0 # params.get_float("horizon/distance", default=2.0)
        vel = 1.0 # params.get_float("trajgen/desired_speed", default=1.0)
        return dist_horizon / vel


def get_distance_horizon():
    time_horizon = -1.5 #params.get_float("horizon/time", default=-1.0)
    if time_horizon > 0.0:
        vel = 1.0 #params.get_float("trajgen/desired_speed", default=1.0)
        return time_horizon * vel
    else:
        return 3.5 # params.get_float("horizon/distance", default=2.0)

# def rotate(xy, theta):
#     # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
#     cos_theta, sin_theta = math.cos(theta), math.sin(theta)

#     return (
#         xy[0] * cos_theta - xy[1] * sin_theta,
#         xy[0] * sin_theta + xy[1] * cos_theta
#     )


# def translate(xy, offset):
#     return xy[0] + offset[0], xy[1] + offset[1]


# if __name__ == '__main__':
#     # Create the square relative to (0, 0)
#     w, h = 100, 100

#     points = [
#         (0, 0),
#         (0, h),
#         (w, h),
#         (w, 0)
#     ]

#     offset = (40000, 50000)
#     degrees = 90
#     theta = math.radians(degrees)

#     # Apply rotation, then translation to each point
#     print [translate(rotate(xy, theta), offset) for xy in points]


def do_polygons_intersect(a, b):
    """
 * Helper function to determine whether there is an intersection between the two polygons described
 * by the lists of vertices. Uses the Separating Axis Theorem
 *
 * @param a an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
 * @param b an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
 * @return true if there is any intersection between the 2 polygons, false otherwise
    """

    polygons = [a, b];
    minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

    for i in range(len(polygons)):

        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        polygon = polygons[i];
        for i1 in range(len(polygon)):

            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon);
            p1 = polygon[i1];
            p2 = polygon[i2];

            # find the line perpendicular to this edge
            normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] };

            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(len(a)):
                projected = normal['x'] * a[j][0] + normal['y'] * a[j][1];
                if (minA is None) or (projected < minA): 
                    minA = projected

                if (maxA is None) or (projected > maxA):
                    maxA = projected

            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(len(b)): 
                projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected

                if (maxB is None) or (projected > maxB):
                    maxB = projected

            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                # print("polygons don't intersect!")
                return False;

    return True