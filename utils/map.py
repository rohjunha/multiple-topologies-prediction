import os

import numpy as np
import torch

from utils import rhctensor


def get_root_cache_dir():
    # TODO: custom path
    path = "~/.cache/mushr_rhc/" # params.get_str("cache", default="~/.cache/mushr_rhc/")
    path = os.path.expanduser(path)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def get_cache_dir(path):
    root = get_root_cache_dir()
    fullpath = os.path.join(root, path)

    if not os.path.isdir(fullpath):
        os.makedirs(fullpath)

    return fullpath


def get_cache_map_dir(map):
    # TODO: map path even if none
    if map is None:
        return get_cache_dir("carla_map")
    else:
        return get_cache_dir(map)


def world2map(mapdata, poses, out=None):
    if out is None:
        print("out cannot be None")
        exit(1)

    assert poses.size() == out.size()

    out[:, :] = poses
    scale = float(mapdata.resolution)

    # translation
    out[:, 0].sub_(mapdata.origin_x).mul_(1.0 / scale)
    out[:, 1].sub_(mapdata.origin_y).mul_(1.0 / scale)
    out[:, 2] += mapdata.angle

    xs = out[:, 0]
    ys = out[:, 1]

    # we need to store the x coordinates since they will be overwritten
    xs_p = xs.clone()

    out[:, 0] = xs * mapdata.angle_cos - ys * mapdata.angle_sin
    out[:, 1] = xs_p * mapdata.angle_sin + ys * mapdata.angle_cos


def world2mapnp(mapdata, poses):
    # translation
    poses[:, 0] -= mapdata.origin_x
    poses[:, 1] -= mapdata.origin_y

    # scale
    poses[:, :2] *= 1.0 / float(mapdata.resolution)

    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = mapdata.angle_cos * poses[:, 0] - mapdata.angle_sin * poses[:, 1]
    poses[:, 1] = mapdata.angle_sin * temp + mapdata.angle_cos * poses[:, 1]
    poses[:, 2] += mapdata.angle


def map2worldnp(mapdata, poses):
    # rotation
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = mapdata.angle_cos * poses[:, 0] - mapdata.angle_sin * poses[:, 1]
    poses[:, 1] = mapdata.angle_sin * temp + mapdata.angle_cos * poses[:, 1]

    # scale
    poses[:, :2] *= float(mapdata.resolution)

    # translate
    poses[:, 0] += mapdata.origin_x
    poses[:, 1] += mapdata.origin_y
    poses[:, 2] += mapdata.angle


def load_permissible_region(map):
    """
        get_map is a function that lazily gets all the mapdata
            * only use if map data is needed otherwise use cached data
    """
    path = get_cache_map_dir(map)
    perm_reg_file = os.path.join(path, "perm_region.npy")

    if os.path.isfile(perm_reg_file):
        pr = np.load(perm_reg_file)
    else:
        # map_data = map.data()
        # array_255 = map_data.reshape((map.height, map.width))
        array_255 = np.ones((128, 128))
        pr = np.zeros_like(array_255, dtype=bool)

        # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
        # With values 0: not permissible, 1: permissible
        pr[array_255 == 0] = 1
        pr = np.logical_not(pr)  # 0 is permissible, 1 is not

        np.save(perm_reg_file, pr)

    return torch.from_numpy(pr.astype(np.int)).type(rhctensor.byte_tensor())
