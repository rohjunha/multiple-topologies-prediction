import os


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
