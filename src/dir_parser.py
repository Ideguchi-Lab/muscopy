import os


def numpy_parser(dir_path):
    """return list of numpy"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".npy"):
            files.append(os.path.join(dir_path, file))
    return files


def png_parser(dir_path):
    """return list of png"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            files.append(os.path.join(dir_path, file))
    return files


def tiff_parser(dir_path):
    """return list of tiff"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".tiff"):
            files.append(os.path.join(dir_path, file))
    return files
