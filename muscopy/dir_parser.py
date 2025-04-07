import os
import pathlib


def numpy_parser(dir_path):
    """Return path list of numpy arrays"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".npy"):
            files.append(os.path.join(dir_path, file))
    return files


def recurcive_numpy_parser(dir_path):
    files = []
    p = pathlib.Path(dir_path)
    for path in p.glob("**/*.npy"):
        files.append(path)
    return files


def png_parser(dir_path):
    """Return list of png"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".png"):
            files.append(os.path.join(dir_path, file))
    return files


def tiff_parser(dir_path):
    """Return list of tiff"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".tiff"):
            files.append(os.path.join(dir_path, file))
    return files


def pkl_parser(dir_path):
    """Return list of pkl"""
    files = []
    for file in os.listdir(dir_path):
        if file.endswith(".pkl"):
            files.append(os.path.join(dir_path, file))
    return files


def recurcive_pkl_parser(dir_path):
    files = []
    p = pathlib.Path(dir_path)
    for path in p.glob("**/*.pkl"):
        files.append(path)
    return files


def recurcive_file_parser(dir_path, file_name):
    files = []
    p = pathlib.Path(dir_path)
    for path in p.glob(f"**/{file_name}"):  # TODO: adapt to regular expression
        files.append(path)
    return files
