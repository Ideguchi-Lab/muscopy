"""Helper functions to check for overlapping images in a directory.

This module provides:

- `check_overlap`: A function to check for overlapping images in a given directory.
"""

import pathlib

import jax.numpy as jnp


def _dummy_image_generator(num: int = 100, size: tuple[int, int] = (512, 512)) -> None:
    if not pathlib.Path("dummy_images").exists():
        pathlib.Path("dummy_images").mkdir()
    for i in range(num // 2):
        image = jnp.zeros(size)
        jnp.save("dummy_images/" + str(i) + ".npy", image)
    for i in range(num // 2, num):
        image = jnp.ones(size)
        jnp.save("dummy_images/" + str(i) + ".npy", image)


def check_overlap(path: str) -> None:
    """Check if there are any overlapping images in the given directory.

    Parameters
    ----------
    path : `str`
        The path to the directory containing the images.
    """
    filelist = list(pathlib.Path(path).iterdir())
    for i in range(len(filelist)):
        for j in range(i + 1, len(filelist)):
            image1 = jnp.load(str(filelist[i]))
            image2 = jnp.load(str(filelist[j]))
            dif = image1 - image2
            if jnp.sum(dif) == 0:
                print(filelist[i], filelist[j])  # noqa: T201
                print("max" + str(filelist[i]) + ": " + str(jnp.max(image1)))  # noqa: T201
                print("max" + str(filelist[j]) + ": " + str(jnp.max(image2)))  # noqa: T201
                print("min" + str(filelist[i]) + ": " + str(jnp.min(image1)))  # noqa: T201
                print("min" + str(filelist[j]) + ": " + str(jnp.min(image2)))  # noqa: T201
    print("end")  # noqa: T201


if __name__ == "__main__":
    _dummy_image_generator()
    check_overlap("dummy_images/")
