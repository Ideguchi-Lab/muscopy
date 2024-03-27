import os

import numpy as np

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

backend = "numpy" if not _cp else "cupy"
print("backend: " + backend)


def null_checker(path, remove=False):
    threshold = 1000
    pathlist = os.listdir(path)
    for filename in pathlist:
        if filename.endswith(".npy"):
            image = xp.load(path + filename)
            if xp.sum(image == 0) > threshold:
                print(filename)
                if remove:
                    os.remove(path + filename)
                    print("removed")


def dummy_image_generator(num=100, size=(512, 512)):
    if not os.path.exists("dummy_images"):
        os.mkdir("dummy_images")
    for i in range(num // 2):
        image = np.zeros(size)
        np.save("dummy_images/" + str(i) + ".npy", image)
    for i in range(num // 2, num):
        image = np.ones(size)
        np.save("dummy_images/" + str(i) + ".npy", image)


def check_overlap(path):
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        for j in range(i + 1, len(filelist)):
            image1 = xp.load(path + filelist[i])
            image2 = xp.load(path + filelist[j])
            dif = image1 - image2
            if xp.sum(dif) == 0:
                print(filelist[i], filelist[j])
                print("max" + filelist[i] + ": " + str(xp.max(image1)))
                print("max" + filelist[j] + ": " + str(xp.max(image2)))
                print("min" + filelist[i] + ": " + str(xp.min(image1)))
                print("min" + filelist[j] + ": " + str(xp.min(image2)))
    print("end")


if __name__ == "__main__":
    dummy_image_generator()
    null_checker("dummy_images/", remove=False)
    check_overlap("dummy_images/")
