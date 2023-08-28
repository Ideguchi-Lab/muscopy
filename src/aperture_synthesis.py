
try:
    import cupy as xp
    _cp = True
except ImportError:
    import numpy as xp
    _cp = False

from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np


class Synthesizer:
    def __init__(self, separation, shape):
        self.sep = separation
        self.shape = shape
        self.fft_images = None
        self.koff_locs = None
        self.koff_centers = dict()
        self.koff_labels = None

    def convert_to_fft(self):
        self.fft_images = []
        print("performing fft2 ...\n", "-"*80)
        for i in tqdm(range(len(self.images))):
            image = self.images[i]
            self.fft_images.append(
                np.fft.fftshift(np.fft.fft2(image))
            )

    def get_koff_locs(self):
        self.koff_locs = []
        assert self.fft_images is not None
        for fft_image in self.fft_images:
            fft_mag = np.abs(fft_image)
            center_loc = (fft_mag.shape[0]//2, fft_mag.shape[1]//2)

            # drop out the area around an origin point
            filter = make_circle(center_loc, self.sep,
                                 fft_mag.shape, highpass=True)
            fft_mag = fft_mag * filter

            centers = self.get_biggest_args(fft_mag, 3, self.sep)
            self.koff_locs.append(centers)

    # methods for getting k_off center
    # use k-means method?
    def get_koff_center(self, show_result):
        assert self.koff_locs is not None
        kmeans = KMeans(n_clusters=2, n_init='auto')
        kmeans.fit(self.koff_locs.reshape(2*len(self.koff_locs)))
        self.koff_labels = kmeans.labels_.reshape(len(self.koff_locs), 2)
        # calculate koff center by circle fitting
        for label in range(0, 1):
            koff_x = [self.koff_locs[i][self.koff_labels[i].tolist().index(
                label)][0] for i in range(len(self.koff_locs))]
            koff_y = [self.koff_locs[i][self.koff_labels[i].tolist().index(
                label)][1] for i in range(len(self.koff_locs))]
            koff_center, _ = circle_fitting(koff_x, koff_y)
            self.koff_centers[label] = koff_center

        # show clustering result
        if show_result:
            # center, locs around circle, labels
            return self.koff_centers, self.koff_locs, self.koff_labels

    def aperture_synthesis(self, label=0):
        synthesized = np.zeros(self.shape)
        divider = np.zeros(self.shape)
        print("synthesizing...\n", "-"*80)
        for i in tqdm(range(len(self.fft_images))):
            fft_image = self.fft_images[i]
            koff = self.koff_locs[i][self.koff_labels[i].tolist().index(label)]
            koff_shift = self._calc_shift_center(
                self.koff_centers[label], koff)
            fft_image_shift = np.roll(
                fft_image, (-koff[0], -koff[1]), axis=(0, 1))
            mask = make_circle(koff_shift, self.sep, self.shape)
            fft_image_shift = fft_image_shift * mask
            synthesized += fft_image_shift
            divider += mask

        divider[divider == 0] = 0
        synthesized /= divider
        synthesized_back = np.fft.ifftshift(synthesized)
        synthesized_back = np.fft.ifft2(synthesized_back)
        return np.abs(synthesized_back)

    @staticmethod
    def _calc_shift_center(koff, koff_center):
        return (koff_center[0] - koff[0], koff_center[1] - koff[1])

    @staticmethod
    def get_biggest_args(array, num, sep_radius):
        # get num biggest point args. sep is necessary for get isolated big points
        centers = []
        while len(centers) < num:
            argmax = np.unravel_index(np.argmax(array))
            centers.append(argmax)
            filter = make_circle(argmax, sep_radius,
                                 array.shape, highppass=True)
            array = array * filter
        return centers


def make_circle(center, radius, array_shape, highpass=False):
    """internal method. return circle filled with 1.

    Args:
        center (tuple of int): center position of the circle.
        radius (float): radius of the circle
        shape (_type_): _description_
        highpass (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: array_shape is invalid

    Returns:
        numpy.ndarray: array whose pass area is filled with 1, otherwise 0.
    """
    x = center[0]
    y = center[1]
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    elif isinstance(array_shape, tuple) & isinstance(array_shape[0], int) & (len(array_shape) == 2):
        pass
    else:
        raise ValueError()

    array = np.zeros(array_shape)
    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            if (i-x)**2 + (j-y)**2 <= radius**2:
                array[i, j] = 1

    if highpass:
        array = np.ones(array_shape) - array

    return array

# for calculating optimal mask radius


def calc_bandwidth(picture_size, NA=1.2, wavelength=532*1e-9, pixelsize=3.45*1e-6, magnification=(200/3)*5):
    if picture_size == tuple:
        if picture_size[0] == picture_size[1]:
            picture_size = picture_size[0]
        else:
            raise ValueError("picture's shape is not square!")
    assert picture_size is int
    pixelsize = pixelsize/magnification
    freq_per_pixel = 1/(pixelsize * picture_size)
    return 2*np.pi*NA/wavelength/freq_per_pixel


def circle_fitting(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    # reduced coordinates
    u = x - x_m
    v = y - y_m

    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # solve linear equation
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # results
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1 - R_1)**2)
    return (xc_1, yc_1), residu_1
