# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
test_data = np.load("odt_test_data/000.npy")
# array_fft = np.fft.fftshift(np.fft.fft2(test_data))
array_fft = np.fft.fftshift(np.fft.fft2(test_data))

# %%
# show data
plt.imshow(np.log(np.abs(array_fft)), cmap="gray")

# %%
