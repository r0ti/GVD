import numpy as np
import os
import matplotlib.pyplot as plt
from task2ab import save_im


def conv_trans(image):
    copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            copy[i][j] = image[image.shape[0] - i - 1][image.shape[1] - j - 1]
    return copy


def convolve_im(im, kernel):
    """ A function that convolves im with kernel
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]
    
    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    # YOUR CODE HERE

    kernel = conv_trans(kernel)
    iH = im.shape[0]
    iW = im.shape[1]

    kH = kernel.shape[0]
    kW = kernel.shape[1]

    h = kH // 2
    w = kW // 2

    newIm = np.zeros(im.shape)

    for i in range(h, iH - h):
        for j in range(w, iW - w):
            sum = 0
            for m in range(kH):
                for n in range(kW):
                    sum = sum + kernel[m][n] * im[i - h + m][j - w + n]
            newIm[i][j] = sum
    return newIm.astype(np.uint8)


if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    im = plt.imread(impath)

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    # Convolve images
    smoothed_im1 = convolve_im(im.copy(), h_a)
    smoothed_im2 = convolve_im(im, h_b)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        f"Your convolve function has to return a np.array. " +\
        f"Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert smoothed_im2.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"

    save_im("convolved_im_h_a.jpg", smoothed_im1)
    save_im("convolved_im_h_b.jpg", smoothed_im2)
