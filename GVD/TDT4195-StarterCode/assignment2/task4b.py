import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils




def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution.
        
        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)

    #np.fft.ifft2(im)


    result = np.zeros(im.shape)
    result[:kernel.shape[0], :kernel.shape[1]] = kernel
    fft_pic = (np.fft.fft2(im))
    fft_shift_pic = np.fft.fftshift(fft_pic)
    fft_kern = (np.fft.fft2(result))
    fft_shift_kern = np.fft.fftshift(fft_kern)

    out = np.abs(np.fft.ifft2(fft_shift_pic * fft_shift_kern))


    if verbose:
        # Use plt.subplot to place two or more images beside each other
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))

        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        plt.imshow(np.abs(fft_shift_kern), cmap="gray")
        plt.subplot(1, 5, 3)
        plt.imshow(np.log(np.abs(fft_shift_pic)), cmap="gray")
        plt.subplot(1, 5, 4)
        plt.imshow(np.log(np.abs(fft_shift_pic * fft_shift_kern)), cmap="gray")
        plt.subplot(1, 5, 5)
        plt.imshow(out, cmap="gray")

    ### END YOUR CODE HERE ###
        plt.show()
    return out


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
