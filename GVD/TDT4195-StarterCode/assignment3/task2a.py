import numpy as np
import skimage
import utils
import pathlib
from skimage import io


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8

    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    # You can also define other helper functions
    # 1. Computing the normalized histogram
    i_w = im.shape[0]
    i_h = im.shape[1]
    i_histo = np.zeros(256)

    for i in range(i_w):
        for j in range(i_h):
            value = im[i][j]
            i_histo[value] += 1

    norm_histo = np.divide(i_histo, i_w * i_h)

    # 2. Computing the cumulative sums
    cumu_sum = np.zeros(256)
    cumu_sum[0] = norm_histo[0]
    for k in range(1, 256):
        cumu_sum[k] = norm_histo[k] + cumu_sum[k - 1]

    # 3. Computing the cumulative means
    cumu_mean = np.zeros(256)
    cumu_mean[0] = norm_histo[0] * i_histo[0]
    for l in range(1, 256):
        cumu_mean[l] = norm_histo[l] * l + cumu_mean[l - 1]

    # 4. Computing global mean
    global_mean = cumu_mean[-1]

    # 5. Computing between-class variance term
    bc_var = np.zeros(256)
    for i in range(256):
        mg = global_mean
        p1 = cumu_sum[i]
        m_i = cumu_mean[i]

        if p1 == 0 or p1 == 1:
            bc_var[i] = 0
        else:
            bc_var[i] = ((mg * p1 - m_i) ** 2) / (p1 * (1 - p1))

    # 6. Obtain Otsu-threshold (Value of k where between-class variance is max)
    otsu_threshold = np.argmax(bc_var)

    return otsu_threshold
    ### END YOUR CODE HERE ###

if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)


