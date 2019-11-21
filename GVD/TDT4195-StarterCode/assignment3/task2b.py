import utils
import numpy as np
from skimage import io

def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """

    ### START YOUR CODE HERE ### (You can change anything inside this block)

    def h_crit(cand_px, seed_px):
        seed_intensity = im[seed_px[0]][seed_px[1]]
        cand_intensity = im[cand_px[0]][cand_px[1]]

        if abs(seed_intensity - cand_intensity) < 50:
            segmented[cand_px[0]][cand_px[1]] = True
            add_candidates(cand_px)

    def add_candidates(input_px):
        n = [input_px[0], input_px[1] - 1]
        nw = [input_px[0] - 1, input_px[1] - 1]
        ne = [input_px[0] + 1, input_px[1] - 1]
        s = [input_px[0], input_px[1] + 1]
        sw = [input_px[0] - 1, input_px[1] + 1]
        se = [input_px[0] + 1, input_px[1] + 1]
        w = [input_px[0] - 1, input_px[1]]
        e = [input_px[0] + 1, input_px[1]]
        neighbours = [n, nw, ne, s, sw, se, w, e]

        for n in neighbours:
            if n not in checked:
                candidates.append(n)
                checked.append(n)

    candidates = []
    checked = []

    segmented = np.zeros_like(im).astype(bool)

    # iterate over seed points
    for row, col in seed_points:
        checked.append([row, col])
        segmented[row, col] = True
        add_candidates([row, col])

        for c in candidates:
            h_crit(c, [row, col])

    return segmented
    ### END YOUR CODE HERE ###



if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [ # (row, column)
        [254, 138], # Seed point 1
        [253, 296], # Seed point 2
        [233, 436], # Seed point 3
        [232, 417], # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, \
        "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, \
        "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)

