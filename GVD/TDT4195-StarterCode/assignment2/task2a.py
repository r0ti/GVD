import skimage.measure
import numpy as np
import utils

def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    """newH = im.shape[0]/stride
    newW = im.shape[1]/stride
    new_im = np.zeros((int(newH), int(newW))) # Kan slenge på en 3 på slutten, men Jeg vil jo ikke at 3 skal bli 0, da får jeg vel ikke farge?
    #np.reshape(new_im, new_im.shape + (3,))
    #new_im = np.zeros(im.shape)
    print(im.shape)
    print(newW)
    print(newH)
    print(stride)
    print(new_im.shape)
    for i in range(int(newH)):
        for j in range(0, int(newW), stride):
            subArray = im[i][j:j+stride+1]
            #print(subArray)
            new_im[i][j] = np.amax(subArray)
            #print(new_im[i][j])
    #np.reshape(new_im, new_im.shape + (3,))
    newH = im.shape[0] // stride
    newW = im.shape[1] // stride
    new_im = np.zeros((int(newH), int(newW)))

    for a in range(newH):
        for b in range(newW):
            for i in range(0, im.shape[0], stride):
                for j in range(0, im.shape[1], stride):
                    subArray = im[i:i+stride+1][j:j+stride+1]
                    print(subArray)
                new_im[a][b] = subArray.max()

    np.reshape(new_im, new_im.shape + (3,))"""

    new_im = skimage.measure.block_reduce(im, (stride, stride, 1), np.max)

    return new_im
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)