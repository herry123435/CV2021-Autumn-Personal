import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """
    gp = []
    gp.append(input_image)
    mid_image = input_image

    for i in range(level):
        mid_image = utils.down_sampling(mid_image)
        gp.append(mid_image)

    return gp


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    lp = []

    for i in range(len(gaussian_pyramid) - 1):
        lp.append(utils.safe_subtract(gaussian_pyramid[i], utils.up_sampling(gaussian_pyramid[i+1])))
    lp.append(gaussian_pyramid[len(gaussian_pyramid) - 1])

    return lp

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """

    gp1 = gaussian_pyramid(image1, level)
    gp2 = gaussian_pyramid(image2, level)
    gpm = gaussian_pyramid(mask, level)

    lp1 = laplacian_pyramid(gp1)
    lp2 = laplacian_pyramid(gp2)

    bp = [] #blended pyramid
    for pix1, pix2, msk in zip(lp1, lp2, gpm):
        bp_l = pix2 * (msk / 255.0) + pix1 * ((255.0 - msk) / 255.0) #bp_layer
        bp.append(bp_l)

    res = bp[len(bp) - 1]
    for i in range(len(bp) - 2, -1, -1):
        res = utils.safe_add(utils.up_sampling(res), bp[i])
        res[res > 255] = 255
        res[res < 0] = 0

    return res


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3


    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
