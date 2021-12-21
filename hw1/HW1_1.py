import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """

    hp = int(size[0]/2) #height pad
    wp = int(size[1]/2) #width pad
    output_image = np.empty((input_image.shape[0]+2*hp, input_image.shape[1]+2*wp, 3), object)
    for k in range(3):
        output_image[hp:input_image.shape[0]+hp,wp:input_image.shape[1]+wp,k] = input_image[:,:,k]
        for i in range(0, hp):
            for j in range(0, wp):
                output_image[i,j,k] = output_image[2*hp - i, 2*wp - j, k]
            for j in range(wp, input_image.shape[1]+wp):
                output_image[i,j,k] = output_image[2*hp - i, j, k]
            for j in range(input_image.shape[1]+wp, output_image.shape[1]):
                output_image[i,j,k] = output_image[2*hp - i, 2*(input_image.shape[1]+wp-1) - j, k]
        for i in range(hp, input_image.shape[0]+hp):
            for j in range(0,wp):
                output_image[i,j,k] = output_image[i, 2*wp - j, k]
            for j in range(input_image.shape[1]+wp, output_image.shape[1]):
                output_image[i,j,k] = output_image[i, 2*(input_image.shape[1]+wp-1) - j, k]
        for i in range(input_image.shape[0]+hp, output_image.shape[0]):
            for j in range(0, wp):
                output_image[i,j,k] = output_image[2*(input_image.shape[0]+hp-1) - i ,2*wp - j, k]
            for j in range(wp, input_image.shape[1]+wp):
                output_image[i,j,k] = output_image[2*(input_image.shape[0]+hp-1) - i, j, k]
            for j in range(input_image.shape[1]+wp, output_image.shape[1]):
                output_image[i,j,k] = output_image[2*(input_image.shape[0]+hp-1) - i, 2*(input_image.shape[1]+wp-1) - j, k]

    return output_image

def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """
    
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)

    Kernel = np.flip(Kernel)
    mid_image = reflect_padding(input_image, Kernel.shape)
    output_image = np.empty(input_image.shape, object)
    temp = []

    h = mid_image.shape[0] - Kernel.shape[0] + 1
    w = mid_image.shape[1] - Kernel.shape[1] + 1

    for k in range(3):
        for i in range(h):
            for j in range(w):
                conv = mid_image[i:i+Kernel.shape[0], j:j+Kernel.shape[1], k] * Kernel
                temp.append(np.sum(conv))
        output_image[:,:,k] = np.array(temp).reshape(input_image.shape[0], input_image.shape[1])
        temp = []

    return output_image


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")

    temp1 = []
    temp2 = []
    mid_image = reflect_padding(input_image, size)
    output_image = np.empty(input_image.shape, object)
    for k in range(3):
        for i in range(mid_image.shape[0] - size[0] + 1):
            for j in range(mid_image.shape[1] - size[1] + 1):
                temp1 = mid_image[i:i+size[0], j:j+size[1], k].flatten().tolist()
                temp1.sort()
                temp2.append(temp1[int((size[0]*size[1] - 1)/2)])
        output_image[:,:,k] = np.array(temp2).reshape(input_image.shape[0], input_image.shape[1])
        temp2 = []

    return output_image


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """

    h = int((size[0]-1)/2)
    w = int((size[1]-1)/2)
    y,x = np.ogrid[-h:h+1, -w:w+1]
    z = np.exp(-( (x*x)/(2.*sigmax*sigmax) + (y*y)/(2.*sigmay*sigmay) ))
    gk = z / np.sum(z)

    output_image = convolve(input_image, gk)

    return output_image


if __name__ == '__main__':
    #image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()