import cv2

import numpy as np

import scipy.signal

import matplotlib.pyplot as plt
  
    
def filterImage_singleColourBand(image, kernel):
    """This function assumes a 2D ndarray - a single colour band"""
    return scipy.signal.convolve2d(image, kernel, mode='same')

# processing functions


def filterRGBImage(image, kernel):
    """This function assumes (i x j) image is represented as an (i, j, 3) RGB ndarray, i.e. (:,:,0) = R, (:m :, 1) = G, (:, :, 2) and kernel is a 2D ndarray, size is less than (i, j).
    Returns the filtered image (i, j, 3) RGB image"""
    
    # convolve 2d the kernel with each channel
    r = scipy.signal.convolve2d(image[:,:,0], kernel, mode='same')
    g = scipy.signal.convolve2d(image[:,:,1], kernel, mode='same')
    b = scipy.signal.convolve2d(image[:,:,2], kernel, mode='same')

    return np.dstack([r, g, b]).astype('uint8')


# kernel functions



def createGaussianKernel(xWindow, yWindow, sigma):
    xWindow = int(xWindow)
    yWindow = int(yWindow)
    
    if yWindow == None:
        yWindow = xWindow
    
    xRange = np.arange(-xWindow, xWindow+1, 1)
    yRange = np.arange(-yWindow, yWindow+1, 1) 
    
    x, y = np.meshgrid(xRange, yRange)
    
    # Gives an (x*y) ndarray = product of 1D gaussians
    Z = gaussian(x, 0, sigma) * gaussian(y, 0, sigma)
    
    return Z / np.sum(Z)


def createLinearKernel(kernelSize):
    kernelSize = int(kernelSize)
    t = 1 - np.abs(np.linspace(-1, 1, kernelSize))
    
    kernel = t.reshape(kernelSize, 1) * t.reshape(1, kernelSize)
    kernel /= kernel.sum()
    
    return kernel
    


# Image file util functions




def defaultReadImageFile(imageFileLocation):
    """Uses OpenCV cv2 to read an image file"""
    return cv2.imread(imageFileLocation)


def readImageFileRGB(imageFileLocation):
    
    """This function takes a (i, j, 3) BGR ndarray as read by opencv, and returns an (i, j, 3) RGB ndarray"""
    image = cv2.imread(imageFileLocation)
    
    # Re-stack to RGB
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]

    return np.dstack([r, g, b])


# Math utils


def gaussian(x, mu, sigma):
    """Returns a Gaussian function with mean mu and variance sigma**2 as a np.narray.
    If input is N-D, N > 1, funciton returns a N-D symmetric Gaussian"""
    x = x - mu
    const = (1 / (np.sqrt(2 * np.pi) * sigma))
    x = const * np.exp(-x**2 / (sigma**2))
    return x
    
    
    
    
# Test utils


image = readImageFileRGB("ship-at-sea.jpg")

gaussianKernel = createGaussianKernel(8, 8, 2)

gaussianFilteredImage = filterRGBImage(image, gaussianKernel)

linearKernel = createLinearKernel(10)

linearFilteredImage = filterRGBImage(image, linearKernel)

# Plot image, kernel and result for each kernel in a grid
plt.subplot(2,3,1)
plt.imshow(image)
plt.subplot(2,3,2)
plt.imshow(gaussianKernel)
plt.subplot(2,3,3)
plt.imshow(gaussianFilteredImage)

plt.subplot(2,3,4)
plt.imshow(image)
plt.subplot(2,3,5)
plt.imshow(linearKernel)
plt.subplot(2,3,6)
plt.imshow(linearFilteredImage)

plt.show()


