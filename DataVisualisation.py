# Visualisation functions

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from pylab import meshgrid, meshgrid, imshow, contour, title

import numpy as np

from skimage import exposure



def plot1dRGBImageHistogram(rgbFrequencies, histRange):
    
    numberBins = np.shape(histRange)[0] -1

    print "\nRed histogram:\n", rgbFrequencies[0]
    print "\nGreen histogram:\n", rgbFrequencies[1]
    print "\nBlue histogram:\n", rgbFrequencies[2]
    print "\nHistogram bins:\n", histRange
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # matplotlib.pyplot.bar(left, height, width=0.8, bottom=None, hold=None, **kwargs)
    # left     the x coordinates of the left sides of the bars
    # height     the heights of the bars

    ind = np.arange(0,256,(256 / numberBins))  # the x locations for the groups
    width = int( (256 / (numberBins * 3.5) ) )    # the width of the bars

    ax.bar( ind , rgbFrequencies[0] , width, color=['red'] )
    ax.bar( ind+width , rgbFrequencies[1] , width, color=['green'] )
    ax.bar( ind+2*width , rgbFrequencies[2] , width, color=['blue'] )

    ax.set_xticks(histRange)
    
    plt.show()


def plotHOGResult(image, hogImage):
    plt.figure(figsize=(8, 4))

    plt.subplot(121).set_axis_off()
    plt.imshow(image, cmap=plt.cm.get_cmap('gray'))
    plt.title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 0.02))

    plt.subplot(122).set_axis_off()
    plt.imshow(hog_image_rescaled, cmap=cm.get_cmap('gray'))
    plt.title('Histogram of Oriented Gradients')
    plt.show()


def plotImageComparison(sourceImage, filteredImage):
    
    plt.subplot(3,1,1)
    plt.imshow(sourceImage, cmap=cm.get_cmap('gray'))
    plt.subplot(3,1,2)
    plt.imshow(filteredImage, cmap=cm.get_cmap('gray'))
    plt.subplot(3,1,3)
    plt.imshow(filteredImage, cmap=cm.get_cmap('gray'))
    plt.imshow(filteredImage - sourceImage, cmap=cm.get_cmap('gray'))
    plt.show()
    

def plotKernel(xRange, yRange, kernel, label):
    fig = plt.figure()
    
    plt.subplot(1,2,1)
    
    # adding the Contour lines with labels
    kernelMax = np.max(kernel)
    kernelMin = np.min(kernel)
    kernelValueThreshold = 10**-1
    
    if np.abs(kernelMax - kernelMin) < kernelValueThreshold:
        print "kernel value range is < " + str(kernelValueThreshold) + ", therefore image will not contains contours"
    else:
        contourIncrement = np.round(np.abs((kernelMax - kernelMin)) * 0.1, 2)
        contourRange = np.arange(np.min(kernel), np.max(kernel) , contourIncrement )
    
        cset = contour(kernel,contourRange, linewidths=2,cmap=cm.get_cmap('set2'))
        plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)

    # latex fashion title
#     title('$z=3*e^{-(x^2+y^2)}$')
    title(label)
    
    plt.imshow(kernel,cmap=cm.get_cmap('gray')) # drawing the function

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(xRange, yRange, kernel, rstride=1, cstride=1, 
    cmap=cm.get_cmap('RdBu'),linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    plt.show()


def createKernalWindowRanges(windowX, windowY, inc):
    windowX = int(windowX)
    windowY = int(windowY)
    
    xRange = np.arange(0, windowX, inc)
    xRange = xRange - (np.floor(np.max(xRange) / 2.0).astype('uint8'))

    yRange = np.arange(0, windowY, inc)
    yRange = yRange - (np.floor(np.max(yRange) / 2.0).astype('uint8'))
    
    X,Y = meshgrid(xRange, yRange)
    return X, Y
