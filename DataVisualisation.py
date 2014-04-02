# Visualisation functions
""" 
Various image and feature plotting tools.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from pylab import meshgrid, meshgrid, imshow, contour, title

import numpy as np

from skimage import exposure



def plot1dRGBHistogram(rgbFrequencies, histRange):
    
    numberBins = np.shape(histRange)[0]
    
    print "\nRed histogram:\n", rgbFrequencies[0] , "\nsize=" , rgbFrequencies[0].size
    print "\nGreen histogram:\n", rgbFrequencies[1], "\nsize=" , rgbFrequencies[1].size
    print "\nBlue histogram:\n", rgbFrequencies[2], "\nsize=" , rgbFrequencies[2].size
    print "\nHistogram bins:\n", histRange, "\nsize=" , histRange.size
    print "\nNumber bins = " + str(numberBins)
    print "\nHistogram range = " , histRange
    print "Histogram bin edges::" , histRange[0 : (histRange.size-1)]
    
    plotRange = histRange[0 : (histRange.size-1)]
    
    fig = plt.figure()
    plt.subplot(1,1,1)

    # matplotlib.pyplot.bar(left, height, width=0.8, bottom=None, hold=None, **kwargs)
    # left     the x coordinates of the left sides of the bars
    # height     the heights of the bars
    
    
    width = int( (256 / (numberBins * 3.5) ) )    # the width of the bars

    plt.bar( plotRange , rgbFrequencies[0] , width, color=['red'] )
    plt.bar( plotRange+width , rgbFrequencies[1] , width, color=['green'] )
    plt.bar( plotRange+2*width , rgbFrequencies[2] , width, color=['blue'] )
    plt.axis([0,255,0,160000])
    
    plt.title("RGB Histogram")
    plt.show()


def plot1dHSVHistogram(hsvHist):
    
    hueHist = hsvHist[0]
    satHist = hsvHist[1]
    valHist = hsvHist[2]
    
    assert np.shape(hueHist[1])[0] == np.shape(satHist[1])[0] == np.shape(valHist[1])[0] , "The number of bins in the HSV channel are not equal"
    
    # matplotlib.pyplot.bar(left, height, width=0.8, bottom=None, hold=None, **kwargs)
    # left     the x coordinates of the left sides of the bars
    # height     the heights of the bars
    fig = plt.figure()
    plt.subplot(3,1,1)
    hueFreq = hueHist[0]
    hueRange = hueHist[1][0 : (hueHist[1].size-1)]
    hueWidth = hueRange[1]-hueRange[0]
    plt.bar( hueRange , hueFreq , hueWidth , color=['yellow'])
    plt.title("Hue Channel")
    plt.axis([0, 1, 0, 160000])
    # The first two values are the rows columns, the last is the index for the subplot
    plt.subplot(3,1,2)
    satFreq = satHist[0]
    satRange = satHist[1][0 : (satHist[1].size-1)]
    satWidth = satRange[1]-satRange[0]
    plt.bar( satRange, satFreq , satWidth, color=['gray'])
    plt.title("Saturation Channel")
    plt.axis([0,1,0,160000])

    plt.subplot(3,1,3)
    valFreq = valHist[0]
    valRange = valHist[1][0 : (valHist[1].size-1)]
    valWidth = valRange[1]-valRange[0]
    plt.bar(valRange, valFreq , valWidth, color=['gray'])
    plt.axis([0,1,0,160000])
    plt.title("Value Channel")
    
    fig.tight_layout()

    plt.show()

def plotHOGResult(image, hogImage):
    
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
