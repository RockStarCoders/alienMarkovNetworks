# collection of fucntions to produce feature vectors from an input image
# Plan to generate:
#    * Histogram of oriented gradient (HOG) features using scikit-image
#    * Colour histograms
#    * TextonBoost features from [http://www.vision.caltech.edu/wikis/EE148/images/8/8a/EE148_Presentation_Will.pdf]
#        |__ Laplacian of Gaussians [http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm]
#        |__ 
#    * Local binary patterns see [http://en.wikipedia.org/wiki/Local_binary_patterns]
import cv2

import numpy as np
from numpy import exp
from numpy import sqrt
from numpy import pi
from numpy import mgrid

from scipy import signal
from scipy.misc import imread
from scipy.stats import norm

import skimage
from skimage import color, feature, exposure


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import meshgrid

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

from DataVisualiation import createKernalWindowRanges, plotKernel



def create1dRGBColourHistogram(imageRGB, numberBins):
    
    # See previous code for example
    
    # check number of bins is an even number [2, 256]
    bins = np.array([2,4,8,16,32,64,128,256])
    check = np.array([numberBins])
    
    if (np.shape(np.intersect1d(bins, check))[0] == 0):
        numberBins = 32
    
    numColours = np.shape(imageRGB)[2]
    
    histograms = None
    
    if(numColours < 3 or numColours > 4):
        
        return histograms
    
    else:
        
        histograms = np.zeros([numberBins, 3]);
    
    # should have a non-null histograms matrix, get colours and normalise to [0-255]
    red = imageRGB[0]
    maxRed = float(np.max(red))
    red = (red / maxRed) * 255.0
    
    green = imageRGB[1]
    maxGreen = float(np.max(green))
    green = (green / maxGreen) * 255.0
    
    blue = imageRGB[2]
    maxBlue = np.max(green);
    blue = (blue / maxBlue) * 255.0
    
#     histogramRange = np.arange(0, 256 +1 , (256 / float(numberBins)) , dtype=int )
    histogramRange = np.arange(0, 256 +1 , (256 / numberBins) , dtype=int )
    
    redHist, redRange = np.histogram(red, histogramRange)
    
    greenHist, greenRange = np.histogram(green, histogramRange)
    blueHist, blueRange = np.histogram(blue, histogramRange)
    
    return np.array([redHist, greenHist, blueHist]) , histogramRange


    # Assume given image as numpy
    print "Finish me!"


def create3dRGBColourHistogram(imageRGB, numberBins):
    
    print "Input image shape == ", np.shape(imageRGB)
    
    bins = np.array([2,4,8,16,32,64,128,256])
    
    # fail if user-input number of bins is not a permitted value
    assert numberBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    
    print "shape of imageRGB[:,:,0] = ", np.shape(imageRGB[:,:,0])
    print "shape of imageRGB[:,:,1] = ", np.shape(imageRGB[:,:,1])
    print "shape of imageRGB[:,:,2] = ", np.shape(imageRGB[:,:,2])
    numPixels = np.shape(imageRGB[:,:,0])[0] * np.shape(imageRGB[:,:,0])[1]
    
    data = imageRGB.reshape((numPixels, 3))
    print "Shape of flattened rgb:: ", np.shape(data)
#     del(imageRGB)
    
    hist, edges = np.histogramdd( data, bins=numberBins, range=[[-0.5,255.5],[-0.5,255.5],[-0.5,255.5]] )
    
    return hist, edges


# TODO implement a HSV or HS colour histogram

# http://scikit-image.org/docs/dev/auto_examples/plot_hog.html#example-plot-hog-py


def createHistogramOfOrientedGradientFeatures(image, numOrientations, cellForm, cellsPerBlock, smoothImage):
    # Assume given image is RGB numpy n-array.
    # wraps scikit-image HOG function.  We convert an input image to 8-bit greyscale
    image = color.rgb2gray(image)
    
    hogResult = feature.hog(image, numOrientations, cellForm, cellsPerBlock, False, smoothImage)
    
    return hogResult


def createImageTextons():
    # see http://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/CRF_TextonBoost_ECCV2006.pdf
    print "Finish me!"
    

def generateFilterbankResponse(image):
    # See [Object Categorization by Learned Universal Visual Dictionary. Winn, Criminisi & Minka, 2005]
     
    # convert RGB to CIELab
    image = color.rgb2lab(image)
    image_L = image[:,:,0]
    image_a = image[:,:,1]
    image_b = image[:,:,2]
    
    # Create filters
    G1, G2, G3, LoG1, LoG2, LoG3,LoG4, Div1xG2, Div1xG3, Div1yG2, Div1yG3 = createDefaultFilterbank()
    
    # Apply filters & append result into 17D vector for each pixel as follows:
    # Name          Defn                 CIE channel
    #                                L        a        b
    # G1            N(0, 1)          yes      yes      yes
    # G2            N(0, 2)          yes      yes      yes
    # G3            N(0, 4)          yes      yes      yes
    # LoG1          Lap(N(0, 1))     yes      no       no
    # LoG2          Lap(N(0, 1))     yes      no       no
    # LoG3          Lap(N(0, 1))     yes      no       no
    # LoG4          Lap(N(0, 1))     yes      no       no
    # Div1xG2       d/dx(N(0,2))     yes      no       no
    # Div1xG3       d/dx(N(0,4))     yes      no       no
    # Div1yG2       d/dy(N(0,2))     yes      no       no
    # Div1yG3       d/dy(N(0,4))     yes      no       no
    
    responseVector = np.array([])
    
    G1response1 = signal.convolve2d(image_L, G1, mode='same')
    G1response2 = signal.convolve2d(image_a, G1, mode='same')
    G1response3 = signal.convolve2d(image_b, G1, mode='same')
    
    G2response1 = signal.convolve2d(image_L, G2, mode='same')
    G2response2 = signal.convolve2d(image_a, G2, mode='same')
    G2response3 = signal.convolve2d(image_b, G2, mode='same')
    
    G3response1 = signal.convolve2d(image_L, G3, mode='same')
    G3response2 = signal.convolve2d(image_a, G3, mode='same')
    G3response3 = signal.convolve2d(image_b, G3, mode='same')
    
    LoG1response1 = signal.convolve2d(image_L, LoG1, mode='same')
    LoG2response1 = signal.convolve2d(image_L, LoG2, mode='same')
    LoG3response1 = signal.convolve2d(image_L, LoG3, mode='same')
    LoG4response1 = signal.convolve2d(image_L, LoG4, mode='same')
    
    Div1xG2response1 = signal.convolve2d(image_L, Div1xG2, mode='same')
    Div1xG3response1 = signal.convolve2d(image_L, Div1xG3, mode='same')
    
    Div1yG2response1 = signal.convolve2d(image_L, Div1yG2, mode='same')
    Div1yG3response1 = signal.convolve2d(image_L, Div1yG3, mode='same')
    
    
    print "Finish me!"
    
    
def createDefaultFilterbank(window):
    # create filterbank as defined in [Object Categorization by Learned Universal Visual Dictionary. Winn, Criminisi & Minka, 2005]
    # Gaussians::  G1 = N(0, 1), G2 = N(0, 2), G3 = N(0, 4)
    # Laplacian of Gaussians:: LoG1 = Lap(N(0, 1)), LoG2=Lap(N(0, 2)), LoG3=Lap(N(0, 4)), LoG4=Lap(N(0, 8))
    # Derivative of Gaussian (x):: Div1xG1 = d/dx N(0,2), Div1xG2=d/dx N(0,4)
    # Derivative of Gaussian (y):  Div1yG1 = d/dy N(0,2), Div1yG2=d/dy N(0,4)
    
    window = int(window)
    
    G1 = gaussian_kernel(window, window, 1, 1)
    G2 = gaussian_kernel(window, window, 2, 2)
    G3 = gaussian_kernel(window, window, 4, 4)
    
    # see http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    LoG1 = laplacianOfGaussian_kernel(window, window, 1, 1)
    LoG2 = laplacianOfGaussian_kernel(window, window, 1, 2)
    LoG3 = laplacianOfGaussian_kernel(window, window, 1, 4)
    LoG4 = laplacianOfGaussian_kernel(window, window, 1, 8)
    
    print "Finish me... I need some validation!"
    return G1, G2, G3, G3, LoG1, LoG2, LoG3, LoG4
    
# Some util functions
    
def getGradientMagnitude(gradX, gradY):
    # magnitude of image gradient
    return np.sqrt(gradX**2 + gradY**2)

def getGradientOrientation(gradX, gradY):
    # orientation of computed gradient
    return np.arctan2(gradX, gradY)




# util methods for setting up filter bank for texton processing
    
def createStandardTextureFilterBank():
    # see http://research.microsoft.com/pubs/67408/criminisi_iccv2005.pdf
    print "Finish me!"



# util methods for gaussian_kernel derivatives

def gaussian_kernel(windowX, windowY, sigma):
    """ Returns a normalized 2D (windowX x windowY grid) Gaussian kernel array for convolution"""
    X,Y = createKernalWindowRanges(windowX, windowY, 0.1)
    
    gKernel = gaussianNormalised(X, 0, sigma) * gaussianNormalised(Y, 0, sigma)
    
    return (gKernel / np.sum(gKernel))


def laplacianOfGaussian_kernel(windowX, windowY, sigma):
    """ Returns a normalized 2D (windowX x windowY grid) Laplacian of Gaussian (LoG) kernel array for convolution"""
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, 0.1)
    
    logKernel = -1 * (1 - ( X**2 + Y**2) ) *  exp (- (X**2 + Y**2) / (2 * sigma))
    
    return (logKernel / np.sum(logKernel))


def gaussian_1xDerivative_kernel(windowX, windowY, sigma):
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, 0.1)
    
    g_dx = gaussianFirstDerivative(X, 0, sigma) * gaussianNormalised(Y, 0, sigma)
        
    return (g_dx / np.sum(g_dx))



def gaussian_1yDerivative_kernel(windowX, windowY, sigma):
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, 0.1)
    
    g_dy = gaussianFirstDerivative(Y, 0, sigma) * gaussianNormalised(X, 0, sigma)
        
    return (g_dy / np.sum(g_dy))


def gaussianNormalised(data, mu, sigma):
    data = data - mu
    g = exp ( - data**2 / (2*sigma**2) )
    return (g / np.sum(g))
    
    
def gaussianFirstDerivative(data, mu, sigma):
    data = data - mu
    g = -data * exp(-data**2 / (2*sigma**2))
    
    return (g / np.sum(g))


# File IO utils
def readImageFileRGB(imageFileLocation):    
    """This function takes a (i, j, 3) BGR ndarray as read by opencv, and returns an (i, j, 3) RGB ndarray"""
    image = cv2.imread(imageFileLocation)
    
    # Re-stack to RGB
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]

    return np.dstack([r, g, b])
    

def getGreyscaleImage(imageRGB):
    return color.rgb2gray(imageRGB)



# Some simple testing
    
image = readImageFileRGB("../ship-at-sea.jpg");
grayImage = color.rgb2gray(image)

# hist, range = create1dRGBColourHistogram(image, 8)
# plot1dRGBImageHistogram(hist, range)

# numBins = 2
# freqs, rangeEdges = create3dRGBColourHistogram(grayImage, numBins)
# print "Number of bins for 3D histogram = " , numBins
# for r in range(0,numBins):
#     for g in range(0,numBins):
#         for b in range(0,numBins):
#             print "\tColour count (bin_" + str(r+1) + ", bin_" + str(g+1) + ", bin_" + str(b+1) + ")" , freqs[r,g,b]

# hogFeature, hogImage = createHistogramOfOrientedGradientFeatures(image, 8, (8,8), (2,2), True, True)
# plotHOGResult(image, hogImage)

# createDefaultFilterbank()


# logKernel = laplacianOfGaussian_kernel(13, 13, 2, 2)
xWindow = 9
yWindow = 9
sigma = 1.4
xRange, yRange = createKernalWindowRanges(xWindow, yWindow, 0.1)
# gKernel = gaussian_kernel(xWindow, yWindow, sigma, sigma)
# print "Gaussian kernel range:: ", np.min(gKernel), np.max(gKernel)
# plotKernel(xRange, yRange, gKernel, "Gaussian kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")

logKernel = laplacianOfGaussian_kernel(xWindow, yWindow, sigma)
print "Laplacian of Gaussian kernel range:: ", np.min(logKernel), np.max(logKernel)
plotKernel(xRange, yRange, logKernel, "LOG kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")

g_dx_kernel = gaussian_1xDerivative_kernel(xWindow, yWindow, sigma)
print "Gaussian X derivative kernel range:: ", np.min(logKernel), np.max(logKernel)
plotKernel(xRange, yRange, g_dx_kernel, "G_dx kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")

g_dy_kernel = gaussian_1yDerivative_kernel(xWindow, yWindow, sigma)
print "Gaussian Y derivative kernel range:: ", np.min(logKernel), np.max(logKernel)
plotKernel(xRange, yRange, g_dy_kernel, "G_dy kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")


# logImage = signal.convolve2d(grayImage, logKernel, mode='same')
# gImage = signal.convolve2d(grayImage, gKernel, mode='same')
# plotFilterComparison(grayImage, gImage)




