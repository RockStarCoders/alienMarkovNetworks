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

from scipy import signal
from scipy.misc import imread

from skimage import feature, color, exposure


import matplotlib.pyplot as plt

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
def createHistogramOfOrientedGradientFeatures(image, numOrientations, cellForm, cellsPerBlock, visualiseHOG, smoothImage):
    # Assume given image is RGB numpy n-array.
    # wraps scikit-image HOG function.  We convert an input image to 8-bit greyscale
    image = color.rgb2gray(image)
    
    hogResult = feature.hog(image, numOrientations, cellForm, cellsPerBlock, visualiseHOG, smoothImage)
    
    return hogResult


def createImageTextons():
    # see http://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/CRF_TextonBoost_ECCV2006.pdf
    print "Finish me!"
    
    
# Some util functions

def getDifferenceOfGradient():
    # See http://www.eng.utah.edu/~bresee/compvision/files/MalikBLS.pdf
    print "Finish me!"


def getXGradient():
    print "Finish me!"


def getYGradient():
    print "Finish me!"

    
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

def createRadialSymmetricFilters():
    # Difference of gaussian or laplacian of gaussian
    # Assume a 2D function, so need bivariate Gaussians
    print "Finish me!"

def createOddSymmetricFilters():
    # See Malik paper
    print "Finish me!"


# util methods for 1D gaussian derivatives
def gaussian(x, mu, sigma):
    x = x - mu
    x = (1 / (sqrt(2 * pi) * sigma)) * exp(-x**2 / (2*(sigma**2)))
    return x
    
def gaussianFirstDerivative(x, mu, sigma):
    x = x - mu
    x = (-x / (sqrt(2 * pi) * (sigma**3) ) ) * exp(-x**2 / (2*sigma**2))
    return x  

def gaussianSecondDerivative(x, mu, sigma):
    x = x - mu
    x = (-1 / (sqrt(2 * pi) * sigma**3 ) ) * exp(-x**2 / 2*sigma**2) * (1 - (x**2 / sigma**2) )
    return x


# See Malik, Countour and Texture Analysis paper - specific form of gaussians - C, l and sigma
def gaussFilter_f1(x, y, C, l, sigma):
    print "Finish me!"
    
def gaussFilter_f2_calc(x, y, C, l, sigma):
    print "Finish me!"

def gaussFilter_f2_from_f1(f1_result):
    print "Hilbert transform.... test me!"
    
    # What does hilbert2 do?
    h2 = signal.hilbert2(f1_result)
    
    # is this the Y axis?
    h1 = signal.hilbert(f1_result, 0, 1)    
    return [[h1] , [h2]]



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


# Visualisation functions

def plot1dRGBImageHistogram(rgbFrequencies, histRange):
    
    numberBins = np.shape(histRange)[0] -1

    print "\nRed histogram:\n", rgbFrequencies[0]
    print "\nGreen histogram:\n", rgbFrequencies[1]
    print "\nBlue histogram:\n", rgbFrequencies[2]
    print "\nHistogram bins:\n", histRange
    
    fig, ax = plt.subplots()

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
    plt.imshow(image, cmap=plt.cm.get_cmap('grey'))
    plt.title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 0.02))

    plt.subplot(122).set_axis_off()
    plt.imshow(hog_image_rescaled, cmap=plt.cm.get_cmap('grey'))
    plt.title('Histogram of Oriented Gradients')
    plt.show()


image = readImageFileRGB("../ship-at-sea.jpg");
# hist, range = create1dRGBColourHistogram(image, 8)
# plot1dRGBImageHistogram(hist, range)

# numBins = 2
# freqs, rangeEdges = create3dRGBColourHistogram(image, numBins)
# print "Number of bins for 3D histogram = " , numBins
# for r in range(0,numBins):
#     for g in range(0,numBins):
#         for b in range(0,numBins):
#             print "\tColour count (bin_" + str(r+1) + ", bin_" + str(g+1) + ", bin_" + str(b+1) + ")" , freqs[r,g,b]

hogFeature, hogImage = createHistogramOfOrientedGradientFeatures(image, 8, (8,8), (2,2), True, True)

plotHOGResult(image, hogImage)

