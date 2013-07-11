# collection of fucntions to produce feature vectors from an input sourceImage
# Plan to generate:
#    * Histogram of oriented gradient (HOG) features using scikit-sourceImage
#    * Colour histograms
#    * TextonBoost features from [Categorization by learned universal dictionary. Winn, Criminisi & Minka 2005]
#    * Local binary patterns see [http://en.wikipedia.org/wiki/Local_binary_patterns]

import numpy as np
from numpy import exp

from scipy import signal


from skimage import color, feature, io

import DataVisualisation
from DataVisualisation import createKernalWindowRanges, plot1dHSVHistogram, plot1dRGBHistogram

increment = 1

def create1dRGBColourHistogram(imageRGB, numberBins):
    # check number of bins is an even number [2, 256]
    bins = np.array([2,4,6,8,10,12,14,16,18,20,24,32,64,128,256])
    
    # fail if user-input number of bins is not a permitted value
    assert numberBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    # Now add 1 to number of bins, so we get the correct number of bin edges
    
    numColourChannels = np.shape(imageRGB)[2]
    
    histograms = None
    
    if(not numColourChannels == 3):
        
        return histograms
    
    else:
        
        histograms = np.zeros([numberBins, 3]);
    
    # should have a non-null histograms matrix, get colours and normalise to [0-255]
    red = imageRGB[:,:,0]
    maxRed = float(np.max(red))
    if not int(np.round(maxRed,0)) == 0:
        red = (red / maxRed) * 255.0
    
    green = imageRGB[:,:,1]
    maxGreen = float(np.max(green))
    if not int(np.round(maxGreen,0)) == 0:
        green = (green / maxGreen) * 255.0
    
    blue = imageRGB[:,:,2]
    maxBlue = np.max(blue)
    if not int(np.round(maxBlue,0)) == 0:
        blue = (blue / maxBlue) * 255.0
    
    histogramRange = np.arange(0, 256 , (255 / numberBins) , dtype=int )
    
    redHist, redRange = np.histogram(red, histogramRange)
    greenHist, greenRange = np.histogram(green, histogramRange)
    blueHist, blueRange = np.histogram(blue, histogramRange)
    
    return np.array([redHist, greenHist, blueHist]) , histogramRange


def create3dRGBColourHistogramFeature(imageRGB, numberBins):
    
    bins = np.array([2,4,6,8,10,12,14,16,18,20,24,32,64,128,256])
    
    # fail if user-input number of bins is not a permitted value
    assert numberBins in bins, ("User specified number of bins is not one of the permitted values:: " + str(bins))
    
    numPixels = np.shape(imageRGB[:,:,0])[0] * np.shape(imageRGB[:,:,0])[1]
    numColourChannels = 3
    data = imageRGB.reshape((numPixels, numColourChannels))
    
    hist, edges = np.histogramdd( data, bins=numberBins, range=[[0,256],[0,256],[0,256]] )
    hist = hist.astype('int')
    return hist, edges


# TODO implement a HSV or HS colour histogram (polar and carteasian)


def create1dHSVColourHistogram(imageHSV, numberBins):
    # http://scikit-image.org/docs/dev/api/skimage.color.html?highlight=hsv#skimage.color.rgb2hsv
    # HSV stands for hue, saturation, and value.
    # In each cylinder, the angle around the central vertical axis corresponds to hue, the distance from the axis corresponds to saturation, and the distance along the axis corresponds to value.
    # H = [0,360], S= [0,1] and V=[0,1]
    bins = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20,24,32])
    assert numberBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    numberBinEdges = numberBins + 1
    
    # assume the 3D array is in H, S, V order
    numColourChannels = np.shape(imageHSV)[2]
    
    histograms = None
    
    if(not numColourChannels == 3):
        return histograms
    else:
        histograms = np.zeros([numberBinEdges, 3]);
    
    # should have a non-null histograms matrix, get channels
    hueMax = 1.0
    saturationMax = 1.0
    valueBrightMax = 1.0
    
    # Need to slice and dice the result from the n,n,3 np array correctly....
    imageHue = imageHSV[:,:,0]
    imageSaturation = imageHSV[:,:,1]
    imageValueBrightness = imageHSV[:,:,2]

    hueHistogramRange = np.linspace(0, hueMax, numberBinEdges)
    saturationHistogramRange = np.linspace(0, saturationMax, numberBinEdges)
    valueBrightHistogramRange = np.linspace(0, valueBrightMax, numberBinEdges)
    
    hueFreq, hueRange = np.histogram(imageHue, hueHistogramRange)
    saturationFreq, saturationRange = np.histogram(imageSaturation, saturationHistogramRange)
    valueBrightFreq, valueBrightRange = np.histogram(imageValueBrightness, valueBrightHistogramRange)
    
    hue = np.array([ hueFreq, hueRange ] )
    sat = np.array([ saturationFreq, saturationRange ] )
    valueBright = np.array( [ valueBrightFreq, valueBrightRange] )
    
    return [ hue, sat, valueBright ]


def create3dHSVColourHistogramFeature(imageHSV, numberBins):
    
    bins = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20,24,32])
    
    # fail if user-input number of bins is not a permitted value
    assert numberBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    
    numPixels = np.shape(imageHSV[:,:,0])[0] * np.shape(imageHSV[:,:,0])[1]
    numColourChannels = 3
    data = imageHSV.reshape((numPixels, numColourChannels))
    
    hist, edges = np.histogramdd( data, bins=numberBins, range=[[0,1.0],[0,1.0],[0,1.0]] )
    hist = hist.astype('int')
    return hist, edges


# TODO Fix up cartesian conversion of HSV HSxyV :)

def createCIELab1DHistogram(sourceImage):
    print "Finish me!"

def createCIEDLab3DHistogram():
    print "Finish me!"


def createHistogramOfOrientedGradientFeatures(sourceImage, numOrientations, cellForm, cellsPerBlock, visualise, smoothImage):
    # Assume given sourceImage is RGB numpy n-array.
    # wraps scikit-sourceImage HOG function.  We convert an input sourceImage to 8-bit grayscale
    sourceImage = color.rgb2gray(sourceImage)
    
    return feature.hog(sourceImage, numOrientations, cellForm, cellsPerBlock, visualise=visualise, normalise=smoothImage)
    


def createLocalBinaryPatternFeatures(imageRGB,orientationBins, neighbourhoodRadius, inputMethod):
    """Returns (i, j) array of Local Binary Pattern values for (i, j) input sourceImage, using scikit-sourceImage.feature.local_binary_pattern."""
    # See [http://scikit-sourceImage.org/docs/dev/api/skimage.feature.html#local-binary-pattern]
    
    grayImage = getGrayscaleImage(imageRGB)
    methods = [ "default", "ror",  "uniform", "var"]
    
    assert inputMethod in methods, "Local binary patterns input method value = " + str(inputMethod) + ".  Not one of permitted values: " + str(methods)
    
    lbpImage = feature.local_binary_pattern(grayImage, orientationBins, neighbourhoodRadius, method=inputMethod) #(sourceImage, P, R, method='default')
    
    return lbpImage



def createImageTextons():
    # see http://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/CRF_TextonBoost_ECCV2006.pdf
    print "Finish me!"
    

def createFilterbankResponse(sourceImage, window):
    # See [Object Categorization by Learned Universal Visual Dictionary. Winn, Criminisi & Minka, 2005]
    
    # convert RGB to CIELab
    sourceImage = color.rgb2lab(sourceImage)
    image_L = sourceImage[:,:,0]
    image_a = sourceImage[:,:,1]
    image_b = sourceImage[:,:,2]
    
    # Create filters - G1, G2, G3, LoG1, LoG2, LoG3,LoG4, dx_G2, dx_G3, dy_G2, dy_G3
    filters = createDefaultFilterbank(window)
    numFilters = np.shape(filters)[0]
#     print "Total number of default filters = " + str(numFilters) + ", from shape=" + str(np.shape(filters))
    
    # Apply filters & append result into 17D vector for each pixel as follows:
    # Name          Defn                 CIE channel
    #                                L        a        b
    # G1            N(0, 1)          yes      yes      yes    1
    # G2            N(0, 2)          yes      yes      yes    2
    # G3            N(0, 4)          yes      yes      yes    3
    # LoG1          Lap(N(0, 1))     yes      no       no     4
    # LoG2          Lap(N(0, 1))     yes      no       no     5
    # LoG3          Lap(N(0, 1))     yes      no       no     6
    # LoG4          Lap(N(0, 1))     yes      no       no     7
    # Div1xG2       d/dx(N(0,2))     yes      no       no     8
    # Div1xG3       d/dx(N(0,4))     yes      no       no     9
    # Div1yG2       d/dy(N(0,2))     yes      no       no     10
    # Div1yG3       d/dy(N(0,4))     yes      no       no     11
    response = np.array([])
    
    for filterNum in range(0,numFilters):
        
        if filterNum == 0:
            response = signal.convolve2d(image_L, filters[filterNum], mode='same')
            response = np.dstack((response, signal.convolve2d(image_a, filters[filterNum], mode='same')))
            response = np.dstack((response, signal.convolve2d(image_b, filters[filterNum], mode='same')))
            
        elif filterNum ==1 or filterNum==2:
            response = np.dstack((response, signal.convolve2d(image_L, filters[filterNum], mode='same')))
            response = np.dstack((response, signal.convolve2d(image_a, filters[filterNum], mode='same'))) 
            response = np.dstack((response, signal.convolve2d(image_b, filters[filterNum], mode='same')))
        
        else:
            response = np.dstack((response, signal.convolve2d(image_L, filters[filterNum], mode='same')))
        
#     print "Size of response data = " + str(np.shape(response))     
    return response

# util methods for setting up filter bank for texton processing
    
def createDefaultFilterbank(window):
    """ Returns a (11, 9, 9) ndarray filterbank as defined in [Object Categorization by Learned Universal Visual Dictionary. Winn, Criminisi & Minka, 2005]"""
    # Gaussians::  G1 = N(0, 1), G2 = N(0, 2), G3 = N(0, 4)
    # Laplacian of Gaussians:: LoG1 = Lap(N(0, 1)), LoG2=Lap(N(0, 2)), LoG3=Lap(N(0, 4)), LoG4=Lap(N(0, 8))
    # Derivative of Gaussian (x):: Div1xG1 = d/dx N(0,2), Div1xG2=d/dx N(0,4)
    # Derivative of Gaussian (y):  Div1yG1 = d/dy N(0,2), Div1yG2=d/dy N(0,4)
    
    G1 = gaussian_kernel(window, window, 1)
    G2 = gaussian_kernel(window, window, 2)
    G3 = gaussian_kernel(window, window, 4)
    
    # see http://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    LoG1 = laplacianOfGaussian_kernel(window, window, 1)
    LoG2 = laplacianOfGaussian_kernel(window, window, 2)
    LoG3 = laplacianOfGaussian_kernel(window, window, 4)
    LoG4 = laplacianOfGaussian_kernel(window, window, 8)
    
    dx_G1 = gaussian_1xDerivative_kernel(window, window, 2)
    dx_G2 = gaussian_1xDerivative_kernel(window, window, 4)
    
    dy_G1 = gaussian_1yDerivative_kernel(window, window, 2)
    dy_G2 = gaussian_1yDerivative_kernel(window, window, 4)
    
    return np.array([G1, G2, G3, LoG1, LoG2, LoG3, LoG4, dx_G1, dx_G2, dy_G1, dy_G2])
    
# Some util functions


def getGradientMagnitude(gradX, gradY):
    # magnitude of sourceImage gradient
    return np.sqrt(gradX**2 + gradY**2)

def getGradientOrientation(gradX, gradY):
    # orientation of computed gradient
    return np.arctan2(gradX, gradY)


# util methods for gaussian_kernel derivatives

def gaussian_kernel(windowX, windowY, sigma):
    """Returns a sum-normalized 2D (windowX x windowY) Gaussian kernel for convolution"""
    X,Y = createKernalWindowRanges(windowX, windowY, increment)
    
    gKernel = gaussianNormalised(X, 0, sigma) * gaussianNormalised(Y, 0, sigma)
    gSum = np.sum(gKernel)
    
    if gSum == 0:
        print "Warning gaussian_kernel:: Not normalising by sum of values, as sum = " + str(gSum)
        return (gKernel)
    else:
        return (gKernel / np.sum(gKernel))


def laplacianOfGaussian_kernel(windowX, windowY, sigma):
    """Returns a sum-normalized 2D (windowX x windowY) Laplacian of Gaussian (LoG) kernel for convolution"""
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, increment)
    
    logKernel = -1 * (1 - ( X**2 + Y**2) ) *  exp (- (X**2 + Y**2) / (2 * sigma))
    gSum = np.sum(logKernel)
    
    if gSum == 0:
        print "Warning LoG_kernel:: Not normalising by sum of values, as sum = " + str(gSum)
        return (logKernel)
    else:
        return (logKernel / np.sum(logKernel))


def gaussian_1xDerivative_kernel(windowX, windowY, sigma):
    """Returns a sum-normalized 2D (windowX x windowY) x-Derivative of Gaussian kernel for convolution"""
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, increment)
    
    g_dx_kernel = gaussianFirstDerivative(X, 0, sigma) * gaussianNormalised(Y, 0, sigma)
    gSum = np.sum(g_dx_kernel)
    
    if gSum == 0:
        print "Warning dx_g_kernel:: Not normalising by sum of values, as sum = " + str(gSum)
        return (g_dx_kernel)
    else:
        return (g_dx_kernel / np.sum(g_dx_kernel))
    

def gaussian_1yDerivative_kernel(windowX, windowY, sigma):
    """Returns a sum-normalized 2D (windowX x windowY) y-Derivative of Gaussian kernel for convolution"""
    # See [http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm]
    X, Y = createKernalWindowRanges(windowX, windowY, increment)
    
    g_dy_kernel = gaussianFirstDerivative(Y, 0, sigma) * gaussianNormalised(X, 0, sigma)
    gSum = np.sum(g_dy_kernel)
    
    if gSum == 0:
        print "Warning dy_g_kernel:: Not normalising by sum of values, as sum = " + str(gSum)
        return (g_dy_kernel)
    else:
        return (g_dy_kernel / np.sum(g_dy_kernel))


def gaussianNormalised(data, mu, sigma):
    """Returns a sum-normalized 2D (windowX x windowY) x-Derivative of Gaussian kernel for convolution"""
    data = data - mu
    g = exp ( - data**2 / (2*sigma**2) )
    gSum = np.sum(g)
    
    if gSum == 0:
        print "Warning gaussianNormalised:: Not normalising by sum of values, as sum = " + str(gSum)
        return (g)
    else:
        return (g / np.sum(g))
    
def gaussianFirstDerivative(data, mu, sigma):
    data = data - mu
    g = -data * exp(-data**2 / (2*sigma**2))
    gSum = np.sum(g)
    
    if gSum == 0:
        print "Warning gaussianFirstDerivative:: Not normalising by sum of values, as sum = " + str(gSum)
        return (g)
    else:
        return (g / np.sum(g))



# File IO utils

def readImageFileRGB(imageFileLocation):    
    """This returns a (i, j, 3) RGB ndarray"""
    
    sourceImage = io.imread(imageFileLocation)
    
    return sourceImage

def getGrayscaleImage(imageRGB):
    """This returns a (i, j) grayscale sourceImage from a (i, j, 3) RGB ndarray, using scikit-sourceImage conversion"""
    return color.rgb2gray(imageRGB)



# Some simple testing
#     
sourceImage = readImageFileRGB("ship-at-sea.jpg");
# grayImage = color.rgb2gray(sourceImage)
numberBins = 4

# HSV tests
#
# print "\nHSV 3D histogram::"
# hist, edges = create3dHSVColourHistogramFeature(color.rgb2hsv(sourceImage), numberBins)
# print hist
# print edges
#
# hsvHist = create1dHSVColourHistogram(sourceImage, numberBins)
# plot1dHSVHistogram(hsvHist)


# RGB tests
#
# rgbHist = create1dRGBColourHistogram(sourceImage, numberBins)
# plot1dRGBHistogram(rgbHist)
#
# hist, edges = create3dRGBColourHistogramFeature(sourceImage, numberBins)
# print hist
# print edges


# HOG tests
#
# hogFeature, hogImage = createHistogramOfOrientedGradientFeatures(sourceImage, 8, (8,8), (2,2), True, True)
# plotHOGResult(sourceImage, hogImage)
# 

# Gaussian kernel tests
# xWindow = 9
# yWindow = 9
# sigma = 1.4
# xRange, yRange = createKernalWindowRanges(xWindow, yWindow, increment)
# 
# g_kernel = gaussian_kernel(xWindow, yWindow, sigma)
# print "Gaussian kernel range:: ", np.min(g_kernel), np.max(g_kernel)
# plotKernel(xRange, yRange, g_kernel, "Gaussian kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")
# filteredImage = signal.convolve2d(grayImage, g_kernel, mode='same')
# plotImageComparison(grayImage, filteredImage)
#   
# log_kernel = laplacianOfGaussian_kernel(xWindow, yWindow, sigma)
# print "Laplacian of Gaussian kernel range:: ", np.min(log_kernel), np.max(log_kernel)
# plotKernel(xRange, yRange, log_kernel, "LOG kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")
# filteredImage = signal.convolve2d(grayImage, log_kernel, mode='same')
# plotImageComparison(grayImage, filteredImage)
#  
# g_dx_kernel = gaussian_1xDerivative_kernel(xWindow, yWindow, sigma)
# print "Gaussian X derivative kernel range:: ", np.min(g_dx_kernel), np.max(g_dx_kernel)
# plotKernel(xRange, yRange, g_dx_kernel, "G_dx kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")
# filteredImage = signal.convolve2d(grayImage, g_dx_kernel, mode='same')
# plotImageComparison(grayImage, filteredImage)
#  
# g_dy_kernel = gaussian_1yDerivative_kernel(xWindow, yWindow, sigma)
# print "Gaussian Y derivative kernel range:: ", np.min(g_dy_kernel), np.max(g_dy_kernel)
# plotKernel(xRange, yRange, g_dy_kernel, "G_dy kernel, sigma= + " + str(sigma) + ", window=(" + str(xWindow) + "," + str(yWindow) + ")")
# filteredImage = signal.convolve2d(grayImage, g_dy_kernel, mode='same')
# plotImageComparison(grayImage, filteredImage)
# 
# 
#  
# lbpImage = createLocalBinaryPatternFeatures(sourceImage, 6, 4, "default")
# print "Local Binary Pattern result::", lbpImage
# plotImageComparison(grayImage, filteredImage)
# 
# 
# response = createFilterbankResponse(sourceImage, xWindow)
# 
# print "\nFilter response shape=" + str(np.shape(response))  
