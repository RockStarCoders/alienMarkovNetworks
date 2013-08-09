# collection of fucntions to produce feature vectors from an input sourceImage
# Plan to generate:
#    * Histogram of oriented gradient (HOG) features using scikit-sourceImage
#    * Colour histograms
#    * TextonBoost features from [Categorization by learned universal dictionary. Winn, Criminisi & Minka 2005]
#    * Local binary patterns see [http://en.wikipedia.org/wiki/Local_binary_patterns]

import numpy as np
from numpy import exp
import Image as pil
import matplotlib.pyplot as plt
from scipy import signal


from skimage import color, feature, io


import pomio

from DataVisualisation import createKernalWindowRanges


# Global variables to ensure consistency in classier training and prediction
increment = 1
numGradientBins = 9
numHistBins = 8

def setNumberGradientBins(numBins):
    numGradientBins = numBins

def setNumberHistogramBins(numBins):
    numHistBins = numBins

#
# Image and label array reshape utils
#

def reshapeImageLabels(msrcImage):
    groundTruth = msrcImage.m_gt
    numPixels = np.shape(groundTruth)[0] * np.shape(groundTruth)[1]
    return np.reshape(groundTruth, (numPixels))
    
def reshapeImageFeatures(imageFeatures):
    # assume (i, j, f) feature data, so feature array per pixel.  Reshape to (i*j , f) array
    numDatapoints = np.shape(imageFeatures)[0] * np.shape(imageFeatures)[1]
    numFeatures = np.shape(imageFeatures)[2]
    
    return np.reshape(imageFeatures, (numDatapoints, numFeatures))



#
#  Aggregated feature generation utils
#


def processLabeledImageData(inputMsrcImages, ignoreVoid=False, nbPerImage=None):
    
    totalImages = np.size(inputMsrcImages)
    
    allFeatures = None
    allLabels = None
    
    if ignoreVoid == True:
        print "\nVoid class pixels WILL NOT be included in the processed feature dataset"
        
        for idx in range(0, totalImages):
            
            print "\tImage" , (idx + 1) , " of " , totalImages
            
            imageResult = generateLabeledImageFeatures(inputMsrcImages[idx] , ignoreVoid=True)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1]
            
            if nbPerImage != None:
                # Randomly select a subset
                if nbPerImage > len(resultLabels):
                    print 'WARNING: Image only has %d pixels, but asking for %d samples' \
                        % (len(resultLabels),nbPerImage)
                    subset = np.arange( len(resultLabels) )
                else:
                    subset = np.random.choice( len(resultLabels), nbPerImage, replace=False )
                resultFeatures = resultFeatures[ subset, : ]
                resultLabels = resultLabels[ subset ]

            if allFeatures == None:
                allFeatures = resultFeatures
            else:
                allFeatures = np.vstack( [ allFeatures, resultFeatures])
                assert (np.shape(allFeatures)[1] == np.shape(resultFeatures)[1]) , \
                    "Check me... why are the row array features different sizes when vstacked??"
            if allLabels == None:
                allLabels = resultLabels
            else:
                allLabels = np.append( allLabels , resultLabels )
        
    else:
        print "\nVoid class pixels WILL be included in the processed feature dataset"
        for idx in range(0, totalImages):
            imageResult = generateLabeledImageFeatures(inputMsrcImages[idx], ignoreVoid=False)
            resultFeatures = imageResult[0]
            resultLabels = imageResult[1]
            
            if allFeatures == None:
                allFeatures = resultFeatures
            else:
                allFeatures = np.vstack( [ allFeatures, resultFeatures])
             
            if allLabels == None:
                allLabels = resultLabels
            else:
                allLabels = np.append( allLabels , resultLabels )
    
    return [ allFeatures , allLabels ]



def generateLabeledImageFeatures(msrcImage, ignoreVoid=False):
    """This function takes an msrcImage object and returns a 2-element list.  The first element contains an array of pixel feature values, the second contains an array of pixel class label.
    The ignoreVoid flag is used to handle the void class label; when True void class pixels are not included in result set, when False void pixels are included."""
    
    if ignoreVoid == False:
        # Just process all pixels, whether void or not
        allPixelFeatures = generatePixelFeaturesForImage(msrcImage.m_img)
        allPixelLabels = reshapeImageLabels(msrcImage)
        
        assert (np.size(allPixelLabels) == np.shape(allPixelFeatures)[0] ), ("Image pixel labels & features are different size! labelSize="\
                                                                            + str(np.size(allPixelLabels)) + ", featureSize=" + str(np.size(allPixelFeatures[0])) + "")
        return [ allPixelFeatures, allPixelLabels ]
        
    else:
        # Need to check result pixel before inclusion in result feature vector
        voidIdx = pomio.msrc_classLabels.index("void")
        nonVoidFeatures = None
        nonVoidLabels = None
        
        allPixelFeatures = generatePixelFeaturesForImage(msrcImage.m_img)
        allPixelLabels = reshapeImageLabels(msrcImage)
        
        numFeatures = np.shape(allPixelFeatures)[1]
        
        assert (np.size(allPixelLabels) == np.shape(allPixelFeatures)[0] ), ("Image pixel labels & features are different size! labelSize="\
                                                                            + str(np.size(allPixelLabels)) + ", featureSize=" + str(np.size(allPixelFeatures[0])) + "")
        # check each pixel label, add to result list iff != void
        
        
        # get boolean array of labels != void index
        nonVoidLabelCondition = (allPixelLabels != voidIdx)
        
        nonVoidLabels = allPixelLabels[nonVoidLabelCondition]
        
        nonVoidRowIdxs = np.arange(0, np.size(allPixelLabels))[nonVoidLabelCondition]
        nonVoidFeatures = allPixelFeatures[nonVoidRowIdxs]
        
        assert (np.size(nonVoidLabels) == np.shape(nonVoidFeatures)[0] ), ("Non-void pixel label & feature data are different size! Non void labelSize=" + str(np.size(nonVoidLabels)) + ", Non void featureSize=" + str(np.shape(nonVoidFeatures)[0]) + "")
        
        return [nonVoidFeatures, nonVoidLabels]



def generatePixelFeaturesForImage(rgbSourceImage):
    """This function takes an RGB image as numpy (i,j, 3) array as input and returns pixel-wise features (i * j , numFeatures) array.
    numGraidentBins is used in Historgram of Orientation (HOG) feature generation.
    numHistBins is used in the colour histogram feature generation (RGB & HSV)"""
    totalImagePixels = np.size(rgbSourceImage[:,:,0])
    
    # RGB features
    
    rgbColourValuesFeature =  createRGBColourValues(rgbSourceImage)
        
#     rgbColour1DHistogramFeatures, range =  create1dRGBColourHistogram(rgbSourceImage)
#     rgbColour1DHistogramFeatures = np.resize(rgbColour1DHistogramFeatures, (totalImagePixels, np.size(rgbColour1DHistogramFeatures[1]) ) )
    
#     rgbColour3DHistogramFeatures, range =  create3dRGBColourHistogramFeature(rgbSourceImage)
#     rgbColour3DHistogramFeatures = np.resize(rgbColour3DHistogramFeatures, (totalImagePixels, np.size(rgbColour3DHistogramFeatures[1]) ) )
    range = None
        
    # HSV features
    
    hsvColourValueFeature =  createHSVColourValues(rgbSourceImage)
    
    hsvSourceImage = color.rgb2hsv(rgbSourceImage)
        
#     hsvColour1DHistogramFeatures, range =  create1dHSVColourHistogram(hsvSourceImage) 
#     hsvColour1DHistogramFeatures = np.resize(hsvColour1DHistogramFeatures, (totalImagePixels, np.size(hsvColour1DHistogramFeatures[1]) ) )
        
    hsvColour3DHistogramFeatures, range =  create3dHSVColourHistogramFeature(hsvSourceImage)
    print 'At this point, the hist size is ', hsvColour3DHistogramFeatures.shape
    hsvColour3DHistogramFeatures = np.resize(hsvColour3DHistogramFeatures, (totalImagePixels, np.size(hsvColour3DHistogramFeatures[1]) ) )
    range = None
    
    # HOG features
    hogFeatureImage =  createHistogramOfOrientedGradientFeatures(rgbSourceImage, numGradientBins, (9,9), (3,3), True, True)
    # this is RxCxNbOrients
    hogFeatures = np.reshape( hogFeatureImage, ( rgbSourceImage[:,:,0].size, numGradientBins ) )
    #hog1Darray = None
    #hogFeatures = np.reshape(hogFeatures, (totalImagePixels , np.size(hogFeatures) / totalImagePixels) )
    
    # Local binary pattern features
    lbpFeatures =  createLocalBinaryPatternFeatures(rgbSourceImage, 4, 2, "default")
    lbpFeatures = np.reshape(lbpFeatures, (totalImagePixels , np.size(lbpFeatures) / totalImagePixels) )
   
    # Testure filter response features
    filterResponseFeatures =  createFilterbankResponse(rgbSourceImage, 15)
    filterResponseFeatures = np.reshape(filterResponseFeatures, ( totalImagePixels , np.size(filterResponseFeatures) / totalImagePixels) )
    
    # Test shapes of resized features
    assert (np.shape(rgbColourValuesFeature)[0] == totalImagePixels) , ("Number of RGB value feature rows not equal to total pixels:: " +str(np.shape(rgbColourValuesFeature)[0]) + ", " + str(totalImagePixels))
#     assert (np.shape(rgbColour1DHistogramFeatures)[0] == totalImagePixels) , ("Number of of RGB 1D histogram features:: " + str(np.shape(rgbColour1DHistogramFeatures)[0]) + ", " + str(totalImagePixels))
#     assert (np.shape(rgbColour3DHistogramFeatures)[0] == totalImagePixels) , ("Number of RGB 3D histogram feature rows not equal to total pixels:: " , np.shape(rgbColour3DHistogramFeatures)[0] + ", " + str(totalImagePixels))
    assert (np.shape(hsvColourValueFeature)[0] == totalImagePixels) , ("Number of HSV value feature rows not equal to total pixels:: " + str(np.shape(hsvColourValueFeature)[0]) + ", " + str(totalImagePixels))
#     assert (np.shape(hsvColour1DHistogramFeatures)[0] == totalImagePixels) , ("Number of HSV 1D histogram feature rows not equal to total pixels:: " + str(np.shape(hsvColour1DHistogramFeatures)[0]) + ", " + str(totalImagePixels))
    assert (np.shape(hsvColour3DHistogramFeatures)[0] == totalImagePixels) , ("Number of HSC 3D histogram feature rows not equal to total pixels:: " + str(np.shape(hsvColour3DHistogramFeatures)[0]) + ", " + str(totalImagePixels))
    assert (np.shape(hogFeatures)[0] == totalImagePixels) , ("Number of HOG features not equal to total pixels:: " + str(np.shape(hogFeatures)[0]) + ", " + str(totalImagePixels))
    assert (np.shape(lbpFeatures)[0] == totalImagePixels) , ("Number of LBP features not equal to total pixels:: " + str(np.shape(lbpFeatures)[0]) + ", " + str(totalImagePixels))
    assert (np.shape(filterResponseFeatures)[0] == totalImagePixels) , ("Number of filter response features not equal to total pixels:: " + str(np.shape(filterResponseFeatures)[0]) + ", " + str(totalImagePixels) )
    
    # Consolidate all features for image, per pixel - skip colors and 1d hists for sake of filesize!
    imageFeatures = np.hstack( [ \
                                rgbColourValuesFeature, \
#                                 rgbColour1DHistogramFeatures, \
#                                 rgbColour3DHistogramFeatures, \
#                                 hsvColourValueFeature, \
#                                 hsvColour1DHistogramFeatures, \
# prob not needed due to crf smoothing      hsvColour3DHistogramFeatures,  \
                                hogFeatures, \
# !!come back to this later                                lbpFeatures, \
                                filterResponseFeatures \
                                ] )
    # 14/06/13 Reference feature set = rgbColourValuesFeature, hsvColour3DHistogramFeatures, hogFeatures, hogFeatures, lbpFeatures & filterResponseFeatures
    
    assert (np.shape(imageFeatures)[0] == totalImagePixels) , ("Number of ImageFeatures rows not equal to total pixels:: " + str(np.shape(imageFeatures)[0]) + ", " + str(totalImagePixels))
    
    return imageFeatures.astype('float')




def createRGBColourValues(imageRGB):
    totalPixels = np.shape(imageRGB)[0] * np.shape(imageRGB)[1]
        # RGB features
    allRed = np.reshape(imageRGB[:,:,0] , (totalPixels, 1) )
    allGreen = np.reshape(imageRGB[:,:,1] , (totalPixels, 1) )
    allBlue = np.reshape(imageRGB[:,:,2] , (totalPixels, 1) ) 
    rgbColourValuesFeature = np.hstack( ( allRed, allGreen, allBlue ) )
    
    return rgbColourValuesFeature

def create1dRGBColourHistogram(imageRGB):
    # check number of bins is an even number [2, 256]
    bins = np.array([2,4,6,8,10,12,14,16,18,20,24,32,64,128,256])
    
    # fail if user-input number of bins is not a permitted value
    assert numHistBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    # Now add 1 to number of bins, so we get the correct number of bin edges
    
    numColourChannels = np.shape(imageRGB)[2]
    
    histograms = None
    
    if(not numColourChannels == 3):
        
        return histograms
    
    else:
        
        histograms = np.zeros([numHistBins, 3]);
    
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
    
    histogramRange = np.arange(0, 256 , (255 / numHistBins) , dtype=int )
    
    redHist, redRange = np.histogram(red, histogramRange)
    greenHist, greenRange = np.histogram(green, histogramRange)
    blueHist, blueRange = np.histogram(blue, histogramRange)
    
    return np.array([redHist, greenHist, blueHist] , dtype='float') , histogramRange


def create3dRGBColourHistogramFeature(imageRGB):
    
    bins = np.array([2,4,6,8,10,12,14,16,18,20,24,32,64,128,256])
    
    # fail if user-input number of bins is not a permitted value
    assert numHistBins in bins, ("User specified number of bins is not one of the permitted values:: " + str(bins))
    
    numPixels = np.shape(imageRGB[:,:,0])[0] * np.shape(imageRGB[:,:,0])[1]
    numColourChannels = 3
    data = imageRGB.reshape((numPixels, numColourChannels))
    
    hist, edges = np.histogramdd( data, bins=numHistBins, range=[[0,256],[0,256],[0,256]] )
    hist = hist.astype('int')
    return hist, edges


# TODO implement a HSV or HS colour histogram (polar and carteasian)
def createHSVColourValues(imageRGB):
    totalPixels = np.shape(imageRGB)[0] * np.shape(imageRGB)[1]
    
    hsvSourceImage = color.rgb2hsv(imageRGB)
    
    allHuePixels = np.reshape(hsvSourceImage[:,:,0] , (totalPixels, 1) )
    allSaturationPixels = np.reshape(hsvSourceImage[:,:,1] , (totalPixels, 1) )
    allValueBrightPixels = np.reshape(hsvSourceImage[:,:,2] , (totalPixels, 1) ) 
    
    hsvColourValueFeature = np.hstack( ( allHuePixels, allSaturationPixels, allValueBrightPixels ) )
    
    return hsvColourValueFeature

def create1dHSVColourHistogram(imageHSV):
    # http://scikit-image.org/docs/dev/api/skimage.color.html?highlight=hsv#skimage.color.rgb2hsv
    # HSV stands for hue, saturation, and value.
    # In each cylinder, the angle around the central vertical axis corresponds to hue, the distance from the axis corresponds to saturation, and the distance along the axis corresponds to value.
    # H = [0,360], S= [0,1] and V=[0,1]
    bins = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20,24,32])
    assert numHistBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    numberBinEdges = numHistBins + 1
    
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

    histRange = np.linspace(0, hueMax, numberBinEdges)
    
    hueFreq, histRange = np.histogram(imageHue, histRange)
    saturationFreq, histRange = np.histogram(imageSaturation, histRange)
    valueBrightFreq, histRange = np.histogram(imageValueBrightness, histRange)
    
#     hue = np.array([ hueFreq, histRange ] )
#     sat = np.array([ saturationFreq, histRange ] )
#     valueBright = np.array( [ valueBrightFreq, histRange] )
    
    return np.array( [ hueFreq, saturationFreq, valueBrightFreq ] , dtype='float' ), histRange


def create3dHSVColourHistogramFeature(imageHSV):
    
    bins = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20,24,32])
    
    # fail if user-input number of bins is not a permitted value
    assert numHistBins in bins, "User specified number of bins is not one of the permitted values:: " + str(bins)
    
    numPixels = imageHSV[:,:,0].size
    numColourChannels = 3
    data = imageHSV.reshape((numPixels, numColourChannels))
    
    hist, edges = np.histogramdd( data, bins=numHistBins, range=[[0,1.0],[0,1.0],[0,1.0]] )
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
    #  RETURNS: R x C x Orients array
    sourceImage = color.rgb2gray(sourceImage)
    
    hogVec = feature.hog(sourceImage, numOrientations, (8,8), (1,1), visualise=False, normalise=smoothImage)#cellForm, cellsPerBlock, visualise=visualise, normalise=smoothImage)
    hogBlocks = np.reshape( hogVec, (sourceImage.shape[0]/8, sourceImage.shape[1]/8, numOrientations) )
    assert hogBlocks.max() <= 1.0
    # resize 
    hogImg = np.zeros( ( sourceImage.shape[0], sourceImage.shape[1], numOrientations ) )
    for o in range(numOrientations):
        hogPerOrient = (255*hogBlocks[:,:,o]).astype('uint8')
        hpoAsPil = pil.fromarray( hogPerOrient, mode='L' )
        hogImg[:,:,o] = np.array( hpoAsPil.resize( \
                [z for z in reversed(sourceImage.shape)], pil.NEAREST ) )
    return hogImg

def createLocalBinaryPatternFeatures(imageRGB, orientationBins, neighbourhoodRadius, inputMethod):
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
    
    #plt.interactive(1)
    #plt.figure()
    for filterNum in range(0,numFilters):
        if filterNum == 0:
            response = signal.convolve2d(image_L, filters[filterNum], mode='same', boundary='symm')
            response = np.dstack((response, signal.convolve2d(image_a, filters[filterNum], mode='same', boundary='symm')))
            response = np.dstack((response, signal.convolve2d(image_b, filters[filterNum], mode='same', boundary='symm')))
            
        elif filterNum ==1 or filterNum==2:
            response = np.dstack((response, signal.convolve2d(image_L, filters[filterNum], mode='same', boundary='symm')))
            response = np.dstack((response, signal.convolve2d(image_a, filters[filterNum], mode='same', boundary='symm'))) 
            response = np.dstack((response, signal.convolve2d(image_b, filters[filterNum], mode='same', boundary='symm')))
        
        else:
            response = np.dstack((response, signal.convolve2d(image_L, filters[filterNum], mode='same', boundary='symm')))
        #plt.imshow( response[:,:,-1] )
        #plt.set_cmap('gray')
        #plt.waitforbuttonpress()
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

# HSV tests
#
# print "\nHSV 3D histogram::"
# hist, edges = create3dHSVColourHistogramFeature(color.rgb2hsv(sourceImage))
# print hist
# print edges
#
# hsvHist = create1dHSVColourHistogram(sourceImage)
# plot1dHSVHistogram(hsvHist)


# RGB tests
#
# rgbHist = create1dRGBColourHistogram(sourceImage)
# plot1dRGBHistogram(rgbHist)
#
# hist, edges = create3dRGBColourHistogramFeature(sourceImage)
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
