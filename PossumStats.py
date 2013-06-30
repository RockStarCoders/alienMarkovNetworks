import numpy as np

from scipy.misc import factorial as factorial

import pomio


def imageCountPerClass(msrcDataLocation):
    
    classes = pomio.msrc_classLabels
    totalClasses = np.size(classes)
    
    classImageCount = np.arange(0,totalClasses)
    
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    
    print "\t*Imported MSRC image data using pomio.py::" , np.shape(msrcImages)
    
    for classIdx in range(0,totalClasses):
        
        classValue = classes[classIdx]
        
        imageCountForClass = 0;
        
        for imageIdx in range(0,np.size(msrcImages)):
            
            image = msrcImages[imageIdx]
            imageGroundTruth = image.m_gt
            
            if classIdx in imageGroundTruth:
                
                imageCountForClass = imageCountForClass + 1
            
        # add total count to the class count result
        print "Class = " + str(classValue), ", image count=" + str(imageCountForClass)
        
        classImageCount[classIdx] = int(imageCountForClass)
    
    result = np.array([classes, classImageCount])
    
    print "Image count per class::\n" , result
    
    return result
    

def pixelCountPerClass(msrcDataLocation):
    classes = pomio.msrc_classLabels
    totalClasses = np.size(classes)
    
    classImageCount = np.arange(0,totalClasses)
    
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    
    print "\t*Imported MSRC image data using pomio.py::" , np.shape(msrcImages)
    
    totalPixels = 0
    
    for classIdx in range(0,totalClasses):
        
        classValue = classes[classIdx]
        
        pixelCountForClass = 0;
        
        for imageIdx in range(0,np.size(msrcImages)):
            
            # can we use a matrix operator to get a count of values in a np.ndarray?
            # a[(25 < a) & (a < 100)].size
            image = msrcImages[imageIdx]
            imageGroundTruth = image.m_gt
            
            pixelCountForClass = pixelCountForClass + imageGroundTruth[(imageGroundTruth == classIdx)].size
            
        # add total count to the class count result
        print "Class = " + str(classValue), ", pixel count=" + str(pixelCountForClass)
        
        classImageCount[classIdx] = int(pixelCountForClass)
    
    result = np.array([classes, classImageCount])
    
    # count up all the pixels
    for imageIdx in range(0,np.size(msrcImages)):
        totalPixels = totalPixels + msrcImages[imageIdx].m_gt.size
    
    
    print "\nTotal pixels in dataset=" + str(totalPixels)
    print "Sum of class pixel counts=" + str(np.sum(result[1].astype('uint')))
    print "\nPixel count per class::\n" , result
    print "\nPixel count difference = " + str(totalPixels - np.sum(result[1].astype('uint'))) 
    
    return result
    

def areAllMsrcGroundTruthImagesTheSameSize(msrcDataLocation):
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    size = int(msrcImages[0].m_gt.size)
    
    equalSize = True
    
    for imageIdx in range(1, np.size(msrcImages)):
        imageSize = int(msrcImages[imageIdx].m_gt.size)
        if not size == imageSize:
            equalSize = False
            print "Image#" + str(imageIdx+1) + " doesn't match first size (" + str(size) + "), imageSize=" + str(imageSize)
    
    return equalSize


def getUniqueImageSizes(msrcDataLocation):
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    uniqueImageSizes = np.array([])
    firstSize = msrcImages[0].m_gt.size
    
    
    uniqueImageSizes = np.append(uniqueImageSizes, firstSize)
    
    for imageIdx in range(1, np.size(msrcImages)):
        
        imageSize = msrcImages[imageIdx].m_gt.size
        
        if not imageSize in uniqueImageSizes:
            uniqueImageSizes = np.append([uniqueImageSizes] , imageSize)
    
    return uniqueImageSizes


def classDistributionPerPixel(msrcData):
    # Pixel-location specific class frequency count.  Need to think about how best to normalise image size and perform count.
    print "Finish me!"
    

def getAreNeighbors(index1, index2, hood):
    
    assert hood == 4 or hood == 8, "hood value should be 4 or 8 to define a 4-neighbourhood or 8-neighbourhood"
    
    xDiff = np.abs(index1[0] - index2[0])
    yDiff = np.abs(index1[1] - index2[1])
    
    assert not (xDiff == 0 and yDiff == 0), "Input index values match::" + str(index1) + " = " + str(index2)
    
    neighbours = False
        
    if hood == 4:
        if (xDiff == 1 and yDiff == 0):
            neighbours = True
            
        elif (xDiff == 0 and yDiff == 1):
            neighbours = True
            
    elif hood == 8:
        if (np.abs(xDiff) == 1 and yDiff == 0):
            neighbours = True
        elif (xDiff == 0 and np.abs(yDiff) == 1):
            neighbours = True
        elif (np.abs(xDiff) ==1 and np.abs(yDiff) == 1):
            neighbours = True
        
    return neighbours

    
def neighbourClassImageCount(msrcDataLocation, outputFileLocation):
    # for each non-matching class pair, count images where they are neighbours
    classes = pomio.msrc_classLabels
    totalClasses = np.size(classes)
    
    msrcImages = pomio.msrc_loadImages(msrcDataLocation)
    print "*Imported MSRC image data using pomio.py::" , np.shape(msrcImages)
    
#     numberOfPairs = factorial(totalClasses) / (factorial(totalClasses -2) * factorial(2))
    
    result = None
    
    for startClassIdx in range(0, totalClasses):
        
        for endClassIdx in range(0, totalClasses):
            
            if (not startClassIdx == endClassIdx) and (endClassIdx > startClassIdx):
                # we have a class pair
                startClassValue = classes[startClassIdx]
                endClassValue = classes[endClassIdx]
                
                neighbourCount = 0
                
                # loop over images, increase count when an image includes startClass-endClass neighbours
                for imageIdx in range(0,np.size(msrcImages)):
                    
                    imageGroundTruth = msrcImages[imageIdx].m_gt
                    
                    # Get binary representation of image for each class
                    startClassChannel = (imageGroundTruth == startClassIdx)
                    startClassChannel = startClassChannel.astype('int')
                    
                    endClassChannel = (imageGroundTruth == endClassIdx)
                    endClassChannel= endClassChannel.astype('int')
                    
                    # get the difference (do not expect pixel to be in both classes!)
                    classDifference = startClassChannel - endClassChannel
                    
                    # get x and y gradient
                    dx_classDiff, dy_classDiff = np.gradient(classDifference)
                     
                    # if absolute value of x or y gradient = 2, neighbour pixels 
                    if(np.any(dx_classDiff == 2) or (np.any(dy_classDiff == 2))):
                        neighbourCount = neighbourCount + 1
                    
                print "Pair:[" + str(startClassValue) + ", " + str(endClassValue) +"] count = " + str(neighbourCount)
                    
                if result == None:
                    result = np.array([ startClassValue, endClassValue, neighbourCount])
                else:
                    result = np.vstack([ result, [ startClassValue, endClassValue, str(neighbourCount)] ] )
                        
    print "Shape of result data::", np.shape(result)
    print "Class pair image count result::",  result
    np.savetxt(outputFileLocation, result, delimiter=",", fmt="%s")
    
    
# Simple tests
msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

imageCountPerClass(msrcData)

pixelCountPerClass(msrcData)

# print "Are images in the MSRC ground truth dataset all the same size?", areAllMsrcGroundTruthImagesTheSameSize(msrcData)

# print "MSRC ground truth dataset contains images with sizes: " + str(getUniqueImageSizes(msrcData))

# print "Are (0,0) and (0,1) 4-neighbours?" + str(getAreNeighbors([0, 0], [0,1], 4))
# print "Are (0,0) and (0,2) 4-neighbours?" + str(getAreNeighbors([0, 0], [0,2], 4))
# print "Are (0,0) and (1,1) 4-neighbours?" + str(getAreNeighbors([0, 0], [1,1], 4))
# print "Are (0,0) and (1,0) 4-neighbours?" + str(getAreNeighbors([0, 0], [1,0], 4))
# print "Are (213,32) and (212,33) 4-neighbours?" + str(getAreNeighbors([213, 32], [212,33], 4))
# print "\nAre (0,0) and (0,1) 8-neighbours?" + str(getAreNeighbors([0, 0], [0,1], 8))
# print "Are (0,0) and (0,2) 8-neighbours?" + str(getAreNeighbors([0, 0], [0,2], 8))
# print "Are (0,0) and (1,1) 8-neighbours?" + str(getAreNeighbors([0, 0], [1,1], 8))
# print "Are (0,0) and (1,0) 8-neighbours?" + str(getAreNeighbors([0, 0], [1,0], 8))
# print "Are (213,32) and (212,33) 8-neighbours?" + str(getAreNeighbors([213, 32], [212,33], 8))

# outputFile = "/home/amb/dev/mrf/msrcStats/classPairImageCounts.csv"
# neighbourClassImageCount(msrcData, outputFile)
