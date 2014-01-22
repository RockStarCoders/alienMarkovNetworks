#!/usr/bin/env python

# evaluation of classifier performance
# assumptions
# calculate performance at pixel level - no superpixel refs
# discount any and all void labels in ground truth; count as incorrect if predicted
# images are the same size :)
# assume that the indexes align i.e. idx=1 refers to the same class

import sys

import pomio, FeatureGenerator, SuperPixels, SuperPixelClassifier

import numpy as np

import argparse

from skimage import io
    

#logFile = open('/home/amb/dev/mrf/zeroAccuracyStatslog.txt' , 'w')
#zeroListFile = open('/home/dev/mrf/zeroAccuracyFileList.txt' , 'w');

avgAccuracyPhrase = "**Avg prediction accuracy="

def evaluateFromFile(evalFile, sourceData, predictDir):
    evalData = None
    
    evalData = pomio.readEvaluationListFromCsv(evalFile)
    
    assert evalData != None , "Exception reading evaluation data from " + str(evalFile)

    print "\nINFO: Eval file list = " + str(evalFile)
    print "INFO: Source data = " + str(sourceData)
    print "\nINFO 1st element in eval result::" , evalData[0]

    if predictDir.endswith("/") == False:
        predictDir = predictDir + "/"

    headers = [ "numCorrectPixels" , "numberValidGroundTruthPixels" , "numberVoidGroundTruthPixels" , "numberPixelsInImage" ]
    
    results = []
    results.append(headers)

    # for each eval pair (prediction labels and ground truth labels) do pixel count
    
    for idx in range(0, len(evalData)):
    
        predictFile = evalData[idx][0]
        gtFile = evalData[idx][1]
        
        gt = loadReferenceGroundTruthLabels(sourceData, gtFile)
        
        predict = loadPredictionImageLabels(predictDir + predictFile)
        
        result = evaluatePrediction(predict, gt, gtFile)
        results.append(result)

    # Aggregate results
    print "Processed total of ", len(results) , "predictions"
    
    sumAccuracy = 0.0
    sumValid = 0.0
    
    # The first entry is headers, so iterate from 1 index
    for idx in range(1, len(results)):
        sumAccuracy = sumAccuracy + results[idx][0]
        sumValid = sumValid + results[idx][1]

    avgAccuracy = (sumAccuracy / sumValid) * 100.0
    
    #logFile.close()
    print  avgAccuracyPhrase + str(avgAccuracy)
    print "Over" , len(results) , "predictions"
    
    print "Processing complete."


def evaluatePrediction(predictLabels, gtLabels, imageName):
    
    assert np.shape(predictLabels) == np.shape(gtLabels) , "Predict image and ground truth image are not the same size..."

    rows = np.shape(predictLabels)[1]
    cols = np.shape(gtLabels)[0]
    
    print "Evaluating image of size = [" , rows, " ," , cols, " ]"
    voidLabel = pomio.getVoidIdx()
    
    allPixels = 0
    voidGtPixels = 0
    correctPixels = 0
    incorrectPixels = 0

    # for each pixel, do a comparision of index    
    for r in range(0,rows):
        
        for c in range(cols):
        
            allPixels = allPixels + 1
            
            gtLabel = gtLabels[c][r]
            predictLabel = predictLabels[c][r]
                
            if gtLabel == voidLabel:
                voidGtPixels = voidGtPixels + 1
            else:
                # only compare if GT isnt void
                if (predictLabel != voidLabel) and (predictLabels[c][r] == gtLabels[c][r]):
                    correctPixels = correctPixels + 1
                else:
                    incorrectPixels = incorrectPixels + 1

    assert allPixels == (rows * cols) , "Total iterated pixels != (rows * cols) num pixels!"
    
    assert allPixels == (voidGtPixels + correctPixels + incorrectPixels) , "Some mismatch on pixel counts:: all" + str(allPixels) + " void=" + str(voidGtPixels) + " correct=" + str(correctPixels) + " incorrect=" + str(incorrectPixels)
    
    validGtPixels = allPixels - voidGtPixels
    
    percentage = float(correctPixels) / float(validGtPixels) * 100.0
    
    if percentage == 0 or percentage == 0.0:
        print "WARNING:: " + str(imageName) + " accuracy is 0%"
        
        data = "ImageName = " + str(imageName) + "\n\tTotal pixels =" + str(allPixels) + "\n\tVOID pixels  = " + str(voidGtPixels) + "\n\tCorrect pixels = " + str(correctPixels) + "\n\tIncorrect pixels=" + str(incorrectPixels) + "\n"
        
        #logFile.write(data)
        #zeroListFile.write(imageName + "\n")
        
    print "Pecentage accuracy = " + str( float(correctPixels) / float(validGtPixels) * 100.0 ) + str("%")
    return [int(correctPixels), int(validGtPixels), int(voidGtPixels), int(allPixels)]


def evaluateClassPerformance(predictedImg, gtImg):
    # need to write something that accumulates stats on a class basis
    print "Finish me!"
    
    assert np.shape(predictImg) == np.shape(gtImg) , "Predict image and ground truth image are not the same size..."
    
    print "Comparing class-level accuracy between ground truth image and predicted image."
    
    # Per class pixel counts
    actualPixelsPerClass = PossumStats.imagePixelCountPerClass(gtImg)
    predictedPixelsPerClass = PossumStats.imagePixelCountPerClass(predictedImg)

    correctPixelsPerClass = np.zeros(numClasses)
    incorrectPixelsPerClass = np.zeros(numClasses)
    
    # either count unique values in gt... or non-zero values in the actualPixelsPerClass variable...
    numberOfActualClasses = len(np.unique(gtImg));
    numberOfPredictedClasses = len(np.unique(predictedImg))
    
    imageSize = np.shape(gtImg);    
    numRows = imgSize[1]
    numCols = imgSize[0]
    
    for row in range(0, numRows):
    
        for col in range(0, numCols):
            
            correctPixelClass = gtImg[col][row]
            predPixelClass = predictedImg[col][row]
                
            if (predPixelClass == corectPixelClass):
                correctPixelsPerClass[correctPixelClass] = correctPixelsPerClass[correctPixelClass] + 1
            else:
                incorrectPixelsPerClass[predPixelClass] = incorrectPixelsPerClass[predPixelClass] + 1

    print "Completed evaluation."
    
    # summary stats
    totalCorrectPixels = np.sum(correctPixelsPerClass)
    totalCorrectClasses = np.sum(correctPixelsPerClass > 0)
    
    totalIncorrectPixels = np.sum(incorrectPixelsPerClass)
    totalIncorrectClasses = np.sum(incorrectPixelsPerClass > 0)
    
    print "Accuracy per class:"
    
    avgClassAccuracy = 0
    for idx in range(0, correctPixelsPerClass):
        accuracy = float(correctPixelsPerClass[idx]) / float(actualPixelsPerClass[idx]) * 100
        avgClassAccuracy = avgClassAccuracy + accuracy
        print "\tClass_" + str(idx) + " accuracy = " + str(accuracy) + "%"
        
    avgClassAccuracy = float(avgClassAccuracy) / float(numCorrectClasses)
    print "\tAverage class accuracy = " + str(avgClassAccuracy) + "%"

        
    print "Incorrect per class:"
    avgIncorrectClass = 0
    print "\tTotal number incorrect pixels = " + str(totalIncorrectPixels)
    for idx in range(0, incorrectPixelsPerClass):
        percentInccorect = float(incorrectPixelsPerClass(idx)) / float(totalIncorrectPixels)  * 100
        avgIncorrectClass = avgIncorrectClass + percentIncorrect
        
        print "\tClass_" + str(idx) + " accounts for " + str(precentIncorrect) + "% of incorrect pixels"
    
    return avgClassAccuracy
    

def loadReferenceGroundTruthLabels(sourceData, imgName):
    gtFile = str(sourceData) + "/GroundTruth/" + str(imgName)
    if "_GT" in imgName:
        imgName = imgName.replace("_GT" , "")

    gtImgLabels = pomio.msrc_loadImages(sourceData , ["Images/" + imgName] )[0].m_gt
    return gtImgLabels



def loadPredictionImageLabels(predictImgLabelsFile):
    # assume an image file, use pomio to convert
    predictLabels = pomio.msrc_convertRGBToLabels( io.imread(predictImgLabelsFile) )
    
    return predictLabels
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate predicted class labels against ground truth image labels')
    
    parser.add_argument('evalFile', type=str, action='store', \
                            help='CSV file listing predictions+ground truth pairs')
    parser.add_argument('sourceData', type=str, action='store', \
                            help='Path to source data directory for reference data (assumed to be same structure as MSRC data)')
    parser.add_argument('predictData', type=str, action='store', \
                            help='Parent path for relative filenames for predictions')
                            
    args = parser.parse_args()

    evalFile = args.evalFile
    sourceData = args.sourceData
    predictData = args.predictData
    
    evaluateFromFile(evalFile, sourceData, predictData)




def test():
    classifierName = "msrcFull_randForest_grid_REF.pkl"
    classifierLocation = "/home/amb/dev/mrf/classifiers/randomForest/superpixel/" + classifierName
    
    classifier = pomio.unpickleObject(classifierLocation)
    carFile = "7_3_s.bmp"
    msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

    car = pomio.msrc_loadImages(msrcData , [ "Images/" + carFile ] )[0]
    groundTruth = car.m_gt
    
    mask = SuperPixels.getSuperPixels_SLIC(car.m_img, 400, 10)
    
    spLabels = SuperPixelClassifier.predictSuperPixelLabels(classifier, car.m_img,400,10,True)[0]
    
    prediction = SuperPixelClassifier.getSuperPixelLabelledImage(car.m_img, mask, spLabels)
    
    # save prediction to file
    pomio.writeMatToCSV(prediction, "/home/amb/dev/eval/test/predict/testPrediction1.labels")
    
    results = evaluatePrediction(prediction, groundTruth)
    print "\nINFO: Car test eval results::\n\t" , results
    
    classResults = evaluateClassPerformance(prediction, groundTruth)
    print "\nINFO: Car test eval class results::\n\t" , classResults
    
    #print "\tNow do a check of ground truth vs ground truth::" , evaluatePrediction(groundTruth, groundTruth)
    #print "\tNow do a check of prediction vs prediction::" , evaluatePrediction(prediction, prediction)

    
    
