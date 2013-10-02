#!/usr/bin/env python

# evaluation of classifier performance
# assumptions
# calculate performance at pixel level - no superpixel refs
# discount any and all void labels in ground truth; count as incorrect if predicted
# images are the same size :)
# assume that the indexes align i.e. idx=1 refers to the same class

import pomio, FeatureGenerator, SuperPixels, SuperPixelClassifier

import numpy as np

import argparse
    

def evaluateFromFile(evalFile, sourceData):
    evalData = None
    
    evalData = pomio.readEvaluationListFromCsv(evalFile)
    
    assert evalData != None , "Exception reading evaluation data from " + str(evalFile)

    print "\nINFO: Eval file list = " + str(evalFile)
    print "INFO: Source data = " + str(sourceData)
    print "\nINFO 1st element in eval result::" , evalData[0]

    results = []

    # for each eval pair (prediction labels and ground truth labels) do pixel count
    
    predictDir = "/home/amb/dev/mrf/data/classifierResults"
    for idx in range(0, len(evalData)):
    
        print "\n\tINFO: Input eval data" , len(evalData[idx]) , "\n\t\t" , evalData[idx] , "\n\t\t" , evalData[idx][0] , "\n\t\t" , evalData[idx][1]
    
        predictFile = evalData[idx][0]
        gtFile = evalData[idx][1]
        
        print "\tINFO: predict file = " + str(predictFile)
        print "\tINFO: ground truth = " + str(gtFile)
        
        gt = loadReferenceGroundTruthLabels(sourceData, gtFile)
        
        predict = loadPredictionImageLabels(predictDir + "/" + predictFile)
        
        result = evaluatePrediction(predict, gt)
        results.append(result.append(gtFile))

    # Results
    print "Processed", len(results) , "evaluation results"
    for idx in range(0, len(results)):
        print "\tResult#1: " , result[idx]
    
    print "Processing complete."


# functions to read predict and ground truth files
def evaluatePrediction(predictLabels, gtLabels):
    print np.shape(predictLabels)
    print np.shape(gtLabels)
    
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
    
    
    print "\tTotal pixels =\t" , allPixels
    print "\tVOID pixels  =\t" , voidGtPixels
    print "\tValid GT pixels =\t" , validGtPixels
    print "\tCorrect pixels =\t" , correctPixels
    print "\tIncorrect pixels=\t" , incorrectPixels
    print "Pecentage accuracy = " + str( float(correctPixels) / float(validGtPixels) * 100.0 ) + str("%")
    
    return [correctPixels, validGtPixels, voidGtPixels, allPixels]

    

def loadReferenceGroundTruthLabels(sourceData, imgName):

    print "\tLoading ground truth labels from " + str(sourceData) + " directory"
    # should be only one image
    gtFile = str(sourceData) + "/GroundTruth/" + str(imgName)
    
    if "_GT" in imgName:
        imgName = imgName.replace("_GT" , "")

    print "\tGT file = " + str(imgName)    
    gtImgLabels = pomio.msrc_loadImages(sourceData , ["Images/" + imgName] )[0].m_gt

    return gtImgLabels



def loadPredictionImageLabels(predictImgLabelsFile):
    # use pomio to read in the labels file
    print "\n\tPredicted labels @ " + str(predictImgLabelsFile)
    
    predictLabels = None
    if predictImgLabelsFile.endswith(".csv"):
        predictLabels = pomio.readMatFromCSV(predictImg)
    elif predictImgLabelsFile.endswith(".pkl"):
        predictLabels = pomio.unpickleObject(predictImgLabelsFile)
    else:
        print "Couldn't load" , predictImgLabelsFile
    
    assert predictLabels != None, "Exception trying to load prediction labels from " + str(predictImgLabelsFile)
    
    return predictLabels



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate predicted class labels against ground truth image labels')
    
    parser.add_argument('evalFile', type=str, action='store', \
                            help='full filename of evaluation file which specifies a comma-separated list of predictions+ground truth pairs')
    parser.add_argument('sourceData', type=str, action='store', \
                            help='Path to source data directory for reference data (assumed to be same structure as MSRC data)')
    args = parser.parse_args()

    evalFile = args.evalFile
    sourceData = args.sourceData
    
    evaluateFromFile(evalFile, sourceData)




def test():
    # Create classifier
    classifierName = "/home/amb/dev/mrf/classifiers/randomForest/superpixel/randyForest_superPixel_maxDepth15_0.6Data.pkl"
    classifier = pomio.unpickleObject(classifierName)
    carFile = "7_3_s.bmp"
    msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

    car = pomio.msrc_loadImages(msrcData , [ "Images/" + carFile ] )[0]
    groundTruth = car.m_gt
    
    mask = SuperPixels.getSuperPixels_SLIC(car.m_img, 400, 10)[1].m_labels
    
    spLabels = SuperPixelClassifier.predictSuperPixelLabels(classifier, car.m_img)[0]
    
    prediction = SuperPixelClassifier.getSuperPixelLabelledImage(car.m_img, mask, spLabels)
    
    # save prediction to file
    pomio.writeMatToCSV(prediction, "/home/amb/dev/eval/test/predict/testPrediction1.labels")
    
    results = evaluatePrediction(prediction, groundTruth)
    
    print "\nINFO: Car test eval results::\n\t" , results
    
    #print "\tNow do a check of ground truth vs ground truth::" , evaluatePrediction(groundTruth, groundTruth)
    #print "\tNow do a check of prediction vs prediction::" , evaluatePrediction(prediction, prediction)


def test2():
    # Simple test to read in a evaluation file and display results
    evalFile = "/home/amb/dev/mrf/eval/testEvalFile.csv"
    msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"
    
    evalResult = evaluateFromFile(evalFile, msrcData)    
    
    print "Test2 evaluation result::\n" + str(evalResult)
    
    
