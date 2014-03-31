#!/usr/bin/env python

# evaluation of classifier performance
# assumptions
# calculate performance at pixel level - no superpixel refs
# discount any and all void labels in ground truth; count as incorrect if predicted
# images are the same size :)
# assume that the indexes align i.e. idx=1 refers to the same class

#
# NOTE: we are in a land where void==0 and really is not used.  Usually convert back to starting at 0
#

import argparse

parser = argparse.ArgumentParser(description='Evaluate predicted class labels against ground truth image labels')

parser.add_argument('evalFile', type=str, action='store', \
                        help='CSV file listing predictions+ground truth pairs')
parser.add_argument('sourceData', type=str, action='store', \
                        help='Path to source data directory for reference data (assumed to be same structure as MSRC data)')
parser.add_argument('predictData', type=str, action='store', \
                        help='Parent path for relative filenames for predictions')

args = parser.parse_args()


from skimage import io
import sys
import pomio, PossumStats, FeatureGenerator, SuperPixels, SuperPixelClassifier
import numpy as np
import pandas    

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

    # Tells us what the elements of the results are
    headers = [ "numCorrectPixels" , "numberValidGroundTruthPixels" , "numberVoidGroundTruthPixels" , "numberPixelsInImage" ]
    
    results = None
    confMat = None

    # for each eval pair (prediction labels and ground truth labels) do pixel count
    
    for idx in range(0, len(evalData)):
    
        predictFile = evalData[idx][0]
        gtFile = evalData[idx][1]
        
        gt = loadReferenceGroundTruthLabels(sourceData, gtFile)
        predict = loadPredictionImageLabels(predictDir + predictFile)
        
        result = evaluatePrediction(predict, gt, gtFile)
        cmat = evaluateConfusionMatrix(predict, gt)

        if idx == 0:
          results = result
          confMat = cmat
        else:
          results += result
          confMat += cmat

    # Aggregate results
    perClass = evaluateClassPerformance( confMat )

    print "Processed total of ", len(evalData) , "predictions:"
    print "  Average accuracy per pixel: ", 100.0 * results[0] / float( results[1] )
    print "  Average accuracy per class: ", perClass.mean()
    print "  Accuracy per class: "
    print pandas.DataFrame( perClass.reshape((1,len(perClass))), \
                              columns=pomio.getClasses()[1:] ).to_string()
    print ""
    print "  Confusion matrix (row=gt, col=predicted): "
    print pandas.DataFrame( confMat, columns=pomio.getClasses()[1:], \
                              index=pomio.getClasses()[1:] ).to_string()
    print "Processing complete."


def evaluatePrediction(predictLabels, gtLabels, gtimageName, dbg=False):
    
    assert np.shape(predictLabels) == np.shape(gtLabels) , "Predict image and ground truth image are not the same size..."

    rows = np.shape(predictLabels)[1]
    cols = np.shape(gtLabels)[0]
    
    if dbg:
      print "Evaluating image of size = [" , rows, " ," , cols, " ]"
    voidLabel = pomio.getVoidIdx()
    
    allPixels = 0
    voidGtPixels = 0
    correctPixels = 0
    incorrectPixels = 0

    # for each pixel, do a comparision of index    
    allPixels = rows * cols
    voidGtPixels    = np.count_nonzero( gtLabels == voidLabel )
    correctPixels   = np.count_nonzero( np.logical_and( gtLabels != voidLabel, predictLabels == gtLabels ) )
    validGtPixels = allPixels - voidGtPixels
    incorrectPixels = validGtPixels - correctPixels

    assert allPixels == (rows * cols) , "Total iterated pixels != (rows * cols) num pixels!"
    assert allPixels == (voidGtPixels + correctPixels + incorrectPixels) , "Some mismatch on pixel counts:: all" + str(allPixels) + " void=" + str(voidGtPixels) + " correct=" + str(correctPixels) + " incorrect=" + str(incorrectPixels)
        
    percentage = float(correctPixels) / float(validGtPixels) * 100.0
    
    if percentage == 0 or percentage == 0.0:
        print "WARNING:: " + str(gtimageName) + " accuracy is 0%"
        
    if dbg:
      print "Pecentage accuracy = " + str( float(correctPixels) / float(validGtPixels) * 100.0 ) + str("%")
    return np.array( [correctPixels, validGtPixels, voidGtPixels, allPixels], dtype=int )



def evaluateConfusionMatrix(predictedImg, gtImg):
    
    assert np.shape(predictedImg) == np.shape(gtImg) , "Predict image and ground truth image are not the same size..."
    
    numClasses = pomio.getNumClasses()
    
    confusionMatrix = np.zeros([numClasses , numClasses] , int)
    # rows are actual, cols are predicted
    for cl in range(numClasses):
      # The plus 1 is because of void!
      clMask = (gtImg == cl+1)
      # It's easy, just histogram those values
      vals = predictedImg[clMask] - 1
      assert np.all( np.logical_and( 0 <= vals, vals < numClasses+1 ) ), vals.max()
      confusionMatrix[cl,:] = np.histogram( vals, range(numClasses+1) )[0]
    
    assert confusionMatrix.sum() == np.count_nonzero( gtImg != pomio.getVoidIdx() ) 

    return confusionMatrix;
    

def evaluateClassPerformance( confusionMatrix ):
    # Returns a vector of percentage accuracies, one per class.
    denom = confusionMatrix.sum(axis=1).astype('float')
    denom[ denom == 0 ] = 1
    return 100.0 * np.diag( confusionMatrix ).astype('float') / denom


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

    evalFile = args.evalFile
    sourceData = args.sourceData
    predictData = args.predictData
    
    evaluateFromFile(evalFile, sourceData, predictData)




def test():

    classifierLocation = "/home/amb/dev/mrf/classifiers/logisticRegression/superpixel/logReg_miniMSRC.pkl"

    classifier = pomio.unpickleObject(classifierLocation)
    carFile = "7_3_s.bmp"
    msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"

    car = pomio.msrc_loadImages(msrcData , [ "Images/" + carFile ] )[0]
    groundTruth = car.m_gt
    
    mask = SuperPixels.getSuperPixels_SLIC(car.m_img, 400, 10)
    
    spLabels = SuperPixelClassifier.predictSuperPixelLabels(classifier, car.m_img,400,10,True)[0]
    
    prediction = SuperPixelClassifier.getSuperPixelLabelledImage(car.m_img, mask, spLabels)
    
    # save prediction to file
    pomio.writeMatToCSV(prediction, "/home/amb/dev/mrf/eval/testPrediction1.labels")
    
    results = evaluatePrediction(prediction, groundTruth , carFile)
    print "\nINFO: Car test eval results::\n\t" , results
    
    classResults = evaluateClassPerformance(prediction, groundTruth)
    print "\nINFO: Car test eval class results::\n\t" , classResults
    
    confusionResults = evaluateConfusionMatrix(prediction, groundTruth)
    print "\nINFO: Car test eval confusion matrix results::\n\t" , "Just sum up entries... ",  np.sum(confusionResults)
    
    #print "\tNow do a check of ground truth vs ground truth::" , evaluatePrediction(groundTruth, groundTruth)
    #print "\tNow do a check of prediction vs prediction::" , evaluatePrediction(prediction, prediction)

    
    
