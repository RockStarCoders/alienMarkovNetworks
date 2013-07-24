import numpy as np

from amb.seg import pomio, FeatureGenerator

from pybrain.tools.shortcuts import buildNetwork

from pybrain.supervised import BackpropTrainer
from pybrain.datasets.classification import ClassificationDataSet


numFeatures = 86
numClasses = pomio.getNumClasses()-1 # no void class
voidClass = 13

def createDefaultNeuralNet():
    net = buildNetwork(numFeatures, np.round( (numFeatures + numClasses) / 2 , 0 ).astype('int'), numClasses, bias=True)
    print "\tNetwork creation complete:"
    print "\t\tinputLayer=" , net['in'] 
    print "\t\thiddenLayer=" , net['hidden0']
    print "\t\toutputLayer=" , net['out']
    print "\tb\tiasLayer=" , net['bias']
    return net

def createTrainingSupervisedDataSet(msrcImages , scale , keepClassDistTrain):
    print "\tSplitting MSRC data into train, test, valid data sets."
    splitData = pomio.splitInputDataset_msrcData(msrcImages, scale, keepClassDistTrain)
    
    print "\tNow generating features for each training image."
    trainData = FeatureGenerator.processLabeledImageData(splitData[0], ignoreVoid=True)
    features = trainData[0]
    numDataPoints = np.shape(features)[0]
    numFeatures = np.shape(features)[1]
    labels = trainData[1]
    numLabels = np.size(labels)
    assert numDataPoints == numLabels , "Number of feature data points and number of labels not equal!"
    
    dataSetTrain = ClassificationDataSet(numFeatures , numClasses)
    
    print "\tNow adding all data points to the ClassificationDataSet..."
    for idx in range(0,numDataPoints):
        feature = trainData[0][idx]
        label =  trainData[1][idx]
        
        binaryLabels = np.zeros(numClasses)
        # to cope with the removal of void class (idx 13)
        if label < voidClass:
            binaryLabels[label] = 1
        else:
            binaryLabels[label-1] = 1
            
        dataSetTrain.addSample(feature , binaryLabels) 
    
    print "\tAdded" , np.size(trainData) , " labeled data points to DataSet."
    return dataSetTrain


def trainNetworkBackprop(network, dataset):
    trainer = BackpropTrainer(network, dataset)
    print "\tInitialised backpropogation traininer.  Now execute until convergence::"
    trainer.trainUntilConvergence()
    print "\tConvergence achieved."
    return [network , trainer ]
    
def predictClass(features, trainedNetwork):
    
    result = trainedNetwork.activateOnDataset(features)
    if result < voidClass:
        return result
    else:
        return result-1




if __name__ == "__main__":
    
    # Create network
    print "*Creating neural net"
    net = createDefaultNeuralNet()
    
    msrcData = "/home/amb/dev/mrf/data/MSRC_ObjCategImageDatabase_v2"
    
    print "\n*Creating training dataset"
    labeledData = createTrainingSupervisedDataSet(msrcData, 0.05, True) 
    
    print "\n*Training network via backpropogation"
    trainingResult = trainNetworkBackprop(net, labeledData)
    
    net = trainingResult[0]
    trainer = trainingResult[1]
    
    predictImage = pomio.msrc_loadImages(msrcData)[1]
    
    print "\n*Read in an image from the MSRC dataset::" , np.shape(predictImage.m_img)
    imageFeatures = FeatureGenerator.generatePixelFeaturesForImage(predictImage.m_img)
    
    print "\n*Using neural net to predict class label::"
    prediction = predictClass(imageFeatures, net)
    print prediction
    
