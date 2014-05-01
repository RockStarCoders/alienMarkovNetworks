import numpy as np

"""
Neural Network Classifier
"""

#from amb.seg import pomio, FeatureGenerator
import pomio
import FeatureGenerator

from pybrain.tools.shortcuts import buildNetwork

from pybrain.supervised import BackpropTrainer
from pybrain.datasets.classification import ClassificationDataSet


numFeatures = 86
numClasses = pomio.getNumClasses() # no void class
voidClass = 13

class NNet:


    def __init__(self, numFeatures, numClasses, nbHidden = None):
        if nbHidden == None:
            nbHidden = np.round( (numFeatures + numClasses) / 2 , 0 ).astype('int')
        self.net = buildNetwork(numFeatures, nbHidden, numClasses, bias=True)
        self.nbFeatures = numFeatures
        self.nbClasses = numClasses
        print "\tNetwork creation complete:"
        print "\t\tinputLayer=" , self.net['in'] 
        print "\t\thiddenLayer=" , self.net['hidden0']
        print "\t\toutputLayer=" , self.net['out']
        print "\tb\tiasLayer=" , self.net['bias']

    def createTrainingSupervisedDataSet(self,msrcImages , scale , keepClassDistTrain):
        print "\tSplitting MSRC data into train, test, valid data sets."
        splitData = pomio.splitInputDataset_msrcData(msrcImages, scale, keepClassDistTrain)
        
        print "\tNow generating features for each training image."
        trainData = FeatureGenerator.processLabeledImageData(splitData[0], ignoreVoid=True)
        features = trainData[0]
        numDataPoints = np.shape(features)[0]
        numFeatures = np.shape(features)[1]
        labels = trainData[1]
        numLabels = np.size(labels) #!!error! nb unique labels, or max label
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

    def createTrainingSetFromMatrix( self, dataMat, labelsVec=None ):
        assert labelsVec==None or dataMat.shape[0] == len(labelsVec)
        #nbFtrs = dataMat.shape[1]
        #nbClasses = np.max(labelsVec) + 1
        if labelsVec != None and np.unique(labelsVec) != range(self.nbClasses):
            print 'WARNING: class labels only contain these values %s ' % (str( np.unique(labelsVec) ))
        dataSetTrain = ClassificationDataSet(self.nbFeatures, numClasses)
        for i in range(dataMat.shape[0]):
            binaryLabels = np.zeros(numClasses)
            if labelsVec != None:
                binaryLabels[labelsVec[i]] = 1
            dataSetTrain.addSample( dataMat[i,:], binaryLabels )
        return dataSetTrain

    def trainNetworkBackprop(self, dataset,maxIter):
        trainer = BackpropTrainer(self.net, dataset)
        print "\tInitialised backpropogation traininer.  Now execute until convergence::"
        trainer.trainUntilConvergence(verbose=True,maxEpochs=maxIter)
        print "\tConvergence achieved."
        #return [network , trainer ]
        
    def predict(self,features):
        nnds = self.createTrainingSetFromMatrix( features, None )
        result = self.net.activateOnDataset(nnds)
        # Turn these outputs into argmax (for each column)
        result = result.argmax( 1 )
        return result
        # if result < voidClass:
        #     return result
        # else:
        #     return result-1
    



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
    # todo: replace with features.computePixelFeatures JRS
    imageFeatures = FeatureGenerator.generatePixelFeaturesForImage(predictImage.m_img)
    
    print "\n*Using neural net to predict class label::"
    prediction = predictClass(imageFeatures, net)
    print prediction
    
