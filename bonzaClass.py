import pickle
import pomio
# Image-oriented tools for classification.

def loadObject(filename):
    filetype = '.pkl'
    if filename.endswith(filetype):
        f = open(filename, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    else:
        print "Input filename did not end in .pkl - trying filename with type appended...."
        f = open( ( str(filename)+".pkl" ), "rb")
        obj = pickle.load(f)
        f.close()
        return obj

# Features is nxd matrix
def classifyFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has %d classes - %s' % \
            ( len(classifier.classes_), str(classifier.classes_) )
    c = classifier.predict( features )
    return c

# IMPORTANT: there is a column for each CLASS, not each LABEL.  Void is not in there.
# You'll need to offset the indices by 1 to compensate.
def classProbsOfFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has %d classes - %s' % \
            ( len(classifier.classes_), str(classifier.classes_) )
    probs = classifier.predict_proba( features )
    if len(classifier.classes_) != pomio.getNumClasses():
        # Transform class probs to the correct sized matrix.
        nbClasses = pomio.getNumClasses()
        n = probs.shape[0]
        cpnew = np.zeros( (n, nbClasses) )
        for i in range( probs.shape[1] ):
            # stuff this set of probs to new label
            cpnew[:,clfr.classes_[i]] = probs[:,i] 
        probs = cpnew
        del cpnew

    assert probs.shape[1] == pomio.getNumClasses()
    return probs
