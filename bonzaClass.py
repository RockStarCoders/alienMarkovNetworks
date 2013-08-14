# Image-oriented tools for classification.

# Features is nxd matrix
def classifyFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has ', len(classifier.classes_),\
            ' - ', classifier.classes_
    c = classifier.predict( features )
    return c

def classProbsOfFeatures( features, classifier, requireAllClasses=True ):
    if requireAllClasses:
        assert classifier.classes_ == np.arange( pomio.getNumClasses() ), \
            'Error: given classifier only has ', len(classifier.classes_),\
            ' - ', classifier.classes_
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
