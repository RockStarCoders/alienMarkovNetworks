import scipy.io
import superPixels
import numpy as np

# This is a superpixel result
def loadISPRSResultFromMatlab( fn ):
  # a mat file, from paul S.
  matdata = scipy.io.loadmat( fn )['superpix']
  matlabels = matdata['label'][0,0] # labels start at 1 here
  matprobs  = matdata['prob'][0,0]
  # the labels are out of order for probabilities.  Paul has: impervious, bldg, car, low veg, tree, clutter
  matprobs = matprobs[:, np.array([1,2,4,5,3,6])-1]
  # currently my code relies on consecutive superpixels starting at 0
  ulabs = np.unique( matlabels )
  # replace matlabels values with renumbered labels.
  labelMap = np.zeros( (ulabs.max()+1,), dtype=int )
  for i,l in enumerate(ulabs):
    labelMap[l] = i
  matlabels = labelMap[ matlabels ]
  nodes, edges = superPixels.make_graph(matlabels) 
  spix = superPixels.SuperPixelGraph(matlabels,nodes,edges)
  # todo: I think I probably need to reduce these probs to correspond to the
  # ordinal labels in the superpixel graph.
  classProbs = matprobs[ ulabs-1, : ]
  return (spix, classProbs)

colourMap = [\
    ('Impervious surfaces' , (255, 255, 255)),
    ('Building' ,            (0, 0, 255)),
    ('Low vegetation' ,      (0, 255, 255)),
    ('Tree' ,                (0, 255, 0)),
    ('Car' ,                 (255, 255, 0)),
    ('Clutter/background' ,  (255, 0, 0))\
]

classLabels = [z[0] for z in colourMap]
