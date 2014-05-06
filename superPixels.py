import os
import sys
import slic
import numpy as np
import skimage
import skimage.data
from skimage.segmentation import slic, felzenszwalb, quickshift, mark_boundaries
import slic
import matplotlib.pyplot as plt
import pomio
import multiprocessing as mp

"""
Functions and Classes for generating and dealing with super-pixels
"""

# Module wraps skimage segementation functions

def displayImage(image, imgTitle, orientation):
    assert orientation == "upper" or orientation == "lower", "orientation parameter to displayImage must be \"upper\" or \"lower\"."
    plt.interactive(True)
    plt.imshow(image, origin=orientation)
    plt.title(imgTitle)
    plt.waitforbuttonpress()

def getSuperPixels_SLIC(image, nbSegments, compactness):
    return slic.slic_n(image, nbSegments, compactness)

def getSuperPixels_Graph(image):
    # See [http://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.felzenszwalb]
    # Function usage:    skimage.segmentation.felzenszwalb(image, scale=1, sigma=0.8, min_size=20)
    # Produces an oversegmentation of a multichannel (i.e. RGB) image using a fast, minimum spanning tree based clustering on the image grid. The parameter scale sets an observation level. Higher scale means less and larger segments. sigma is the diameter of a Gaussian kernel, used for smoothing the image prior to segmentation.
    # image : (width, height, 3) or (width, height) ndarray. Input image.
    # scale : float. Free parameter. Higher means larger clusters.
    # sigma : float. Width of Gaussian kernel used in preprocessing.
    # min_size : int. Minimum component size. Enforced using postprocessing.
    superPixelImage = skimage.segmentation.felzenszwalb(image, scale=25, sigma=1.0, min_size=50)
    return superPixelImage
    

def getSuperPixels_Quickshift(image):
    # See [http://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.quickshift]
    # image : (width, height, channels) ndarray.  Input image.
    # ratio : float, optional, between 0 and 1 (default 1). Balances color-space proximity and image-space proximity. Higher values give more weight to color-space.
    # kernel_size : float, optional (default 5). Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
    # max_dist : float, optional (default 10). Cut-off point for data distances. Higher means fewer clusters.
    # return_tree : bool, optional (default False). Whether to return the full segmentation hierarchy tree and distances.
    # sigma : float, optional (default 0).  Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    # convert2lab : bool, optional (default True). Whether the input should be converted to Lab colorspace prior to segmentation. For this purpose, the input is assumed to be RGB.
    # random_seed : None (default) or int, optional. Random seed used for breaking ties.
    superPixelImage = skimage.segmentation.quickshift(image)
    return superPixelImage
    

def generateImageWithSuperPixelBoundaries(image, segmentationMask):
    # Function returns an image with superpixel boundaries displayed as lines.  It is assumed that the image was the source for the segmentation mask.
    # See [http://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.mark_boundaries]
    # Function signature: skimage.segmentation.mark_boundaries(image, label_img, color=(1, 1, 0), outline_color=(0, 0, 0))
    
    superPixelImage = mark_boundaries(image, segmentationMask)
    return superPixelImage

# Note this only finds edges on a 4-grid.
def make_graph(grid):
    # get unique labels
    vertices = np.unique(grid)
 
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x/num_vertices]] for x in edges]
 
    return vertices, edges

def show_graph(grid, vertices, edges):
    # compute region centers:
    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]
    centers = dict()
    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
     
    # plot labels
    plt.imshow(grid,interpolation='None')
     
    # overlay graph:
    for edge in edges:
        plt.plot([centers[edge[0]][0],centers[edge[1]][0]],
                 [centers[edge[0]][1],centers[edge[1]][1]])
    




class SuperPixelGraph:
    def __init__(self,labels,nodes,edges):
        self.m_labels = labels
        self.m_nodes = nodes
        self.m_edges = edges
        # for now all our code relies on the superpixels being consecutive.
        assert np.all(self.m_nodes == np.arange(len(self.m_nodes)))

    def draw(self):
        show_graph(self.m_labels, self.m_nodes, self.m_edges)

    def imageFromSuperPixelData( self, data ):
        # data is an nxD matrix, where n is the number of superpixels.
        if type(data) == list or data.ndim==1:
            data = np.array( [ data ] ).transpose()
            #print data
        assert data.ndim == 2 and data.shape[0] == self.getNumSuperPixels(), \
            'dodgy data shape = %s' % data.shape
        D = data.shape[1]
        H,W = self.m_labels.shape
        # for a given region, make the data same for all pixels in that region
        res = data[ self.m_labels.ravel(), : ]
        res = res.reshape( (H,W,D) )
        if res.shape[2] == 1:
            # essentially 2D in this case
            res = res.squeeze()
        return res

    # Returns: (adjMatrix,nbAdjInvolvingVoid,nbAdj)
    def countClassAdjacencies( self, nbClasses, allSPClassLabels ):
        counts = np.zeros( ( nbClasses, nbClasses ) )
        voidLabel = pomio.getVoidIdx()
        adjVoidCount = 0
        adjCount = len(self.m_edges)

        for (ei, ej) in self.m_edges:
            ci = allSPClassLabels[ ei ]
            cj = allSPClassLabels[ ej ]
            # Not doing stats for void
            if ci != voidLabel and cj != voidLabel:
                counts[ ci, cj ] += 1
                counts[ cj, ci ] += 1
            else:
                adjVoidCount += 1
        return (counts, adjVoidCount, adjCount)

    def getNumSuperPixels( self ):
        return len(self.m_nodes)

    def getLabelImage( self ):
        return self.m_labels

def computeSuperPixelGraph( imgRGB, method, params ):
    if method == 'slic':
        nbSegments, compactness = params[0], params[1]
        labels = getSuperPixels_SLIC(imgRGB, nbSegments, compactness)
        nodes, edges = make_graph(labels) 
    else:
        raise Exception('invalid superpixel method %s' % method)
    return SuperPixelGraph(labels,nodes,edges)

def computeSuperPixelGraphMulti( imgRGBArray, method, params, nbCores=1 ):
  if nbCores>1:
    job_server = mp.Pool(nbCores)
    jres = [ job_server.apply_async( computeSuperPixelGraph, ( img, method, params ) ) \
               for img in imgRGBArray ]
    tOutSecs = 10*60 # 10 mins
    res = [ jr.get(timeout=tOutSecs) for jr in jres ]
  else:
    res = [ computeSuperPixelGraph( img, method, params ) for img in imgRGBArray ]
  return res

###################################
# tests
###################################

def testSLIC_lenaRGB(numSuperPixels, compactness):
    lenaImg = skimage.data.lena()
    
    spMask = getSuperPixels_SLIC(lenaImg, numSuperPixels, compactness)
    
    lena_superPixels_SLIC = generateImageWithSuperPixelBoundaries(lenaImg, spMask )
    displayImage(lena_superPixels_SLIC, imgTitle="Lena SLIC" , orientation="upper")
    
def testGraph_lenaRGB():
    lenaImg = skimage.data.lena()
    lena_superPixels_Graph = generateImageWithSuperPixelBoundaries(lenaImg, getSuperPixels_Graph(lenaImg) )
    displayImage(lena_superPixels_Graph, imgTitle="Lena Graph" , orientation= "upper")
    
def testQuickshift_lenaRGB():
    lenaImg = skimage.data.lena()	
    lena_superPixels_Quickshift = generateImageWithSuperPixelBoundaries(lenaImg, getSuperPixels_Quickshift(lenaImg) )
    displayImage(lena_superPixels_Quickshift, imgTitle="Lena Quickshift" , orientation="upper")


def testSuperPixelOnImage(image, superPixelAlgoName):
    if (superPixelAlgoName == "SLIC" or superPixelAlgoName == "Quickshift" or superPixelAlgoName == "Graph" ):
        print "\tINFO: Using " + str(superPixelAlgoName) + " with default settings to generate superpixel over-segmentation"
    else:
        print "\tWARN: Defaulting to SLIC algorithm with default settings to generate superpixel over-segmentation"
    
    if(superPixelAlgoName == "SLIC"):
        spMask = getSuperPixels_SLIC(image, 400, 10)
        displayImage( generateImageWithSuperPixelBoundaries(image, spMask) , imgTitle="Car SLIC" , orientation="lower" )
    elif(superPixelAlgoName == "Quickshift"):
        displayImage( generateImageWithSuperPixelBoundaries(image, getSuperPixels_Quickshift(image) ) , imgTitle="Car Quickshift" , orientation="lower" )
    elif(superPixelAlgoName == "Graph"):
        displayImage( generateImageWithSuperPixelBoundaries(image, getSuperPixels_Graph(image) ) , imgTitle="Car Graph" , orientation="lower" )



def testSLIC_broomBroomRGB(carImg):
    testSuperPixelOnImage(carImg, "SLIC")


def testGraph_broomBroomRGB(carImg):
    
    testSuperPixelOnImage(carImg, "Graph")


def testQuickshift_broomBroomRGB(carImg):
    testSuperPixelOnImage(carImg, "Quickshift")



################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    # Examples on given image
    infile = sys.argv[1]
    nbSuperPixels = int(sys.argv[2])
    superPixelCompactness = float(sys.argv[3])

    image = skimage.io.imread(infile)

    print "Oversegmentation examples will be displayed."
    
    #testSLIC_lenaRGB(int(sys.argv[2]),int(sys.argv[3]))
    testSLIC_lenaRGB(nbSuperPixels,superPixelCompactness)
    testGraph_lenaRGB()
    testQuickshift_lenaRGB()
    
    # Examples on car image (idx ) from MSRC
    print "\tOversegmentation with car image from MSRC dataset::\n"
    print "\tSLIC algo:"
    testSLIC_broomBroomRGB(image)
    
    print "\tGraph algo:"
    testGraph_broomBroomRGB(image)

    print "\tQuickshift algo:"
    testQuickshift_broomBroomRGB(image)
    
    print "Test super pixel examples complete."
