# Module for image and data i/o
import glob
import pylab
import numpy as np


# MSRC Image Segmentation database V2:
#
#   http://research.microsoft.com/en-us/projects/ObjectClassRecognition/
#
# It comes with a html doc file describing the labels rgb signatures.

msrc_classToRGB = {\
'void'     : (0     ,0       ,0     ), \
'building' : (128   ,0       ,0     ), \
'grass'    : (0     ,128     ,0     ), \
'tree'     : (128   ,128     ,0     ), \
'cow'      : (0     ,0       ,128   ), \
'horse'    : (128   ,0       ,128   ), \
'sheep'    : (0     ,128     ,128   ), \
'sky'      : (128   ,128     ,128   ), \
'mountain' : (64    ,0       ,0     ), \
'aeroplane': (192   ,0       ,0     ), \
'water'    : (64    ,128     ,0     ), \
'face'     : (192   ,128     ,0     ), \
'car'      : (64    ,0       ,128   ), \
'bicycle'  : (192   ,0       ,128   ), \
'flower'   : (64    ,128     ,128   ), \
'sign'     : (192   ,128     ,128   ), \
'bird'     : (0     ,64      ,0     ), \
'book'     : (128   ,64      ,0     ), \
'chair'    : (0     ,192     ,0     ), \
'road'     : (128   ,64      ,128   ), \
'cat'      : (0     ,192     ,128   ), \
'dog'      : (128   ,192     ,128   ), \
'body'     : (64    ,64      ,0     ), \
'boat'     : (192   ,64      ,0     )  \
}

msrc_classLabels = msrc_classToRGB.keys()

def msrc_convertRGBToLabels( imgRGB ):
    imgL = 255 * np.ones( imgRGB.shape[0:2], dtype='uint8' )
    # For each label, find matching RGB and set that value
    l = 0
    for lname,clr in msrc_classToRGB.items():
        # Get a mask of matching pixels
        msk = np.logical_and( imgRGB[:,:,0]==clr[0], \
                                  np.logical_and( imgRGB[:,:,1]==clr[1], \
                                                      imgRGB[:,:,2]==clr[2] ) )
        # Set these in the output image
        imgL[msk] = l
        l += 1
    # Check we got every pixel
    assert( not np.any( imgL == 255 ) )
    return imgL

class msrc_Image:
    'Structure containing image and ground truth from MSRC v2 data set'
    m_img = None
    m_gt  = None
    m_hq  = None
    m_imgFn = None
    m_gtFn  = None
    m_hqFn  = None

    def __init__( self,  fn, gtfn, hqfn ):
        # load the image (as numpy nd array, 8bit)
        self.m_img = pylab.imread( fn )
        self.m_gt  = msrc_convertRGBToLabels( pylab.imread( gtfn ) )
        # not necessarily hq
        try:
            self.m_hq  = msrc_convertRGBToLabels( pylab.imread( hqfn ) )
        except IOError:
            self.m_hq = None
        self.m_imgFn = fn
        self.m_gtFn  = gtfn
        self.m_hqFn  = hqfn
   
# dataSetPath is the base directory for the data set (subdirs are under this)
# Returns a list of msrc_image objects.
def msrc_loadImages( dataSetPath ):
    res = []
    # For each image file:
    for fn in glob.glob( dataSetPath + '/Images/*.bmp' ):
        # load the ground truth, convert to discrete label
        gtfn = fn.replace('Images/', 'GroundTruth/').replace('.bmp','_GT.bmp')
        hqfn = fn.replace('Images/', 'SegmentationsGTHighQuality/').replace('.bmp','_HQGT.bmp')
        # create an image object, stuff in list
        res.append( msrc_Image( fn, gtfn, hqfn ) )
        #break
    return res
