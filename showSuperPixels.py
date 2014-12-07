#!/usr/bin/env python

"""
Command-line utility to display super-pixel boundaries on an image
"""

import argparse

parser = argparse.ArgumentParser(description='Command-line utility to display super-pixel boundaries on an image')
parser.add_argument('imagefile', type=str, action='store', \
                        help='filename of input RGB image.')
parser.add_argument('spfile', type=str, action='store', \
                        help='filename of data file containing superpixels.  Could be pkl or matlab.  Leave empty to compute.')

parser.add_argument('--nbSuperPixels', type=int, default=400, \
                        help='Desired number of super pixels in SLIC over-segmentation')
parser.add_argument('--superPixelCompactness', type=float, default=10.0, \
                        help='Super pixel compactness parameter for SLIC')

args = parser.parse_args()

import pickle as pkl
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pomio
import slic
import superPixels
import skimage
import isprs
import amntools

imgRGB = amntools.readImage( args.imagefile )

if len(args.spfile) == 0:
  numberSuperPixels = args.nbSuperPixels
  superPixelCompactness = args.superPixelCompactness
  # Turn image into superpixels.
  spix = superPixels.computeSuperPixelGraph( imgRGB, 'slic', [numberSuperPixels,superPixelCompactness] )
elif args.spfile.endswith('.pkl'):
  superPixelInput = pomio.unpickleObject(args.spfile)
  spix = superPixelInput[0]
  classProbs = superPixelInput[1]
  colourMap = pomio.msrc_classToRGB
elif args.spfile.endswith('.mat'):
  spix, classProbs = isprs.loadISPRSResultFromMatlab( args.spfile )
  colourMap = isprs.colourMap
else:
  assert False


# Display superpixel boundaries on image.
plt.imshow( superPixels.generateImageWithSuperPixelBoundaries( imgRGB, spix.m_labels ) )
plt.show()
