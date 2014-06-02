#!/usr/bin/env python
'''
Display features in the given binary file.  Mainly for debugging.

see testFeatures.py
'''

import argparse

parser = argparse.ArgumentParser(description='Show features in binary file')

parser.add_argument('ftrs', type=str, action='store', \
                        help='filename of pkl or csv features data')
parser.add_argument('--labs', type=str, action='store', default=None,\
                        help='filename of pkl or csv training labels data.  Optional.')
parser.add_argument('--nshow', type=int, default=3,\
                      help='Max number of features to show at once on scatter plot grid')
parser.add_argument('--nstart', type=int, default=0,\
                      help='Index of feature to start at')

args = parser.parse_args()

import amntools
import matplotlib.pyplot as plt
import pomio

plt.interactive(1)

# Load the features and labels
if args.ftrs.endswith('.pkl'):
    ftrs = pomio.unpickleObject( args.ftrs )
else:
    ftrs = pomio.readMatFromCSV( args.ftrs )
N = ftrs.shape[0]
D = ftrs.shape[1]
print '%d feature vectors of dimensionality = %d' % (N,D)

if args.labs == None:
  labs = None
else:
  if args.labs.endswith('.pkl'):
      labs = pomio.unpickleObject( args.labs )
  else:
      labs = pomio.readMatFromCSV( args.labs ).astype(np.int32)


# show labels
if labs != None:
  plt.figure()
  plt.hist( labs, pomio.getNumClasses() )
  plt.title('Class counts')
  plt.xticks( range(pomio.getNumClasses()),
              pomio.getClasses()[:pomio.getNumClasses()],
              size='small' )


# show at most 9 features at once
fstarts = range( args.nstart, D, args.nshow )
plt.figure()

for fs in fstarts:
  fend = min(D-1,fs+args.nshow-1)
  print '  Displaying features %d-%d:' % (fs, fend)
  rng = range( fs, fend+1 )
  fnames = [ 'F%d' % x for x in rng ]
  amntools.gplotmatrix( ftrs[:,rng], labs, featureNames=fnames )
  plt.waitforbuttonpress()

plt.interactive(0)
plt.show()
