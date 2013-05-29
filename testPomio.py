#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
# Give it the path to the data set as an arg
import pomio

X = pomio.msrc_loadImages( sys.argv[1] )
print X

# Display the images
plt.ion()
for img in X:
    plt.figure(1)
    plt.imshow( img.m_img )
    plt.title('Image %s' % img.m_imgFn)
    plt.figure(2)
    plt.imshow( img.m_gt )
    plt.title('labels')
    plt.draw()
    plt.pause(0.2)

#plt.ioff()
#plt.show()
