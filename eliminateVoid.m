% script to create smooshed images, to compute adjacency.
% aim is to get rid of all "label=0" pixels and replace with a non-zero
% label that should be in some spatial sense nearest. 
%
% J SHerrah 10 July 2013

% example gt label image, in rgb
fn = '12_11_s_GT.bmp'
% read it
x=imread(fn);
% convert to integer labels, 0 should be 'void'.  In this example
% hard-wired 4 as number of unique labels (inc void, so 3 is max label)
y = rgb2ind(x,4);
% display
myim(y);
% some non-void regions touch.  We need to get rid of these pixels first so
% all distinct regions are separated by some void pixels.  Otherwise they
% get merged as one region.
% Use morphology to find "edges"
ej = ((imdilate(y,strel('square',3))-y)>0);
% impose these as void on a working image, z
z = y; z(ej) = 0;
% display, now we should se void between all regions
myim(z);
% compute distance transform: at void pixels, euclidean dst to nearest
% non-void pixel
D = bwdist(z>0);
% watershed fills in void pixels with label of nearest non-void region.
% WARNING: the labels from watershed do not correspond to those in the
% ground truth image.  You would need to do something like:
%   for each ground truth label:
%     look up corresponding label in watershed image
%     replace watershed labels with ground truth label
myim(watershed(D));
% watershed boundaries are still there as 0 (void).  Eliminate them by
% dilating.
myim(imdilate(watershed(D),strel('square',3)),[],[0,3])
