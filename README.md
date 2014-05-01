*Alien Markov Networks*: experiments in MRFs for image segmentation
------------------------------------------------------------------

*Please report errors in these instructions*

This library contains a set of python tools for performing image segmentation
using Markov Random Fields (MRFS) and/or Conditional Random Fields (CRFs).

The default matrix orientation is numberExamples x numberDimensions, which is 
in line with scikit learn and matlab.

The first aim is to get some standard results on the MSRC data set.  The target
platform is Ubuntu 13.04.  By far the easiest way to use the software is on a VM
using vagrant.  If you're running Ubuntu 13.04, you could try running parts of
the provisioning script in the vagrant directory (bootstrap.sh) to install the
necessary dependencies.

[MSRC data is available here] (http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx)


## To Install on Machine

Clone the repo.

Install some needed dependencies:
```
apt-get update
apt-get install -y emacs git-core cython python-pip python-dev \
    python-numpy python-matplotlib python-scipy python-pandas \
    libboost-dev
pip install scikit-learn scikit-image
```

Get a local installation of the _slic_ python library:
```
cd mylibdir
git clone https://github.com/amueller/slic-python.git
cd slic-python
make
```

Build the cython components:
```
cd alienMarkovNetworks/maxflow
python setup.py build_ext --inplace
```

Add these libraries to your python path, possibly in your .bashrc or the like:
```
echo 'export PYTHONPATH="$PYTHONPATH":mylibdir/slic-python:mycodedir/alienMarkovNetworks/maxflow' \
    >> ~/.bashrc
```


## To Install on VM

*NOTE:* vagrant uses virtualbox by default.  For me this required switching
virtualisation on in the bios.

Run these commands:
```
  sudo apt-get install vagrant
  vagrant box add raring64 \
         http://cloud-images.ubuntu.com/vagrant/raring/current/raring-server-cloudimg-amd64-vagrant-disk1.box
  git clone https://github.com/RockStarCoders/alienMarkovNetworks.git
  cd alienMarkovNetworks/vagrant
  vagrant up
  vagrant ssh
  cd /vagrant/alienMarkovNetworks/
```
Go get some coffee, and after that you should be up and running inside the VM.

On the host machine, you can provide the MSRC data to the VM as follows:
```
  cd alienMarkovNetworks
  ./createMSRCPartition.sh -c  /path/to/MSRC_ObjCategImageDatabase_v2 \
      MSRC_dataSplit_Shotton/Train.txt \
      MSRC_dataSplit_Shotton/Validation.txt \
      MSRC_dataSplit_Shotton/Test.txt  \
      vagrant/msrcData
```


## RUN

These instructions will allow you to reproduce a result on the MSRC data set.

1. Download the [MSRC data] (http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx).  Let's call that path _DATAPATH_.
2. Copy it to partitioned sub-directories:
```
  mkdir DATAPATH/msrcPartitioned
  cd alienMarkovNetworks
  ./createMSRCPartition.sh -c  DATAPATH/MSRC_ObjCategImageDatabase_v2 \
      MSRC_dataSplit_Shotton/Train.txt \
      MSRC_dataSplit_Shotton/Validation.txt \
      MSRC_dataSplit_Shotton/Test.txt  \
      DATAPATH/msrcPartitioned
```
3. Run the following script:
```
mkdir OUTPUTDIR
./reproduceMSRC.sh DATAPATH/msrcPartitioned OUTPUTDIR 8
```
The number 8 above is the number of cores to use.  Limiting factor is RAM. I
have 32 GB RAM and 8 is good for me.  In OUTPUTDIR this produces labelled 
images, as well as accuracy metrics to standard out.

