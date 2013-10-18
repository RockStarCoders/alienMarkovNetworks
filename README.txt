alien markov networks: experiments in MRFs for image segmentation
------------------------------------------------------------------

Currently the aim is to get some decent results on the MSRC data set.  The
target platform is Ubuntu 13.04.  By far the easiest way to use the software is
on a VM using vagrant.  If you're running Ubuntu 13.04, you could try running
the provisioning script in the vagrant directory (bootstrap.sh) to install the
necessary dependencies.

# VM

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


# INSTALL

You will have to install some python packages, like numpy, sklearn, etc.

Some pip installed like pybrain.

Some you download, make sure they are on your python path:
```
  slic-python:  git clone https://github.com/amueller/slic-python.git
```
You will need the microsoft data set probably, from:
```
  http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx
```


# BUILD

It's python, you don't.  But you do have to build the C++ cython bindings:
```
   cd maxflow
   python setup.py build_ext --inplace
```
