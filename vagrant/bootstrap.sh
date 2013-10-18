#!/usr/bin/env bash

# Install packages
apt-get update
apt-get install -y emacs git-core cython python-pip python-dev \
    python-numpy python-matplotlib python-scipy \
    libboost-dev
pip install scikit-learn scikit-image

# Clone slic repo
cd /vagrant && \
    git clone https://github.com/amueller/slic-python.git
# Build it
cd /vagrant/slic-python && make

# Clone our repo
cd /vagrant && \
    git clone https://github.com/RockStarCoders/alienMarkovNetworks.git
# Build it
cd /vagrant/alienMarkovNetworks/maxflow/ && \
    python setup.py build_ext --inplace

# Make these findable to python
echo "ADDING PYTHONPATH to .bashrc"
echo "" >> /home/vagrant/.bashrc
echo 'export PYTHONPATH="$PYTHONPATH":/vagrant/slic-python:/vagrant/alienMarkovNetworks/maxflow' \
    >> /home/vagrant/.bashrc

