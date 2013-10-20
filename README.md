*Alien Markov Networks*: experiments in MRFs for image segmentation
------------------------------------------------------------------

Currently the aim is to get some decent results on the MSRC data set.  The
target platform is Ubuntu 13.04.  By far the easiest way to use the software is
on a VM using vagrant.  If you're running Ubuntu 13.04, you could try running
the provisioning script in the vagrant directory (bootstrap.sh) to install the
necessary dependencies.

[MSRC data is available here] (http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx)


## VM

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

Inside the VM, first create features from the data.  For each of the training,
validation and test sets of images, features can be created for a range of
oversegmentation parameter settings as follows:

```  
  mkdir /vagrant/features
  cd /vagrant/alienMarkovNetworks/
  ./createMSRCFeatures.sh /vagrant/msrcData/training   /vagrant/features/msrcTraining
  ./createMSRCFeatures.sh /vagrant/msrcData/validation /vagrant/features/msrcValidation
  ./createMSRCFeatures.sh /vagrant/msrcData/test       /vagrant/features/msrcTest
```

Each feature set consists of 3 files: features, labels and super-pixel adjacency statistics.
Train a classifier with default parameters on each of these training sets, and test on
the corresponding validation set:

```
  ./evaluateMSRCOverseg.sh /vagrant/features/msrc
```

Of the parameters considered, the best values are ??.  Use this data set to
perform a grid search for parameters of the random forest classifier.  Uses
cross-validation on the training set.  In this context, the validation set can
be a hold-out test set to check the generalisation.

```
    nohup time ./trainClassifier.py \
     --outfile /vagrant/classifier_msrc_rf_400-10_grid.pkl \
      --type=randyforest --paramSearchFolds=5 \
       --ftrsTest=/vagrant/features/msrcValidation_slic-400-010.00_ftrs.pkl \
        --labsTest=/vagrant/features/msrcValidation_slic-400-010.00_labs.pkl \
        /vagrant/features/msrcTraining_slic-400-010.00_ftrs.pkl \
        /vagrant/features/msrcTraining_slic-400-010.00_labs.pkl \
         --nbJobs=2 &
```

