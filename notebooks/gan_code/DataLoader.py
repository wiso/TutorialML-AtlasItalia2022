import os
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import copy

import random, numpy
random.seed(10)
numpy.random.seed(10)

from numpy import array

class DataLoader:
    def __init__(self, fns): 
        self.ekin_all = []
        self.phimod_all = []
        self.eta_all = []
        self.Energies_all = []
        self.X_all = []
        self.Labels_all = []  
        self.maxVoxelAll = 0    
        self.maxVoxelMid = 0
        self.maxVoxels = []
        self.fns = fns

        print ("Loading data")

        max_exp = 10
        self.ekins = [2 ** i for i in range(8, max_exp)]
        self.midEnergy = self.ekins[0]
        self.LoadData()

    def LoadData(self):
        for index, p in enumerate(self.ekins):
            fileName = self.fns[p]
            print("Opening file %s" % fileName)
            df = pd.read_csv(fileName, header=None, engine='python', dtype=np.float64)
            df = df.fillna(0)
           
            nevents = len(df)
            print("Loaded momentum %d GeV with %d events and %d columns" % (p, nevents, len(df.columns)))
                       
            ekin = np.array([self.ekins[index]] * nevents)
            self.ekin_all.append(ekin)

            #Set array to zero to remove conditioning
            etaColumn = np.zeros(nevents)
            self.eta_all.append(etaColumn)
            self.Energies_all.append(df.to_numpy())
            self.E_min = np.min(self.ekins)
            self.E_max = np.max(self.ekins)

        self.CreateLabelsArray()
             
    def CreateLabelsArray(self):
        for index, ekin in enumerate(self.ekins):
            self.DefineEnergyLabels(index)
            labels = np.vstack((self.ekin_all[index], self.eta_all[index])).T
            self.Labels_all.append(labels)
            self.NormaliseData(index)
            self.X_all.append(self.Energies_all[index])
        
    def DefineEnergyLabels(self, index):       
        ekin = self.ekin_all[index][0]
        label = DataLoader.energyLabel(ekin, self.E_min, self.E_max)

        nevents = len(self.ekin_all[index])
        self.ekin_all[index] = np.array([label]*nevents)
                    
    def NormaliseData(self, index):
        self.ApplyNormalisation(index)

    def ApplyNormalisation(self, index):
        Energies=self.Energies_all[index]
        ekin=self.ekins[index]
        
        Energies = Energies/ekin
        print("Data was normalised by %f" % ekin)
        
        self.Energies_all[index] = Energies
        
    def getDim(self):
        return len(self.Energies_all[0][0])
        
    def getMaxVoxelMid(self):
        return self.maxVoxelMid 

    def getMaxVoxelAll(self):
        return self.maxVoxelAll

    def getMidEnergy(self):
        return self.midEnergy

    def getMaxVoxels(self):
        return self.maxVoxels

    def getAllTrainData(self, min, max):
        x_all = []
        label_all = []

        for i in range(min, max+1): 
            sampleIndex = i - 8
            x = copy.copy(self.X_all[sampleIndex])
            label = self.Labels_all[sampleIndex]

            x_all.extend(x)
            label_all.extend(label)

        x_all = [tf.convert_to_tensor(np.asarray(x), dtype=tf.float32) for x in x_all]
        label_all = [tf.convert_to_tensor(np.array(label), dtype=tf.float32) for label in label_all]

        return x_all, label_all
        
    @staticmethod
    def energyLabel(energy, E_min, E_max):
        return math.log(energy/E_min) / math.log(E_max/E_min)

    @staticmethod
    def momentumsToEKins(momentums, mass):
      ekins = []
      for momentum in momentums:
        ekin = math.sqrt(momentum*momentum+mass*mass) - mass
        ekins.append(ekin)
      
      return ekins
