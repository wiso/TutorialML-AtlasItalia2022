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
    def __init__(self): 
        self.ekin_all = []
        self.phimod_all = []
        self.eta_all = []
        self.Energies_all = []
        self.X_all = []
        self.Labels_all = []  
        self.maxVoxelAll = 0    
        self.maxVoxelMid = 0
        self.maxVoxels = []

        print ("************* DATA READER ***************")
        print ("Loading data")

        for i in range(8, 23):
            self.ekins.append(2**i)

        self.midEnergy = self.ekins[15]
        self.LoadData()

    def LoadData(self):
        print("----Loading files----")
        #print ("  Label is " + self.dataParameters.label_definition.name)
        #print ("  Normalised using " + self.dataParameters.voxel_normalisation.name)
        for index, p in enumerate(self.ekins):
            fileName = f"../gan_inputs/pid22_E%s_eta_20_25_voxalisation.csv" % (p)
            print("Opening file " + fileName)
            df = pd.read_csv(fileName, header=None, engine='python', dtype=np.float64)
            df = df.fillna(0)

            phimod = df.iloc[ : , 0 ].to_numpy()
            etaColumn = abs(df.iloc[ : , 1 ].to_numpy())
            
            first_column = df.columns[0]
            second_column = df.columns[1]
            df = df.drop([first_column], axis=1) #Removing the first element which is phiMod
            df = df.drop([second_column], axis=1) #Removing the first element which is eta
           
            Energies=df.to_numpy()
            nevents=len(Energies)
            
            #Set array to zero to remove conditioning
            phimod = np.zeros(nevents)
            etaColumn = np.zeros(nevents)

            print("Loaded momentum " + str(p)) 
            print("from file " + fileName)
            print("with " + str(nevents) + " events")
            print("Vector of data of size " + str(len(Energies[0]))) 
            assert not np.any(np.isnan(Energies))
           
            ekin = np.array([self.ekins[index]] * nevents)
            
            self.ekin_all.append(ekin)
            self.eta_all.append(etaColumn)
            self.Energies_all.append(Energies)
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
        label = DataLoader.energyLabel(self.dataParameters.label_definition, ekin, self.E_min, self.E_max)

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
            sampleIndex = i - self.dataParameters.min_expE
            x = copy.copy(self.X_all[sampleIndex])
            label = self.Labels_all[sampleIndex]

            x_all.extend(x)
            label_all.extend(label)

        x_all = [tf.convert_to_tensor(np.asarray(x), dtype=tf.float32) for x in x_all]
        label_all = [tf.convert_to_tensor(np.array(label), dtype=tf.float32) for label in label_all]

        return x_all, label_all
        
    @staticmethod
    def energyLabel(labelType, energy, E_min, E_max):
        return math.log(energy/E_min) / math.log(E_max/E_min)

    @staticmethod
    def momentumsToEKins(momentums, mass):
      ekins = []
      for momentum in momentums:
        ekin = math.sqrt(momentum*momentum+mass*mass) - mass
        ekins.append(ekin)
      
      return ekins
