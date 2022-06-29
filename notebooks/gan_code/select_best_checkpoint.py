#!/usr/bin/env python3
import numpy as np
import math
import argparse 
import os,sys,ctypes
import ROOT 
import shutil
import ctypes
import glob

ROOT.gROOT.SetBatch(True)

from conditional_wgangp import WGANGP
from DataReader import DataLoader

save_plots = True
lparams = {'xoffset' : 0.1, 'yoffset' : 0.27, 'width'   : 0.8, 'height'  : 0.35}
canvases = []   

histos_vox = []
input_files_vox = []

minFactor = 3
maxFactor = 3  
particleName="#gamma"
particle="photons"

output_best_checkpoints="best_iteration"
if not os.path.exists(output_best_checkpoints):
  os.makedirs(output_best_checkpoints)

maxVoxel = 0
midEnergy = 0
step = 50

wgan = WGANGP()

dl = DataLoader()
ekins = dl.ekins

firstPosition = 0

print("Opening vox files")
for index, energy in enumerate(ekins):    
  #print(" Energy ", energy)
  input_file_vox = ('rootFiles/pid22_E%s_eta_20_25.root' % (energy))
  print(" Opening file: " + input_file_vox)
  infile_vox = ROOT.TFile.Open(input_file_vox, 'read') 
  input_files_vox.append(infile_vox)
  tree = infile_vox.Get('rootTree') 
  
  h = ROOT.TH1F("h","",100,0,energy*2) 
  tree.Draw("etot>>h","","off")
  xmax=h.GetBinCenter(h.GetMaximumBin());
  minX = max(0, xmax-minFactor*h.GetRMS()) #max(0, xmax-minFactors[item]*h.GetRMS())
  maxX = xmax+maxFactor*h.GetRMS()
  print("min "+ str(minX) + " max " + str(maxX))
      
  h_vox = ROOT.TH1F("h_vox","",30,minX/1000,maxX/1000) 
  tree.Draw("etot/1000>>h_vox","","off")
  h_vox.Scale(1/h_vox.GetEntries())
  histos_vox.append(h_vox)

print ("Running from %i to %i in step of %i" %(0, 1000, step))
for iteration in range(0, 1000, step):
  try:
    histos_gan =[]
    canvas = ROOT.TCanvas('canvas_h', 'Total Energy comparison plots', 900, 900)
    canvas.Divide(4,4)
    legendPadIndex = 16

    canvases.append(canvas)

    chi2_tot = 0.
    ndf_tot = 0
    input_files_gan = []

    for index, energy in enumerate(ekins):     
      ekin_sample = ekins[index]
      energyArray = np.array([energy] * nevents)
      etaArray = np.zeros(nevents) 
      labels = np.vstack((energyArray, etaArray)).T   

      data = wgan.load(iteration, labels, n_events, 'checkpoints')
      data = data * ekin_sample       #needed for conditional
        
      h_vox = histos_vox[index]
      h_gan = ROOT.TH1F("h_gan","",30,h_vox.GetXaxis().GetXmin(),h_vox.GetXaxis().GetXmax())


      E_tot = data.numpy().sum(axis=1)
      for e in E_tot:
        h_gan.Fill(e/1000)
      
      h_gan.Scale(1/h_gan.GetEntries())
      h_gan.SetLineColor(ROOT.kRed)
      h_gan.SetLineStyle(7)
      m = [h_vox.GetBinContent(h_vox.GetMaximumBin()),h_gan.GetBinContent(h_gan.GetMaximumBin())]
      h_vox.GetYaxis().SetRangeUser(0,max(m) *1.25)
      histos_gan.append(h_gan)
      h_vox.GetYaxis().SetTitle("Entries")

      xAxisTitle = "Energy [GeV]"
      h_vox.GetXaxis().SetTitle(xAxisTitle)  
      h_vox.GetXaxis().SetNdivisions(506)
      chi2 = ctypes.c_double(0.)
      ndf = ctypes.c_int(0)
      igood = ctypes.c_int(0)
      histos_vox[index].Chi2TestX(h_gan, chi2, ndf, igood, "WW")
      ndf = ndf.value
      chi2=chi2.value
      chi2_tot += chi2
      ndf_tot += ndf

      if (ndf != 0):
        print("Iteration %s Energy %s : chi2/ndf = %.1f / %i = %.1f\n" % (iteration, energy, chi2, ndf, chi2/ndf))

      # Plotting

      canvas.cd(index+1)
      histos_vox[index].Draw("HIST")
      histos_gan[index].Draw("HIST same")

      # Legend box                                                                                                                                                                            
      if (energy > 1024):
          energy_legend =  str(round(energy/1000,1)) + " GeV"
      else:
          energy_legend =  str(energy) + " MeV"
      t = ROOT.TLatex()
      t.SetNDC()
      t.SetTextFont(42)
      t.SetTextSize(0.1)
      t.DrawLatex(0.2, 0.83, energy_legend)
   
    # Total Energy chi2
    chi2_o_ndf = chi2_tot / ndf_tot
    print("Iteration %s Total Energy : chi2/ndf = %.1f / %i = %.3f\n" % (iteration, chi2_tot, ndf_tot, chi2_o_ndf))
    chi2File = "%s/chi2.txt" % (output_best_checkpoints, pid, eta_min, eta_max)
    if chi2_o_ndf > 0:
      f = open(chi2File, 'a')
      f.write("%s %.3f\n" % (iteration, chi2_o_ndf))
      f.close()
    else:
      print("Something went wrong, chi2 will not be written. Chi2/ndf is %f " % (chi2_o_ndf))
      print(E_tot)
      continue

    # Legend box particle
    leg = MakeLegend( lparams )
    leg.SetTextFont( 42 )
    leg.SetTextSize(0.1)
    canvas.cd(legendPadIndex)
    leg.AddEntry(h_vox,"Geant4","l") #Geant4
    leg.Draw()
    leg.AddEntry(h_gan,"GAN","l")  #WGAN-GP
    leg.Draw('same')
    legend = (particleName + ", " + str('{:.2f}'.format(int(20)/100,2)) + "<|#eta|<" + str('{:.2f}'.format((int(20)+5)/100,2)))
    ROOT.ATLAS_LABEL_BIG( 0.1, 0.9, ROOT.kBlack, legend )

    # Legend box Epoc&chi2 

    t = ROOT.TLatex()
    t.SetNDC()
    t.SetTextFont(42)
    t.SetTextSize(0.1)
    t.DrawLatex(0.1, 0.18, "Iter: %s" % (iteration))
    t.DrawLatex(0.1, 0.07, "#scale[0.8]{#chi^{2}/NDF = %.0f/%i = %.1f}" % (chi2_tot, ndf_tot, chi2_o_ndf))


    #Copy best epoch files, including plots
    epochs, chi2_o_ndf_list = np.loadtxt(chi2File, delimiter=' ', unpack=True)
    
    checkpointName =  "Plot_comparison_tot_energy"
      
    if round(chi2_o_ndf,3) <= np.amin(chi2_o_ndf_list) and chi2_o_ndf > 0:
      print ("Better chi2, creating plots")
      inputFile_Plot_png="%s/%s.png" % (output_best_checkpoints, checkpointName)
      inputFile_Plot_eps="%s/%s.eps" % (output_best_checkpoints, checkpointName)
      inputFile_Plot_pdf="%s/%s.pdf" % (output_best_checkpoints, checkpointName)
      canvas.SaveAs(inputFile_Plot_png) 
      canvas.SaveAs(inputFile_Plot_eps) 
      canvas.SaveAs(inputFile_Plot_pdf) 
     
      print("Epoch with lowest chi2/ndf is %s with a value of %.3f" % (epoch, chi2_o_ndf))
      #Now save best epoch number to file
      chi2File = "%s/chi2/epoch_best_chi2_%s_%s_%s.txt" % (output_best_checkpoints, pid, eta_min, eta_max)
      f = open(chi2File, 'w')
      f.write("%s %.3f\n" % (iteration, chi2_o_ndf))
      f.close() 

    if (save_plots) :
      checkpointName =  "Plot_comparison_tot_energy_%i" % (iteration)
      inputFile_Plot_png="%s/%s.png" % (output_best_checkpoints, checkpointName)
      inputFile_Plot_eps="%s/%s.eps" % (output_best_checkpoints, checkpointName)
      inputFile_Plot_pdf="%s/%s.pdf" % (output_best_checkpoints, checkpointName)
      canvas.SaveAs(inputFile_Plot_png) 
      canvas.SaveAs(inputFile_Plot_eps) 
      canvas.SaveAs(inputFile_Plot_pdf) 

  except:
    print("Something went wrong in iteration %s, moving to next one" % (iteration))
    print("exception message ", sys.exc_info()[0])     

