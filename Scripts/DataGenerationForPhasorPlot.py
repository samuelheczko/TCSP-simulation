#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:50:35 2021

@author: samuelheczko
"""

import MonteCarlo as MC
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pandas as pd

means=np.linspace(0.001,1.0,10)
Simualted_lifetimesPileup = np.zeros(len(means))
#
Simualted_lifetimesNoPileup = np.zeros(len(means))
Simualted_lifetimesPileupCorrection = np.zeros(len(means))
Simualted_lifetimesPileupCorrectionOwn = np.zeros(len(means))
df = pd.DataFrame()
lifetimes = np.array([10,20,30,40,50,60])

for s in lifetimes:
    
    for n in range(len(means)):
        T1,T2,T3,lifetime=MC.main(means[n], 100000, s)
        ##MC>saveData("T1"+str(means[n]),T1,"T2",T2) MAKE PANDAS
        MC.add_to_dataframe(df, T1, "PileUpOn, mean = " + str(means[n]) + "lifetime = " + str(lifetime))
        MC.add_to_dataframe(df, T2, "PileUpOff, mean = " + str(means[n]) + "lifetime = " + str(lifetime))
        
        
        plt.hist(T1, bins = "auto", density = True)
       
        P = ss.expon.fit(T1)
        rX = np.linspace(0,200, 200)
        rP = ss.expon.pdf(rX, *P)
        plt.plot(rX[1:], rP[1:])
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = on, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
        plt.show()
        Simualted_lifetimesPileup[n]=P[1]
        
        
        plt.hist(T2, bins = "auto", density = True)
        P = ss.expon.fit(T2)
        rX = np.linspace(0,200, 200)
        rP = ss.expon.pdf(rX, *P)
        plt.plot(rX[1:], rP[1:])
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = off, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
        plt.show()
        Simualted_lifetimesNoPileup[n]=P[1]

MC.save_files_pandas(df,"DataForPhasorPlot")
    