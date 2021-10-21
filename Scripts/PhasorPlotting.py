#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 00:35:11 2021

@author: samuelheczko
"""

import numpy as np
import MonteCarlo as MC
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import math
means=np.linspace(0.001,0.5,5)
lifetimes = np.array([10,12.5,15,17.5,20])
df = MC.read_pandas("DataForPhasorPlot")
FinalData = pd.DataFrame()
#MC.add_to_dataframe(FinalData,) df,added_list,column_name)

##1. Define some basic fucitons needed for plotting the phasor plot

def universal_semicircle():
    gs = np.linspace(0,1,100)
    Ess = np.sqrt(0.25-(gs-0.5)**2)
    return Ess,gs
def HistoData(data,length,Normal=True):
    if Normal:
        data1,bins=np.histogram(data,bins=100,range = (0,length),density = True)
    else:
        data1,bins=np.histogram(data,bins=100,range = (0,length),density = False)
    return data1,bins


##2.start the creating the phasor plot by creating the arrays to which the data will be saved to
## 50 data points: 5 means, 5 lifetimes, and two pileup condtions for each
   
length  = 150
slist = np.zeros(50)
glist = np.zeros(50)
Booleanslist = np.zeros(50)
Meanlist = np.zeros(50)
Lifetimelist = np.zeros(50)

nameslist = []
Booleans = [True,False]
Count = 0
for L in range(5):
    for M in range(5):
        for B in Booleans:
        
            LTI = L
            MI = M
            Pileup = B
            omega = math.pi*2/110
            
            if Pileup:       
                CN = "PileUpOn, mean = " + str(means[MI]) + "lifetime = " + str(lifetimes[LTI])
            else:
                CN = "PileUpOff, mean = " + str(means[MI]) + "lifetime = " + str(lifetimes[LTI]) 
            plt.hist(df[CN],bins = 100,range = (0,length))
            plt.title(CN)
            plt.show()
            #sns.histplot(df[CN],kde=True,bins=100,range = (0,200))
            
            #plt.plot(bins[:-1],data1)
            
            data1,bins=HistoData(df[CN],length)
            ##integrate to get the I
            integralNorm = np.trapz(data1,bins[:-1])
            ##G
            COS = np.cos(omega*bins[:-1])
            data2 = COS*data1
            integralG = np.trapz(data2,bins[:-1])
            G = integralG/integralNorm
            ##S
            SIN = np.sin(omega*bins[:-1])
            data3 = SIN*data1
            integralS = np.trapz(data3,bins[:-1])
            S = integralS/integralNorm
            
            nameslist.append(CN)
            slist[Count] = S
            glist[Count] = G
            
            
            if Pileup:
                Booleanslist[Count] = 1
            else:
                Booleanslist[Count] = 0
                
            
            Meanlist[Count] = means[MI]
            Lifetimelist[Count] = lifetimes[LTI]
            
            
            
            Count = Count+1
            
FinalData["S"] = slist # df,added_list,column_name
FinalData["G"] = glist
FinalData["Lifetime"] = Lifetimelist
FinalData["Mean"] = Meanlist
FinalData["Pileup"] = Booleanslist
FinalData["namelist"] = nameslist

MC.save_files_pandas(FinalData,"DataFrameForPhasorPlot0110")



    






