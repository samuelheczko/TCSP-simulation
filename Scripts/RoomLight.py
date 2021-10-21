#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:27:16 2021

@author: samuelheczko
"""


import MonteCarlo as MC
import matplotlib.pyplot as plt
import numpy as np



MeanN = 25
measured_mean,mean,df = MC.roomlightPlot(10000,MeanN) #input the amount of needed datapoints per measurement, amount of means to be tested in range 0.009 and 0.6


fig2, ax2 = plt.subplots()
ax2.scatter(measured_mean,mean, marker="x",color = "k",label = "Simulation results")
ax2.plot(mean,mean,label = "Control line")
plt.ylabel("Imputted Poission mean")
plt.xlabel("Measured Poission mean")
leg = ax2.legend(prop={'size': 10})

name = "MeanDataDF"
MC.save_files_pandas(df,name) #optonal: save df 
mean = np.linspace(0.009, 0.6,MeanN)
plt.show()