#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:39:44 2021

@author: samuelheczko
"""
import random as rnd

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import misc
import scipy.stats as ss
from sklearn import preprocessing
from numpy import asarray
from numpy import savetxt
import pandas as pd


from scipy.optimize import curve_fit
path = "/Volumes/SamuelSSD/School/KurfMonteCarlo/Program"##save/load the data form here
datapath = path + "/Data/" ## definethis


scaler = preprocessing.MinMaxScaler()

## 1. do the probabilty function with normalisation and associate r value
def poission(howMany,mean):
    su = np.zeros(howMany+1)
    k = np.zeros(howMany)
    mean = mean
    for s in range(howMany):
        k[s] = ((mean**s)/(math.factorial(s)))*math.exp(-mean)
        su[s+1] = k[s] + su[s]
    finalsu=np.delete(su,0)
    return finalsu,k #output: the probability of each x (k) + density 

def find_R(howfine,finalsu): ##slow method make faster
    r = np.linspace(0,1-(1/howfine),num = howfine)
    Plist=np.zeros(len(r))
    for m in range(len(r)):
        for h in range(len(finalsu)-1):
            if finalsu[h]<= r[m] <= finalsu[h+1]:
                Plist[m]=h+1
    return r,Plist #output: associate r with the each probabilty (here by index)

def normalise1(FullList,r2):
    result = np.where(FullList == 0)     
    NormialisedList=np.delete(FullList,result)
    NormialisedR=np.linspace(0,0.9999,num = len(NormialisedList))
    
    return NormialisedList,NormialisedR,result #discard the zeros from the probability distribution
        
    

def test_plots(): #just some testing no need to take seriously
    sum1, k = poission(20, 0.09)
    distributon,Ps = find_R(1000000, sum1)
    NormPS,normD,result = normalise1(Ps,distributon)
    plt.plot(k)
    plt.plot(sum1)
    plt.show()
    plt.plot(distributon,Ps)
    plt.show()
    plt.plot(normD,NormPS)
    plt.show()

#2.how many photons 
    #a) draw a random number and associate the amount of photons
def how_many_photons(finalsu,normD): #input the list of probabilites and linear r, get a possion distributed variable back
    r = np.random.rand(1)
    #Loop=True
    #LoopN=0
    dx = normD[1] - normD[0]
    interval = int((r-normD[0])//dx)
    if interval >= len(finalsu):
        P =finalsu[-1]
    
    else:
        P=finalsu[interval]
        
    return int(P)
#k = 1000
#listP = np.zeros(k)
#for n in range(k):
 #  listP[n] = how_many_photons(NormPS,normD)
#plt.hist(listP)
#3. arrival time
def find_arrival_time(how_many,lifetime):
    t = np.zeros(10)
    for n in range(how_many):
        random = np.random.rand()
        t[n]=-math.log(1-random)*lifetime
    zeros = np.where(t == 0)     
    t2=np.delete(t,zeros)  
    return t2
def arrange_arrival_time(times):
    t1 = np.amin(times)
    loc=np.where(times == np.amin(times))
    t3 = np.delete(times,loc)
    return t1, t3
def main(mean,number_of_collected_photons,lifetime):
    sum1, k = poission(30, mean)
    distributon,Ps = find_R(1000000, sum1)
    NormPS,normD,result = normalise1(Ps,distributon)
    t1list = np.zeros(number_of_collected_photons)
    t2list = []
    t3list = []
    for n in range(number_of_collected_photons):
        k = how_many_photons(NormPS, normD)
        lf = lifetime
        t2 = find_arrival_time(k, lf)
        for s in t2:
            t2list.append(s)
        t1,t3 = arrange_arrival_time(t2)
        t1list[n]=t1
        for s in t3:
            t3list.append(s)
    return t1list,t2list,t3list,lifetime
#T1,T2,T3,Lifetime=main(0.01, 10000, 3)


def RoomLight(mean,number_of_collected_photons,lifetime):
    sum1, k = poission(5, mean) #the exact amount of steps in poission dont matter as we only trawck "yes" or "no"
    distributon,Ps = find_R(1000000, sum1)
    NormPS,normD,result = normalise1(Ps,distributon) #no normalisation as we want tot see when the 1st photon hits
    t1list = np.zeros(number_of_collected_photons)
    t2list = []
    t3list = []
    for n in range(number_of_collected_photons):
        time_of_excitation = n%700
        k = how_many_photons(Ps, distributon)
        lf = lifetime
        t2 = find_arrival_time(k, lf) + time_of_excitation
        for s in t2:
            t2list.append(s)
        t1,t3 = arrange_arrival_time(t2)
        t1list[n]=t1
        for s in t3:
            t3list.append(s)
    return t1list,t2list,t3list,lifetime


def RoomLight2(lambda1,whole_time,number_of_experimets,number_of_cycles):
    times=np.zeros(number_of_experimets)
    alltimes = []
    for n in range(number_of_experimets):
        for k in range(number_of_cycles):
            r = np.random.poisson(lam=lambda1)
            #r = np.random.rand()
           
            if r != 0:
                if  times[n] == 0:
                    times[n]=k+1
                    break
                alltimes.append(k+1)
                #break
    return times,alltimes

#def RoomLight3(mean,n,t):
#    Dt = t/n
 #   PulseRate=mean/t
  #  for n in range n:
        
   # probabilityofemission

def RoomLight4(lambda1,whole_time,number_of_experimets,number_of_cycles,sum1, k,distributon,Ps,scalefactor): #same but with my own possion generator
    times=np.zeros(number_of_experimets)
    alltimes = []
    sum1, k = sum1, k
    distributon,Ps = distributon,Ps
    for n in range(number_of_experimets):
        for k in range(number_of_cycles): ## For each moment in time (during the exitation cycle, we ask how many photons are counted. We record only the first one
            m = how_many_photons(Ps, distributon) #finds a random number in poission
           
            if m != 0:
                if  times[n] == 0:
                    times[n]=k/scalefactor
                    break
                alltimes.append(k/scalefactor)
                #break
    return times,alltimes ## Get back arrival times of first photons + all of the times
    
    
def count_zero_in_decimal_number(number):
    zeros = 0
    while number < 0.1:
        number *= 10
        zeros += 1
    return zeros

def add_to_dataframe(df,added_list,column_name):
    df.loc[:,column_name] = pd.Series(added_list)
    
    return df
def save_files_pandas(df,name):
    df.to_csv(datapath+name+".csv")
def read_pandas(name):
    df = pd.read_csv(datapath+name+".csv")
    return df

    

def roomlightPlot(howManyIterations,MeanN): ## Doesn't inc
    mean = np.linspace(0.009, 0.6,MeanN) #create arrays of mean per second
    measured_mean = np.zeros(len(mean))
    targetbinN = 700 #number of bins, experimantally found to be sufficent
    df = pd.DataFrame()
    
    for m in range(len(mean)):
        maxtime = int(6/mean[m])#
        scalefactor = targetbinN/maxtime
        
        
        sum1, k = poission(5, mean[m]/scalefactor)   
        distributon,Ps = find_R(1000000, sum1)
        ttimes,alltimes = RoomLight4(mean[m]/scalefactor, 1, howManyIterations, targetbinN,sum1, k,distributon,Ps,scalefactor) 
        #plt.hist(ttimes)
        ttimesW = np.where(ttimes == 0)     
        ttimes2=np.delete(ttimes,ttimesW)
        plt.hist(ttimes2,bins=targetbinN,density = True)
        P = ss.expon.fit(ttimes2)
        rX = np.linspace(0,maxtime,targetbinN)
        rP = ss.expon.pdf(rX, *P)
        plt.plot(rX[1:], rP[1:])
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = on, Decay Read from the fit = {P[1]}' )
        plt.show()
        measured_mean[m] = 1/P[1]
        number_of_zeros=count_zero_in_decimal_number(measured_mean[m])
        measured_mean[m] = measured_mean[m] - 0.1**(number_of_zeros+2)
        df = add_to_dataframe(df, ttimes2, "mean = " + str(mean[m]))
        
    
    
    
    #fig2, ax2 = plt.subplots()    
    #ax2.scatter(measured_mean,mean, marker="x",color = "k",label = "Simulation results")
    #ax2.plot(mean,mean,label = "Control line")
    #plt.ylabel("Imputted Poission mean")
    #plt.xlabel("Measured Poission mean")
    #leg = ax2.legend(prop={'size': 10})
    #plt.show()
    return measured_mean,mean,df


def saveData(title1,data1,title2,data2): #plug the larger data later
    for n in range(len(data2)-len(data1)):
            data1 = np.append(data1,0)
    data2 = np.vstack((data1,data2))
    data3 = data2.T
    np.savetxt("columns_"+ str(title1) + "_"+ str(title1) + ".csv",data3,delimiter=",")

def getDataAsNpArrays(filename): #just remember the filename in quotes
    data = np.loadtxt(str(filename),delimiter=",")
    data1 = data[:,0]
    data2 = data[:,1]
    result1 = np.where(data1 == 0)
    result2 = np.where(data2 == 0)
    data1=np.delete(data1,result1)
    data2=np.delete(data2,result2)
    return data1,data2








#for n in range(len(means)):
#    T1,T2,T3,lifetime=main(means[n], 1000000, 10)
#    saveData("T1"+str(means[n]),T1,"T2",T2)
#    
#    plt.hist(T1, bins = "auto", density = True)
    
#    P = ss.expon.fit(T1)
#    rX = np.linspace(0,200, 200)
#    rP = ss.expon.pdf(rX, *P)
#    plt.plot(rX[1:], rP[1:])
#    plt.xlabel(r'Decay Time')
#    plt.ylabel(r'Normalised distribution')
#    plt.title(f'Pileup = on, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
#    plt.show()
#    Simualted_lifetimesPileup[n]=P[1]
    
    
#    plt.hist(T2, bins = "auto", density = True)
#    P = ss.expon.fit(T2)
#    rX = np.linspace(0,200, 200)
#    rP = ss.expon.pdf(rX, *P)
#    plt.plot(rX[1:], rP[1:])
#    plt.xlabel(r'Decay Time')
#    plt.ylabel(r'Normalised distribution')
#    plt.title(f'Pileup = off, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
#    plt.show()
#    Simualted_lifetimesNoPileup[n]=P[1]
def exponential(x, a):
    return a*np.exp(-a*x)
def sin(x,w):
    return np.sin(x*w)
def cos(x,w):
    return np.cos(x*w)


def GetTheData():
    means=np.linspace(0.001,0.5,10)
    Simualted_lifetimesPileup = np.zeros(len(means))
#
    Simualted_lifetimesNoPileup = np.zeros(len(means))
    Simualted_lifetimesPileupCorrection = np.zeros(len(means))
    Simualted_lifetimesPileupCorrectionOwn = np.zeros(len(means))
    
    
    for n in range(len(means)):
        T1,T2=getDataAsNpArrays("/Volumes/SamuelSSD/"+"columns_T1"+str(means[n])+"_T1"+ str(means[n])+".csv") #columns_T10.5_T10.5
        #saveData("T1"+str(means[n]),T1,"T2",T2)
        
        plt.hist(T1, bins = 300, density = True)
        
        P = ss.expon.fit(T1,floc=0)
        rX = np.linspace(0,150, 300)
        rP = ss.expon.pdf(rX, *P)
        plt.plot(rX[1:], rP[1:])
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = on, Lifetime Read from the fit = {P[1]}' )
        plt.show()
        Simualted_lifetimesPileup[n]=P[1]
        
        
        plt.hist(T2, bins = 300, density = True)
        P = ss.expon.fit(T2,floc=0)
        rP = ss.expon.pdf(rX, *P)
        plt.plot(rX[1:], rP[1:])
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = off, Lifetime Read from the fit = {P[1]}' )
        plt.show()
        
        Simualted_lifetimesNoPileup[n]=P[1]
        
        

        #unique, counts = np.unique(Interval, return_counts=True)
        height,bins = np.histogram(T1,np.arange(0,150,0.1),density=True)
        correctedHeight=np.zeros(len(height))
        PlottingBins=np.zeros(len(bins)-1)
        
        for m in range(len(height)):
            correctedHeight[m] = height[m] + ((height[m])**2)*0.5
        for s in range(len(PlottingBins)):
            PlottingBins[s] = (bins[s+1] + bins[s])/2
            #PlottingBins[s] = bins[s]
        plt.bar(PlottingBins,height)
        pars,cov = curve_fit(f=exponential, xdata =PlottingBins, ydata=height, p0=[0], bounds=(-np.inf, np.inf))
        plt.plot(rX,exponential(rX, pars), "r")
        lifetime = 1/pars
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = Not Corrected, Lifetime Read from the fit = {lifetime}' )
        
        plt.show()
        
        #plt.scatter(PlottingBins, height)

        
        plt.bar(PlottingBins,height)
        pars, cov = curve_fit(f=exponential, xdata=PlottingBins, ydata=height, p0=[0], bounds=(-np.inf, np.inf))
        plt.plot(rX,exponential(rX, pars),"r",label="curvefit")
        plt.plot(rX, rP,"g",label="scipyfit")
        lifetime = 1/pars
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        #plt.yscale("log")
        plt.legend()
        plt.title(f'Pileup = Not Corrected, Lifetime Read from the fit = {lifetime}' )

        plt.show()
        Simualted_lifetimesPileupCorrectionOwn[n] = lifetime
        
        plt.plot(PlottingBins, correctedHeight)

        pars, cov = curve_fit(f=exponential, xdata =PlottingBins, ydata=correctedHeight, p0=[0], bounds=(-np.inf, np.inf))
        plt.plot(rX,exponential(rX, pars))
        lifetime = 1/pars
        plt.xlabel(r'Decay Time')
        plt.ylabel(r'Normalised distribution')
        plt.title(f'Pileup = corrected, Lifetime Read from the fit = {lifetime}' )

        plt.show()
        
        Simualted_lifetimesPileupCorrection[n] = lifetime
        
        
        
        
        
        print(P)
 
        
        
    return (Simualted_lifetimesNoPileup,Simualted_lifetimesPileup,means,Simualted_lifetimesPileupCorrection,Simualted_lifetimesPileupCorrectionOwn,P)
#define objective

def objective(x,a):
    return a*x + 1
#find y data
def Run_The_Old_Scrip():
    Simualted_lifetimesNoPileup,Simualted_lifetimesPileup,means,Simualted_lifetimesPileupCorrection,Simualted_lifetimesPileupCorrectionOwn,P = GetTheData()
    k = Simualted_lifetimesPileup/Simualted_lifetimesNoPileup
    #curvefit
    popt,_ = curve_fit(objective, means, k)
    a = popt
    print('y = %.5f * x'% (a))
    fig2, ax2 = plt.subplots()
    ax2.scatter(means, k,label="Simulation Results")
    x_line = np.arange(min(means), max(means),0.01)
    y_line = objective(x_line, a)
    plt.ylabel("Normalised decay time")
    plt.xlabel("Poission mean")
    ax2.plot(x_line, y_line, '--', color='red',label="Fitted Curve")
    leg = ax2.legend(prop={'size': 10})
    plt.show()
    ax2.scatter(means,Simualted_lifetimesPileupCorrection)
    ax2.scatter(means,Simualted_lifetimesNoPileup)
    plt.show()
    
    k = Simualted_lifetimesPileupCorrectionOwn/Simualted_lifetimesNoPileup
    #curvefit
    popt,_ = curve_fit(objective, means, k)
    a = popt
    print('y = %.5f * x' % (a))
    fig2, ax2 = plt.subplots()
    ax2.scatter(means, k,label="Simulation Results")
    x_line = np.arange(min(means), max(means),0.01)
    y_line = objective(x_line, a)
    plt.ylabel("Normalised decay time")
    plt.xlabel("Poission mean")
    ax2.plot(x_line, y_line, '--', color='red',label="Fitted Curve")
    leg = ax2.legend(prop={'size': 10})
    plt.show()
    ax2.scatter(means,Simualted_lifetimesPileupCorrection)
    ax2.scatter(means,Simualted_lifetimesNoPileup)
    plt.show()


  

   
    
#measuredMean,Mean = roomlightPlot(20000)
#saveData("measuredMeans", measuredMean,"realMean ", Mean)

#data1,data2 = getDataAsNpArrays("columns_measuredMeans_measuredMeans.csv")



    
    


    


#plt.hist(alltimes,bins=50)



#plt.show()            
        
        
        
    
    


#t1,t2,t3,lifetime = main(0.6,1000000,5)
#plt.hist(t1, bins = "auto", density = True)
#P = ss.expon.fit(t1)
#rX = np.linspace(0,200, 200)
#rP = ss.expon.pdf(rX, *P)
#plt.plot(rX[1:], rP[1:])
#plt.xlabel(r'Decay Time')
#plt.ylabel(r'Normalised distribution')
#plt.title(f'Pileup = on, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
#plt.show()

#plt.hist(t2, bins = "auto", density = True)
#P = ss.expon.fit(t2)
#rX = np.linspace(0,200, 200)
#rP = ss.expon.pdf(rX, *P)
#plt.plot(rX[1:], rP[1:])
#plt.xlabel(r'Decay Time')
#plt.ylabel(r'Normalised distribution')
#plt.title(f'Pileup = off, Lifetime = {lifetime}, Lifetime Read from the fit = {P[1]}' )
#plt.show()


sum1, k = poission(30, 0.7)
distributon,Ps = find_R(1000000, sum1)
plt.plot(distributon,Ps)
plt.title("Mean = 0.7")
plt.ylabel(r'Numer of photons in Poisson statistics')
plt.xlabel(r'Uniform distribution')
plt.show()

NormPS,normD,result = normalise1(Ps,distributon)
plt.plot(normD,NormPS)
plt.title("Mean = 0.7, normalised")
plt.ylabel(r'Numer of photons in Poisson statistics')
plt.xlabel(r'Uniform distribution')
plt.ylim(0, 7.5)
plt.show()


    