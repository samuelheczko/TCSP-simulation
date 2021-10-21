#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:06:05 2021

@author: samuelheczko
"""

l1=np.trapz(test1[0],test1[1])
k1=[]
for i in range(len(test1[0])):
    k1.append(test1[1][i]*np.cos(9*(10**(-2))*test1[2][i]))
f1=np.trapz(test1[0],k1)
g1=f1/l1
kk1=[]
for i in range(len(test1[0])):
    kk1.append(test1[1][i]*np.sin(9*(10**(-2))*test1[2][i]))
ff1=np.trapz(test1[0],kk1)
s1=ff1/l1

x=np.linspace(0,1,100)
y=[]
for i in range(len(x)):
    if -x[i]*x[i]+x[i]>=0:
        y.append(np.sqrt(-x[i]*x[i]+x[i]))

plt.title('Phasor plot of first arriving photons (with mu=0.31 and tau=4e-9)')
plt.plot(x,y)
plt.plot(g1,s1, 'rp', markersize=14)