#!bin/usr/env python

foo = [
lambda Delta,V,J,n,h,d: (abs(h)-d*J)*(abs(n)-1)+n*(Delta+d*V), #FM
lambda Delta,V,J,n,h,d: d*J*(abs(n)-1)+abs(n)*(Delta+d*V), #AFM
lambda Delta,V,J,n,h,d: Delta + d * V * (2*abs(n)-1), #COI
lambda Delta,V,J,n,h,d: d*V*(2*abs(n)-1)+Delta*(1-abs(n))-abs(n)*abs(h), #PM-COI
lambda Delta,V,J,n,h,d: d*J*(2*abs(n)-1)+abs(n)*(Delta-abs(h)), #FR-AFM
lambda Delta,V,J,n,h,d: d*J*(1-2*abs(n))+abs(h)*(abs(n)-1)+Delta*abs(n),
lambda Delta,V,J,n,h,d: d*V*(2*abs(n)-1)+abs(h)*(abs(n)-1)+Delta*abs(n),
lambda Delta,V,J,n,h,d: (Delta-abs(h))/2,
]

phasecolors = ['green', 'orange', 'red', 'pink', 'yellow', 'lightgreen', 'blue', 'purple'] #FM, AFM, COI, PM-COI, FR-AFM, FR-FM, FR-PM, FR-COII

phasenames = ['FM', 'AFM', 'COI', 'PM-COI', 'FR-AFM', 'FR-FM', 'FR-PM', 'FR-COII']

paramsnames = ['Delta', 'V', 'J', 'n', 'h', 'd']

from sys import argv
from numpy import argmin
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv

def plotlattice(data):
    fig, ax = plt.subplots()
    L = len(data)
    ax.plot(L,L)
    for i in range(L):
        for j in range(L):
                ax.add_patch(Rectangle((j,L-i), 1, 1, edgecolor='white', facecolor=phasecolors[data[i][j][2]]))
                       
    ax.set_aspect('equal', adjustable='box')
    plt.show() 

def phase_find(args):
    n = args[3];
    if n == 0:
        phnum = [0,1,2,7]
    if (n < 0.5):
        phnum = [0,1,2,3,4,5,7]
    if (n >= 0.5):
        phnum = [0,1,2,6]
    energy_list = [e(*args) for e in [foo[i] for i in phnum]]
    return phnum[argmin(energy_list)]
    
if __name__ == '__main__':
    Deltab = [i/10 for i in range(-50,50)]
    Jtab = [i/10 for i in range(-50,50)]
    args = list(map(lambda x: float(x), argv[1:5])) #V,n,h,d
    
    phtab = [[0 for i in range(len(Deltab))] for j in range(len(Jtab))]
    for i in range(len(Deltab)):
        for j in range(len(Jtab)):
            arg = [Deltab[i], args[0], Jtab[j], args[1], args[2], args[3]]
            phtab[i][j]= [Deltab[i],Jtab[j],phase_find(arg)]
    
    plotlattice(phtab)