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

phasenames = ['FM', 'AFM', 'COI', 'PM-COI', 'FR-AFM', 'FR-FM', 'FR-PM', 'FR-COII']

paramsnames = ['Delta', 'V', 'J', 'n', 'h', 'd']

import sys
from numpy import argmin
import csv
from taskparser import return_task_list

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

    fname = sys.argv[1]
    args = return_task_list(fname) #D,V,J,n,h,d
    
    phtab = [[0 for i in range(len(phasenames))] for j in range(len(args))]
             
    for i in range(len(args)):
        phtab[i][phase_find(args[i])] = 1 
        
    with open('gs_data.csv', 'w') as f:
        fieldnames = paramsnames + phasenames
        
        writer = csv.writer(f, delimiter=';', lineterminator="\r")
        writer.writerow(fieldnames)
        for i in range(len(phtab)):
            writer.writerow(args[i]+phtab[i])
