import numpy as np
import math
from multiprocessing import Pool
from os import cpu_count
from sys import stdout

def realSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    a = arguments[2]
    b = arguments[3]
    c = arguments[4]
    tau = arguments[5]
    q = arguments[6]
    hmaxT = arguments[7]
    sumR = 0.
    alpha = 0
    sphereRadius = hmaxT**2
    sumCell = 0.0

    #Sum of the first atom with all the atoms within the cell
    if 0 in range(limInf, limSup):
        for beta in range (len(q)):
            d = np.linalg.norm(tau[:,alpha] - tau[:,beta]) 
            if ( alpha != beta ):
                # print('distance: ' + str(d))
                # print('E'+str(alpha)+str(beta)+': '+str(q[alpha]*q[beta]/d))
                # print()
                sumR += q[alpha]*q[beta]/d
                sumCell += q[alpha]*q[beta]/d
        print('Sum of the cell: '+str(sumCell))

    sumEachRest = 0.0
    #Sum of each atom in the cell with the rest of cells
    for h in range (limInf, limSup):
        for k in range (-hmaxT, hmaxT+1):
            for l in range(-hmaxT, hmaxT+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                T = h*a + k*b + l*c
                # rbox_sq = h**2 + k**2 + l**2
                # if rbox_sq > hmaxT**2: # Skip if outside maximum sphere of boxes
                #     continue
                for beta in range (len(q)):
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T) 
                    # if d > sphereRadius: # Skip if outside maximum sphere of boxes
                    #     continue
                    if ( d != 0 ):
                        # print('distance: ' + str(d))
                        # print('factors: ' +str(h)+'  '+str(k)+'   '+str(l))
                        # print('E'+str(alpha)+' with rest: '+str((q[alpha]*q[beta]/d)/2.0))
                        # print()
                        sumR += (q[alpha]*q[beta]/d)
                        sumEachRest += (q[alpha]*q[beta]/d)
    
    print('Sum each atom with rest: '+str(sumEachRest))
    return sumR

def sumCharges(aL, a, b, c, q, tau, hmaxT=20, parallel=True):
    # hmaxT defines the limit of the T vector

    if parallel:
        #Multiprocessing
        numCpus = cpu_count()

        #Function to obtain the summation limits for each of the processor cores
        def intervals(maximum, numCpus):
            increment = (maximum*2+1) / numCpus
            if increment >= 1:
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a, b, c, tau, q, maximum) for i in range(numCpus)]
            else:
                increment = 1
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a, b, c, tau, q, maximum) for i in range((maximum*2+1))]

        ###########################    
        with Pool() as p:
            resReal = p.map(realSum, intervals(hmaxT, numCpus))
        realSpace = np.sum(resReal)
        ###########################
    else:
        realSpace = realSum([-hmaxT, hmaxT+1, a, b, c, tau, q, hmaxT])

    return realSpace