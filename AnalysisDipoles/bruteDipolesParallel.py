import numpy as np
from multiprocessing import Pool
from os import cpu_count

def realSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    a = arguments[2]
    b = arguments[3]
    c = arguments[4]
    tau = arguments[5]
    q = arguments[6]
    spins = arguments[7]
    hmaxT = arguments[8]
    sumR = 0.
    sumDipole = 0.
    alpha = 0
    for h in range (limInf, limSup):
        for k in range (-hmaxT, hmaxT+1):
            for l in range(-hmaxT, hmaxT+1):
                T = h*a + k*b + l*c
                for beta in range (2):
                    dVec = tau[:, alpha] - tau[:, beta] + T
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T) 
                    if ( d != 0 ):
                        #sumR += q[alpha]*q[beta]/d
                        firstTerm = np.dot(spins[:,alpha], spins[:,beta]) /d**3
                        secondTerm = 3*np.dot(spins[:, alpha], dVec)*np.dot(spins[:, beta], dVec) /d**5
                        sumDipole += firstTerm - secondTerm
    #                else:
    #                    print("d =0")
    return sumR, sumDipole

def sumDipoles(aL, a, b, c, q, tau, spins, hmaxT=20):
    # hmaxT defines the limit of the T vector

    #Multiprocessing
    numCpus = cpu_count()

    def intervals(maximum, numCpus, a1, b1, c1):
        increment = (maximum*2+1) / numCpus
        if increment >= 1:
            return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, spins, maximum) for i in range(numCpus)]
        else:
            increment = 1
            return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, spins, maximum) for i in range((maximum*2+1))]

    ###########################    
    with Pool() as p:
        resReal = p.map(realSum, intervals(hmaxT, numCpus, a, b, c))
    realSpace = np.sum(list(zip(*resReal))[0])
    realSpaceDipole = np.sum(list(zip(*resReal))[1])
    ###########################

    return  realSpace, realSpaceDipole