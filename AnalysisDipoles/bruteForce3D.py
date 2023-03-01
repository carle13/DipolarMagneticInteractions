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
    spins = arguments[6]
    hmaxT = arguments[7]
    sumDipole = 0.
    sphereRadius = hmaxT**2
    sumCell = 0.0

    #Compute terms within the cell
    if 0 in range(limInf, limSup):
        print(tau[0])
        for alpha in range(len(tau[0])):
            for beta in range(len(tau[0])):
                dVec = tau[:, alpha] - tau[:, beta]
                d = np.linalg.norm(dVec) 
                # if d > sphereRadius:  # Skip if outside maximum sphere of boxes
                #     continue
                # distances.append(d**3)
                if ( beta > alpha ):
                    dHat = dVec / d
                    #sumR += q[alpha]*q[beta]/d
                    firstTerm = np.dot(spins[:,alpha], spins[:,beta])
                    secondTerm = 3*np.dot(spins[:, alpha], dHat)*np.dot(spins[:, beta], dHat)
                    # print('distance: ' + str(d))
                    # print('vector: ' + str(dHat))
                    # print('First term: ' +str(firstTerm))
                    # print('Second term: ' +str(secondTerm))
                    # print('E'+str(alpha)+str(beta)+': '+str((firstTerm - secondTerm) / d**3))
                    # print()
                    sumDipole += (firstTerm - secondTerm) / d**3
                    sumCell += (firstTerm - secondTerm) / d**3

    print('Sum of the cell: '+str(sumCell))
    sumEachRest = 0.0
    #Contributions of each site with the rest of the cells
    for alpha in range(len(tau[0])):
        for h in range (limInf, limSup):
            for k in range (-hmaxT, hmaxT+1):
                for l in range(-hmaxT, hmaxT+1):
                    T = h*a + k*b + l*c
                    if h == 0 and k == 0 and l==0:
                        continue
                    for beta in range (len(tau[0])):
                        dVec = tau[:, beta] - tau[:, alpha] + T
                        d = np.linalg.norm(dVec) 
                        dHat = dVec / d
                        # if d > sphereRadius:  # Skip if outside maximum sphere of boxes
                        #     continue
                        # distances.append(d**3)
                        if ( d != 0 ):
                            #sumR += q[alpha]*q[beta]/d
                            firstTerm = np.dot(spins[:,alpha], spins[:,beta])
                            secondTerm = 3*np.dot(spins[:, alpha], dHat)*np.dot(spins[:, beta], dHat)
                            # print('distance: ' + str(d))
                            # print('vector: ' + str(dHat))
                            # print('First term: ' +str(firstTerm))
                            # print('Second term: ' +str(secondTerm))
                            # print('E'+str(alpha)+' with rest: '+str((firstTerm - secondTerm) / d**3))
                            # print()
                            sumDipole += ((firstTerm - secondTerm) / d**3)/2.0
                            sumEachRest += ((firstTerm - secondTerm) / d**3)/2.0
        #                else:
        #                    print("d =0")
    print('Sum each atom with rest: '+str(sumEachRest))
    return sumDipole

def sumDipoles(a, b, c, tau, spins, hmaxT=20, parallel=True):
    # hmaxT defines the limit of the T vector
    if parallel:
        #Multiprocessing
        numCpus = cpu_count()

        def intervals(maximum, numCpus, a1, b1, c1):
            increment = (maximum*2+1) / numCpus
            if increment >= 1:
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, spins, maximum) for i in range(numCpus)]
            else:
                increment = 1
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, spins, maximum) for i in range((maximum*2+1))]

        ###########################    
        with Pool() as p:
            resReal = p.map(realSum, intervals(hmaxT, numCpus, a, b, c))
        realSpaceDipole = np.sum(resReal)
        ###########################

    else:
        realSpaceDipole = realSum([-hmaxT, hmaxT+1, a, b, c, tau, spins, hmaxT])

    return  realSpaceDipole