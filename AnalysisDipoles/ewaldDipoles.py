import numpy as np
import math
from multiprocessing import Pool
from os import cpu_count

def reciprocalSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    astar = arguments[2]
    bstar = arguments[3]
    cstar = arguments[4]
    tau = arguments[5]
    spins = arguments[6]
    eta = arguments[7]
    hmaxg = arguments[8]
    sumDipoleReal = 0.
    sumDipoleImaginary = 0.

    # #Sum of the each atom with all the atoms within the cell (counting once)
    # if 0 in range(limInf, limSup):
    #     for alpha in range(len(q)):
    #         for beta in range (len(q)):
    #             d = np.linalg.norm(tau[:,alpha] - tau[:,beta])
    #             if ( beta > alpha ):
    #                 arg = np.dot(G,tau[:,alpha] - tau[:,beta])

    #Sum of each atom with atoms in the other unit cells
    for alpha in range(len(tau[0])):
        for h in range (limInf, limSup):
            for k in range (-hmaxg, hmaxg+1):
                for l in range (-hmaxg, hmaxg+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    G  = h*astar + k*bstar + l*cstar
                    G2 = np.linalg.norm(G)**2
                    for beta in range(len(tau[0])):
                        if G2 != 0:
                            arg = np.dot(G,tau[:,alpha] - tau[:,beta])
                            exp1 = math.exp(-(4.0*G2/eta))
                            sumDipoleReal += np.dot(spins[:, alpha], G)*np.dot(spins[:, beta], G) * math.cos(arg)*exp1/G2
                            sumDipoleImaginary += np.dot(spins[:, alpha], G)*np.dot(spins[:, beta], G) * math.sin(arg)*exp1/G2
    return sumDipoleReal, sumDipoleImaginary

def realSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    a = arguments[2]
    b = arguments[3]
    c = arguments[4]
    tau = arguments[5]
    spins = arguments[6]
    eta = arguments[7]
    hmaxT = arguments[8]
    sumDipole = 0.

    sumCell = 0.0
    #Sum of the each atom with all the atoms within the cell
    if 0 in range(limInf, limSup):
        for alpha in range(len(tau[0])):
            for beta in range(len(tau[0])):
                if ( alpha != beta ):
                    dVec = tau[:, alpha] - tau[:, beta]
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta])
                    dHat = dVec / d
                    alp = 0.5*math.sqrt(eta)
                    d2 = d**2
                    bSite = (math.erfc(alp*d) + math.exp(-(d2*alp**2))*2*alp*d/math.sqrt(math.pi))
                    cSite = (3*math.erfc(alp*d) + math.exp(-(d2*alp**2))*(3+2*d2*alp**2)*2*alp*d/math.sqrt(math.pi))
                    firstTerm = np.dot(spins[:,alpha], spins[:,beta]) * bSite
                    secondTerm = np.dot(spins[:, alpha], dHat)*np.dot(spins[:, beta], dHat)*cSite
                    sumDipole += (firstTerm - secondTerm)/d**3
                    sumCell += (firstTerm - secondTerm)/d**3
        print('Sum of the cell: '+str(sumCell))

    sumEachRest = 0.0
    #Sum of each atom with all other atoms in neighboring cells
    for alpha in range(len(tau[0])):
        for h in range (limInf, limSup):
            for k in range (-hmaxT, hmaxT+1):
                for l in range(-hmaxT, hmaxT+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    T = h*a + k*b + l*c
                    for beta in range(len(tau[0])):
                        dVec = tau[:, alpha] - tau[:, beta] + T
                        d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T)
                        dHat = dVec / d
                        if ( d != 0 ):
                            alp = 0.5*math.sqrt(eta)
                            bSite = (math.erfc(alp*d) + math.exp(-alp**2*d**2)*2*alp*d/math.sqrt(math.pi))
                            cSite = (3*math.erfc(alp*d) + math.exp(-alp**2*d**2)*(3+2*alp**2*d**2)*2*alp*d/math.sqrt(math.pi))
                            firstTerm = np.dot(spins[:,alpha], spins[:,beta]) * bSite
                            secondTerm = np.dot(spins[:, alpha], dHat)*np.dot(spins[:, beta], dHat)*cSite
                            sumDipole += (firstTerm - secondTerm)/d**3
                            sumEachRest += (firstTerm - secondTerm)/d**3
    
    print('Sum each atom with rest: '+str(sumEachRest))
    return sumDipole

def ewald(a, b, c, tau, spins, eta=4, hmaxg=20, hmaxT=20, parallel=True):
    # Defines the Ewald variables
    # eta is the convergent parameter for the Ewald's sum
    # hmaxg defines the Ewald limit of the G vector
    # hmaxT defines the Ewald limit of the T vector
    Vol = np.abs(np.dot(a, np.cross(b, c)))
    print('Volume is: ' +str(Vol))
    # Vol = float((aL)**3)

    # Calculating reciprocal lattice vectors
    astar =  (2*(math.pi/Vol))*np.cross(b,c)
    bstar =  (2*(math.pi/Vol))*np.cross(c,a)
    cstar =  (2*(math.pi/Vol))*np.cross(a,b)

    if parallel:
        #Multiprocessing
        numCpus = cpu_count()

        def intervals(maximum, numCpus, a1, b1, c1):
            increment = (maximum*2+1) / numCpus
            if increment >= 1:
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, spins, eta, maximum) for i in range(numCpus)]
            else:
                increment = 1
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, spins, eta, maximum) for i in range((maximum*2+1))]

        ###########################    
        with Pool() as p:
            resReciprocal = p.map(reciprocalSum, intervals(hmaxg, numCpus, astar, bstar, cstar))
        reciprocalDipoleReal = np.sum(list(zip(*resReciprocal))[0])
        reciprocalDipoleImaginary = np.sum(list(zip(*resReciprocal))[1])

        reciprocalDipoleReal = (4*math.pi/Vol)*reciprocalDipoleReal
        reciprocalDipoleImaginary = (4*math.pi/Vol)*reciprocalDipoleImaginary

        ###########################    
        with Pool() as p:
            resReal = p.map(realSum, intervals(hmaxT, numCpus, a, b, c))
        realSpaceDipole = np.sum(resReal)

        ###########################
        sumDipoleK = - (math.sqrt(eta)**3) / (math.sqrt(math.pi)*6)

    else:
        reciprocalDipoleReal, reciprocalDipoleImaginary = reciprocalSum([-hmaxg, hmaxg+1, astar, bstar, cstar, tau, spins, eta, hmaxg])
        reciprocalDipoleReal = (4.0*math.pi/Vol)*reciprocalDipoleReal / len(tau[0])
        reciprocalDipoleImaginary = (4.0*math.pi/Vol)*reciprocalDipoleImaginary / len(tau[0])

        realSpaceDipole = realSum([-hmaxT,  hmaxT+1, a, b, c, tau, spins, eta, hmaxT]) / len(tau[0])

        sumDipoleK = - (2.0*(math.sqrt(eta)/2.0)**3) / (math.sqrt(math.pi)*3.0)

    return reciprocalDipoleReal, reciprocalDipoleImaginary, realSpaceDipole, sumDipoleK