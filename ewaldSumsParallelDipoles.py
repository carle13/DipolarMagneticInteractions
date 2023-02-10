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
    q = arguments[6]
    spins = arguments[7]
    eta = arguments[8]
    hmaxg = arguments[9]
    sumG_r = 0.
    sumG_i = 0.
    sumDipoleReal = 0.
    sumDipoleImaginary = 0.
    alpha = 0
    for h in range (limInf, limSup):
        for k in range (-hmaxg, hmaxg+1):
            for l in range (-hmaxg, hmaxg+1):
                G  = h*astar + k*bstar + l*cstar
                G2 = np.linalg.norm(G)**2
                for beta in range (2):
                    arg = np.dot(G,tau[:,alpha] - tau[:,beta])
                    if G2 != 0:
                        sumG_r = sumG_r + q[alpha]*q[beta] * math.cos(arg)*math.exp(-G2/eta)/G2
                        sumG_i = sumG_i + q[alpha]*q[beta] * math.sin(arg)*math.exp(-G2/eta)/G2
                        sumDipoleReal += np.dot(spins[:, alpha], G)*np.dot(spins[:, beta], G) * math.cos(arg*2*math.pi)*math.exp(-G2*math.pi/eta)/G2
                        sumDipoleImaginary += np.dot(spins[:, alpha], G)*np.dot(spins[:, beta], G) * math.sin(arg*2*math.pi)*math.exp(-G2*math.pi/eta)/G2
    #                else:
    #                    print("G=0",h,k,l)
    return sumG_r, sumG_i, sumDipoleReal, sumDipoleImaginary

def realSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    a = arguments[2]
    b = arguments[3]
    c = arguments[4]
    tau = arguments[5]
    q = arguments[6]
    spins = arguments[7]
    eta = arguments[8]
    hmaxT = arguments[9]
    sumR = 0.
    sumDipole = 0.
    alpha = 0
    for h in range (limInf, limSup):
        for k in range (-hmaxT, hmaxT+1):
            for l in range(-hmaxT, hmaxT+1):
                T = h*a + k*b + l*c
                for beta in range (2):
                    site = tau[:, alpha] - tau[:, beta] + T
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T) 

                    if ( d != 0 ):
                        alp = 1/2.*math.sqrt(eta)
                        sumR = sumR + q[alpha]*q[beta]*math.erfc(alp*d)/d
                        bSite = (math.erfc(alp*d) + math.exp(-alp**2*d**2)*2*alp*d/math.sqrt(math.pi))/d**3
                        cSite = (3*math.erfc(alp*d) + math.exp(-alp**2*d**2)*(3+2*alp**2*d**2)*2*alp*d/math.sqrt(math.pi))/d**5
                        firstTerm = np.dot(spins[:,alpha], spins[:,beta]) * bSite
                        secondTerm = np.dot(spins[:, alpha], site)*np.dot(spins[:, beta], site)*cSite
                        sumDipole += firstTerm - secondTerm
    #                else:
    #                    print("d =0")
    return sumR, sumDipole

def ewald(aL, a, b, c, q, tau, spins, eta=4, hmaxg=20, hmaxT=20):
    # Defines the Ewald variables
    # eta is the convergent parameter for the Ewald's sum
    # hmaxg defines the Ewald limit of the G vector
    # hmaxT defines the Ewald limit of the T vector
    Vol = float((aL)**3)

    # Calculating reciprocal lattice vectors
    astar =  (2*(math.pi/Vol))*np.cross(b,c)
    bstar =  (2*(math.pi/Vol))*np.cross(c,a)
    cstar =  (2*(math.pi/Vol))*np.cross(a,b)

    #Multiprocessing
    numCpus = cpu_count()

    def intervals(maximum, numCpus, a1, b1, c1):
        increment = (maximum*2+1) / numCpus
        if increment >= 1:
            return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, spins, eta, maximum) for i in range(numCpus)]
        else:
            increment = 1
            return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, spins, eta, maximum) for i in range((maximum*2+1))]

    ###########################    
    with Pool() as p:
        resReciprocal = p.map(reciprocalSum, intervals(hmaxg, numCpus, astar, bstar, cstar))
    reciprocalReal = np.sum(list(zip(*resReciprocal))[0])
    reciprocalImaginary = np.sum(list(zip(*resReciprocal))[1])
    reciprocalDipoleReal = np.sum(list(zip(*resReciprocal))[2])
    reciprocalDipoleImaginary = np.sum(list(zip(*resReciprocal))[3])

    reciprocalReal = 4*math.pi/Vol*reciprocalReal
    reciprocalImaginary = 4*math.pi/Vol*reciprocalImaginary
    reciprocalDipoleReal = 4*math.pi/Vol*reciprocalDipoleReal
    reciprocalDipoleImaginary = 4*math.pi/Vol*reciprocalDipoleImaginary
    #print("Reciprocal space sum Real part : %10.8f" % reciprocalReal)
    #print("Reciprocal space sum Img  part : %10.8f" % reciprocalImaginary)

    ###########################    
    with Pool() as p:
        resReal = p.map(realSum, intervals(hmaxT, numCpus, a, b, c))
    realSpace = np.sum(list(zip(*resReal))[0])
    realSpaceDipole = np.sum(list(zip(*resReal))[1])

    #print("Real space sum                 : %10.8f" % realSpace)

    ###########################
    sumK = -math.sqrt(eta/math.pi)
    sumDipoleK = -math.sqrt(eta/math.pi)*(2/3)
    #print("Constant term                  : %10.8f" % sumK)
    #print()

    #print("Madelung constant Vohra  : %10.8f" %  (reciprocalReal + sumK + realSpace))
    #print("Madelung constant Kittel : %10.8f" % ((reciprocalReal + sumK + realSpace)*math.sqrt(3)/2))
    return  reciprocalReal, reciprocalImaginary, realSpace, sumK, reciprocalDipoleReal, reciprocalDipoleImaginary, realSpaceDipole, sumDipoleK