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
    eta = arguments[7]
    hmaxg = arguments[8]
    sumG_r = 0.
    sumG_i = 0.
    alpha = 0

    # #Sum of the each atom with all the atoms within the cell (counting once)
    # if 0 in range(limInf, limSup):
    #     for beta in range (len(q)):
    #         d = np.linalg.norm(tau[:,alpha] - tau[:,beta]) 
    #         if ( beta > alpha ):
    #             arg = np.dot(G,tau[:,alpha] - tau[:,beta])
    #             sumG_r = sumG_r + q[alpha]*q[beta] * math.cos(arg)*math.exp(-G2/eta)/G2
    #             sumG_i = sumG_i + q[alpha]*q[beta] * math.sin(arg)*math.exp(-G2/eta)/G2

    for h in range (limInf, limSup):
        for k in range (-hmaxg, hmaxg+1):
            for l in range (-hmaxg, hmaxg+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                G  = h*astar + k*bstar + l*cstar
                G2 = np.linalg.norm(G)**2
                for beta in range (len(q)):
                    arg = np.dot(G,tau[:,alpha] - tau[:,beta])
                    if G2 != 0:
                        sumG_r = sumG_r + q[alpha]*q[beta] * math.cos(arg)*math.exp(-G2/eta)/G2
                        sumG_i = sumG_i + q[alpha]*q[beta] * math.sin(arg)*math.exp(-G2/eta)/G2
    return sumG_r, sumG_i

def realSum(arguments):
    limInf = arguments[0]
    limSup = arguments[1]
    a = arguments[2]
    b = arguments[3]
    c = arguments[4]
    tau = arguments[5]
    q = arguments[6]
    eta = arguments[7]
    hmaxT = arguments[8]
    sumR = 0.
    alpha = 0

    #Sum of the each atom with all the atoms within the cell (counting once)
    if 0 in range(limInf, limSup):
        for beta in range (len(q)):
            d = np.linalg.norm(tau[:,alpha] - tau[:,beta]) 
            if ( d != 0 ):
                sumR = sumR + q[alpha]*q[beta]*math.erfc(1/2.*math.sqrt(eta)*d)/d

    for h in range (limInf, limSup):
        for k in range (-hmaxT, hmaxT+1):
            for l in range(-hmaxT, hmaxT+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                T = h*a + k*b + l*c
                for beta in range (len(q)):
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T) 
                    if ( d != 0 ):
                        sumR = sumR + q[alpha]*q[beta]*math.erfc(1/2.*math.sqrt(eta)*d)/d
    return sumR

def ewald(aL, a, b, c, q, tau, eta=4, hmaxg=20, hmaxT=20, parallel=True):
    # Defines the Ewald variables
    # eta is the convergent parameter for the Ewald's sum
    # hmaxg defines the Ewald limit of the G vector
    # hmaxT defines the Ewald limit of the T vector
    Vol = np.abs(np.dot(a, np.cross(b, c)))
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
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, eta, maximum) for i in range(numCpus)]
            else:
                increment = 1
                return [(round(i * increment - maximum), round((i + 1) * increment - maximum), a1, b1, c1, tau, q, eta, maximum) for i in range((maximum*2+1))]

        ###########################    
        with Pool() as p:
            resReciprocal = p.map(reciprocalSum, intervals(hmaxg, numCpus, astar, bstar, cstar))
        reciprocalReal = np.sum(list(zip(*resReciprocal))[0])
        reciprocalImaginary = np.sum(list(zip(*resReciprocal))[1])

        reciprocalReal = (4*math.pi/Vol)*reciprocalReal
        reciprocalImaginary = (4*math.pi/Vol)*reciprocalImaginary
        #print("Reciprocal space sum Real part : %10.8f" % reciprocalReal)
        #print("Reciprocal space sum Img  part : %10.8f" % reciprocalImaginary)

        ###########################    
        with Pool() as p:
            resReal = p.map(realSum, intervals(hmaxT, numCpus, a, b, c))
        realSpace = np.sum(resReal)

        #print("Real space sum                 : %10.8f" % realSpace)

        ###########################
        sumK = -math.sqrt(eta/math.pi)
        #print("Constant term                  : %10.8f" % sumK)
        #print()

        #print("Madelung constant Vohra  : %10.8f" %  (reciprocalReal + sumK + realSpace))
        #print("Madelung constant Kittel : %10.8f" % ((reciprocalReal + sumK + realSpace)*math.sqrt(3)/2))
    else:
        reciprocalReal, reciprocalImaginary = reciprocalSum([-hmaxg, hmaxg+1, astar, bstar, cstar, tau, q, eta, hmaxg])
        reciprocalReal = (4*math.pi/Vol)*reciprocalReal
        reciprocalImaginary = (4*math.pi/Vol)*reciprocalImaginary

        realSpace = realSum([-hmaxT,  hmaxT+1, a, b, c, tau, q, eta, hmaxT])

        sumK = -math.sqrt(eta/math.pi)

    return  reciprocalReal, reciprocalImaginary, realSpace, sumK