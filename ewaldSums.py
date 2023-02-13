import numpy as np
import math

def ewald(aL, a, b, c, q, tau, eta=4, hmaxg=20, hmaxT=20):
    # Defines the Ewald variables
    # eta is the convergent parameter for the Ewald's sum
    # hmaxg defines the Ewald limit of the G vector
    # hmaxT defines the Ewald limit of the T vector
    Vol = float((aL)**3)

    # Calculating reciprocal lattice vectors
    astar =  (2*(math.pi/Vol))*np.cross(b,c)
    bstar =  (2*(math.pi/Vol))*np.cross(c,a)
    cstar =  (2*(math.pi/Vol))*np.cross(a,b)

    ###########################
    sumG_r = 0.
    sumG_i =0.
    alpha = 0
    for h in range (-hmaxg, hmaxg+1):
        for k in range (-hmaxg, hmaxg+1):
            for l in range (-hmaxg, hmaxg+1):
                G  = h*astar + k*bstar + l*cstar
                G2 = np.linalg.norm(G)**2
                for beta in range (len(q)):
                    arg = np.dot(G,tau[:,alpha] - tau[:,beta])
                    if G2 != 0:
                        sumG_r = sumG_r + q[alpha]*q[beta] * math.cos(arg)*math.exp(-G2/eta)/G2
                        sumG_i = sumG_i + q[alpha]*q[beta] * math.sin(arg)*math.exp(-G2/eta)/G2
    #                else:
    #                    print("G=0",h,k,l)
    sumG_r = 4*math.pi/Vol*sumG_r
    sumG_i = 4*math.pi/Vol*sumG_i
    print("Reciprocal space sum Real part : %10.8f" % sumG_r)
    print("Reciprocal space sum Img  part : %10.8f" % sumG_i)

    ###########################
    sumR = 0.
    alpha = 0
    for h in range (-hmaxT, hmaxT+1):
        for k in range (-hmaxT, hmaxT+1):
            for l in range (-hmaxT, hmaxT+1):
                T = h*a + k*b + l*c
                for beta in range (len(q)):
                    d = np.linalg.norm(tau[:,alpha] - tau[:,beta] + T) 

                    if ( d != 0 ):
                        sumR = sumR + q[alpha]*q[beta]*math.erfc(1/2.*math.sqrt(eta)*d)/d
    #                else:
    #                    print("d =0")
    print("Real space sum                 : %10.8f" % sumR)

    ###########################
    sumK = -math.sqrt(eta/math.pi)
    print("Constant term                  : %10.8f" % sumK)
    print()

    print("Madelung constant Vohra  : %10.8f" %  (sumG_r + sumK + sumR))
    print("Madelung constant Kittel : %10.8f" % ((sumG_r + sumK + sumR)*math.sqrt(3)/2))
    return  sumG_r, sumG_i, sumR, sumK