import math

M = 0.
n = 1
while True:
    oldM = M
    for m in range(1,n+1,2):
            fac = 2
            if n == m:
                fac = 1
            M += fac * math.cosh(0.5*math.pi*math.hypot(n, m))**-2
    if abs(M - oldM) < 0.0001:
        break
    n += 2
print(-12*math.pi*M)