"""
author: yunxiao zhang
email: yunxiao9277@gmail.com
date: 2020.12.02
based on He Xiaoming FEM Course: chapter 1, page 1->47
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos
from scipy.integrate import quad
from numpy.linalg import solve, det

"""
(01 problem to solve): 
-(cu')' = f     on domain [0,1]
u=g(x)          on boundary
"""

"(02 define functions)"
c = lambda x: exp(x)
f = lambda x: -exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))

"(03 define boundary condition)"
ua = 0
ub = cos(1)

"(04 define domian and mesh)"
a = 0
b = 1
N = 4       # number of mesh elements
h = (b-a)/N

"(05 define matrix)"
# to solve: Ax = b
A = np.zeros((N+1,N+1))
x = np.linspace(a,b,N+1)
b = np.zeros(N+1)

"(06 assemble matrix A)"
# algorithm 1 chap 1 page 36: compute stiffness matrix A
for j in range(N+1):
    if j<=N-1:
        A[j+1,j] = -1/h**2*quad(c,x[j],x[j+1])[0]
    if j>=1:
        A[j-1,j] = -1/h**2*quad(c,x[j-1],x[j])[0]
    if j>=1 and j<=N-1:
        A[j,j] = 1/h**2*(quad(c,x[j-1],x[j])[0]+quad(c,x[j],x[j+1])[0])

A[0,0] = 1/h**2*quad(c,x[0],x[1])[0]
A[N,N] = 1/h**2*quad(c,x[N-1],x[N])[0]

"(07 assemble b)"
# algorithm 2 chap 1 page 40: compute load vector b
for i in range(1,N):
    f1 = lambda l:f(l)*(l-x[i-1])/h
    f2 = lambda l:f(l)*(x[i+1]-l)/h
    b[i] = quad(f1,x[i-1],x[i])[0]+quad(f2,x[i],x[i+1])[0]

f1 = lambda l:f(l)*(x[1]-l)/h
b[0] = quad(f1,x[0],x[1])[0]
f1 = lambda l:f(l)*(l-x[N-1])/h
b[N] = quad(f1,x[N-1],x[N])[0]

"(08 implentment boundary condition)"
# algorithm 3 chap 1 page 4: handle Drichlet boundary
A[0,:]=0
A[0,0]=1
b[0]=ua

A[N,:]=0
A[N,N]=1
b[N]=ub

"(09 solve Ax = b)"
# d = det(A)
u = solve(A, b)

u_exact = lambda x: x*cos(x)
uu = u_exact(x)

"(10 compute error)"
# see chap 1 page 83 for error test
red = "\033[31m"
ok = "\033[0m"
error = max(u - uu)
print(f"{red}error: {error:.5f}{ok}")

"(11 plot)"
todo = ['plot']

if 'plot' in todo:
    # exact u for plot
    a = 0
    b = 1
    hh = 0.01
    xx = np.arange(a,b+hh,hh)
    uu = u_exact(xx)

    plt.plot(x,u,'bo',label="KFEM Solution")
    plt.plot(xx,uu,'-r',label="Exact Solution")
    plt.legend()
    plt.savefig('(KFEM 1D)')
    plt.show()