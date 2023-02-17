"""
author: yunxiao zhang
email: yunxiao9277@gmail.com
date: 2020.12.07
based on He Xiaoming FEM Course: chapter 1, page 47->84
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

"(03 define domian and mesh)"
a = 0
b = 1
N = 10      # number of mesh elements
Nm = N+1    # number of mesh nodes
h = (b-a)/N

"(04 define information matrix P and T)"
# node index of all mesh nodes [global node number]
P = np.linspace(a,b,Nm)
# global node indices of the mesh nodes of all the mesh elements, T[mesh number, global node number]
T = np.array([[i for i in range(N)],[i+1 for i in range(N)]]) 

Nb = N+1    # total number of basis functions
Nlb = 2     # number of local basis functions

# For the linear finite elements used here, Pb = P and Tb = T
Pb = P      # coordinates of all nodes
Tb = T      # global node indices of all the mesh elements

"(05 define boundary conditions)"
Nbn = 2     # number of boundary
boundary = [['Dirichlet','Dirichlet'],[0,N]]

# boundary function
def g(x):
    if x==0:
        return 0
    if x==1:
        return cos(1)

"(06 define local linear basis functions)"
def psi(x,E,beta):
    if beta==0:
        return (P[E[1]]-x)/h
    if beta==1:
        return (x-P[E[0]])/h

"(07 assemble matrix A and b)"
A = np.zeros((Nb,Nb))
b = np.zeros(Nb)

# algorithm 4 chap 1 page 62: compute A 
for n in range(N):
    E = Tb[:,n]
    for alpha in range(Nlb):
        for beta in range(Nlb):
            ff = lambda x: c(x)*(-1)**(alpha+beta)/h**2
            A[E[beta],E[alpha]] += quad(ff,P[E[0]],P[E[1]])[0]

# algorithm 5 chap 1 page 73: compute load vector b
for n in range(N):
    E = Tb[:,n]
    for beta in range(Nlb):
        ff = lambda l: f(l)*psi(l,E,beta)
        b[E[beta]] += quad(ff,P[E[0]],P[E[1]])[0]

# algorithm 6: handle Drichlet boundary
for k in range(Nbn):
    if boundary[0][k] == 'Dirichlet':
        i = boundary[1][k]
        A[i,:]=0
        A[i,i]=1
        b[i]=g(Pb[i])

"(08 solve Ax = b)"
# d = det(A)
u = solve(A, b)
u_exact = lambda x: x*cos(x)
uu = u_exact(P)

"(09 compute error)"
red = "\033[31m"
ok = "\033[0m"
error = max(u - uu)
print(f"{red}error: {error:.5f}{ok}")

"(10 exact solution uu for plot)"
a = 0
b = 1
hh = 0.01
xx = np.arange(a,b+hh,hh)
uu = u_exact(xx)

"(11 plot KFEM and exact solution)"
if 1:
    plt.plot(P,u,'bo',label="KFEM Solution")
    plt.plot(xx,uu,'-r',label="Exact Solution")

def broadcast(f,array,p=None):
    if len(p)==1:
        return [f(i,p[0]) for i in array]
    if len(p)==2:
        return [f(i,p[0],p[1]) for i in array]
    return [f(i) for i in array]

"(12 plot interpolation and base functions)"
todo = ['plot base functions']
if 1:
    for n in range(N):
        E = Tb[:,n]
        xx = [x for x in np.linspace(P[E[0]],P[E[1]],10)]
        for beta in range(Nlb):
            if 'plot base functions' in todo:
                plt.plot(xx,0.1*np.array(broadcast(psi,xx,p=[E,beta])),'-',color='0.5')
        psi_sum = np.sum([np.array(broadcast(psi,xx,p=[E,beta]))*u[E[beta]] for beta in range(Nlb)],axis=0)
        # plt.plot(xx,psi_sum)   
      

plt.legend()
plt.savefig("(KFEM 1D)(Local Basis Function)")  
plt.show()