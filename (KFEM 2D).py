"""
author: yunxiao zhang
email: yunxiao9277@gmail.com
date: 2021.01.13
based on He Xiaoming FEM Course: chapter 4
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array, arange
from scipy.integrate import quad,dblquad, nquad
from numpy.linalg import solve, det

from kfem_lib.kfem_basis_functions import *
from kfem_lib.kfem_meshing import *

"""
# ===================== (01 PROBLEM) ===================== #
# ut-∇·(c∇u) = f 
# int{domain}{c∇u·∇v} = int{domain}{fv} on 2d domain
# week form:
# (ut,v) + a(u,v) = (f,v)
# u=g on boundary
# u=u0 at t=0
# 2d domain [0,1]x[0,1]
"""

c = lambda x,y: 2
f = lambda x,y: -3*exp(x+y)
g = lambda x,y: exp(x+y)
u_exact = g

"# ===================== (02 MESH) ===================== #"
a1=0
b1=1
c1=0
d1=1
N1=5
N2=5
N=2*N1*N2
P,T = trangular_2d(a1,b1,c1,d1,N1,N2)

"# ===================== (03 BASIS TYPE) ===================== #"
basis_function_type = "linear"

if basis_function_type=="linear":
    psi_test = basis_function_linear_2d 
    psi_trail = basis_function_linear_2d
    Nb=(N1+1)*(N2+1)
    Nlb_trial=3
    Nlb_test =3
    Pb=P
    Tb_trail=T
    Tb_test = Tb_trail

"# ===================== (04 BOUNDARY) ===================== #"
boundary = np.concatenate((arange(N2+1),arange(N2+1)+N1*(N2+1),arange(1,N1)*(N2+1),arange(1,N1)*(N2+1)+N2))
Nbn = 2*(N1+N2) # number of boundary node

"# ===================== (05 COMPUTE A) ===================== #"

def compute_A(c_in,r,s,p,q):
    c=c_in
    A = np.zeros((Nb,Nb))
    for n in range(N):
        x1 = P[0,T[0,n]]
        y1 = P[1,T[0,n]]
        x2 = P[0,T[1,n]]
        y2 = P[1,T[1,n]]
        x3 = P[0,T[2,n]]
        y3 = P[1,T[2,n]]
        for alpha in range(Nlb_trial):
            for beta in range(Nlb_test):
                i = Tb_test[beta,n]
                j = Tb_trail[alpha,n]
                ff = lambda x,y: c(x,y)*psi_trail(x,y,x1,y1,x2,y2,x3,y3,alpha,[r,s])*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q]) 
                if n%2==0:    
                    ll = lambda x: (y3-y2)*(x-x2)/(x3-x2)+y2
                    A[i,j] += dblquad(lambda y,x: ff(x,y), x1, x2, y1, lambda x: ll(x))[0]
                else:
                    ll = lambda x: (y1-y2)*(x-x2)/(x1-x2)+y2
                    A[i,j] += dblquad(lambda y,x: ff(x,y), x1, x2, lambda x: ll(x), y3)[0]
    return A

def A():
    A = compute_A(c,1,0,1,0) + compute_A(c,0,1,0,1)
    for k in range(Nbn):
        i = boundary[k]
        A[i,:]=0
        A[i,i]=1
    return A

M = compute_A(lambda x,y:1 ,0,0,0,0)

"# ===================== (06 COMPUTE B) ===================== #"

def compute_b(p,q):
    b = np.zeros(Nb)

    for n in range(N):
        x1 = P[0,T[0,n]]
        y1 = P[1,T[0,n]]
        x2 = P[0,T[1,n]]
        y2 = P[1,T[1,n]]
        x3 = P[0,T[2,n]]
        y3 = P[1,T[2,n]]
        for beta in range(Nlb_test):
            i = Tb_test[beta,n]
            ff = lambda x,y: f(x,y)*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q])
            if n%2==0:    
                ll = lambda x: (y3-y2)*(x-x2)/(x3-x2)+y2
                b[i] += dblquad(lambda y,x: ff(x,y), x1, x2, y1, lambda x: ll(x))[0]
            else:
                ll = lambda x: (y1-y2)*(x-x2)/(x1-x2)+y2
                b[i] += dblquad(lambda y,x: ff(x,y), x1, x2, lambda x: ll(x), y3)[0]
    return b

def b():
    b = compute_b(0,0)
    for k in range(Nbn):
        i = boundary[k]
        b[i]=g(Pb[0][i],Pb[1][i])    
    return b

"# ===================== (07 SOLVE) ===================== #"

A = A()
b = b()
u = solve(A, b)
uu = u_exact(Pb[0,:],Pb[1,:])

error = max(abs(u - uu))
red = "\033[31m"
ok = "\033[0m"
print(f"{red}error: {error:.5f}{ok}")

if 1:
    plt.plot(u,'bo', label="KFEM Solution")
    plt.plot(uu,'-r.', label="Exact Solution")
    b = [u[i] for i in boundary]
    plt.plot(boundary,b,'gs',label = "Fixed Boundary")

plt.savefig("(KFEM 2D)") 
plt.show()
