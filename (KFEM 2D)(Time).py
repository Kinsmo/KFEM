# chapter 4 of tianyuan fem course
# author: yunxiao zhang
# email: yunxiao9277@gmail.com
# date: 2021.1.13

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array, arange
from scipy.integrate import quad,dblquad, nquad
from scipy.optimize import fmin
from numpy.linalg import solve, det
from math import floor, ceil

from kfem_lib.kfem_basis_functions import *
from kfem_lib.kfem_meshing import *
from kfem_lib.kfem_integration import *

# ===================== PROBLEM ===================== #
# ut−∇·(c∇u) = f 
# int{domain}{c∇u·∇v} = int{domain}{fv} on 2d domain
# week form:
# (ut,v) + a(u,v) = (f,v)
# u=g on boundary
# u=u0 at t=0
# 2d domain [0,1]x[0,1]

c = lambda x,y,t: 2
f = lambda x,y,t: -3*exp(x+y+t)
g = lambda x,y,t: exp(x+y+t)
u_exact = g

# ===================== MESH ===================== #
a1=0
b1=1
c1=0
d1=1
N1=5
N2=5
N=2*N1*N2
P,T = trangular_2d(a1,b1,c1,d1,N1,N2)

# ===================== BASIS TYPE ===================== #
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

# ===================== BOUNDARY ===================== #
boundary = np.concatenate((arange(N2+1),arange(N2+1)+N1*(N2+1),arange(1,N1)*(N2+1),arange(1,N1)*(N2+1)+N2))
Nbn = 2*(N1+N2) # number of boundary node

# ===================== COMPUTE A ===================== #

def compute_A(c_in,t,r,s,p,q):
    c=c_in
    A = np.zeros((Nb,Nb))
    for n in range(N):
        x1 = P[0,T[0,n]]
        y1 = P[1,T[0,n]]
        x2 = P[0,T[1,n]]
        y2 = P[1,T[1,n]]
        x3 = P[0,T[2,n]]
        y3 = P[1,T[2,n]]
        vertices_triangle = [[x1,x2,x3],[y1,y2,y3]]
        for alpha in range(Nlb_trial):
            for beta in range(Nlb_test):
                i = Tb_test[beta,n]
                j = Tb_trail[alpha,n]
                ff = lambda x,y: c(x,y,t)*psi_trail(x,y,x1,y1,x2,y2,x3,y3,alpha,[r,s])*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q]) 
                A[i,j] += gauss_quad_2d(ff,vertices_triangle)
    return A

def At(t):
    A = compute_A(c,t,1,0,1,0) + compute_A(c,t,0,1,0,1)
    for k in range(Nbn):
        i = boundary[k]
        A[i,:]=0
        A[i,i]=1
    return A

M = compute_A(lambda x,y,t:1 ,0,0,0,0,0)

# ===================== COMPUTE B ===================== #

def compute_b(t,p,q):
    b = np.zeros(Nb)

    for n in range(N):
        x1 = P[0,T[0,n]]
        y1 = P[1,T[0,n]]
        x2 = P[0,T[1,n]]
        y2 = P[1,T[1,n]]
        x3 = P[0,T[2,n]]
        y3 = P[1,T[2,n]]
        vertices_triangle = [[x1,x2,x3],[y1,y2,y3]]
        for beta in range(Nlb_test):
            i = Tb_test[beta,n]
            ff = lambda x,y: f(x,y,t)*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q])
            b[i] += gauss_quad_2d(ff,vertices_triangle)
    return b

def bt(t):
    b = compute_b(t,0,0)
    for k in range(Nbn):
        i = boundary[k]
        b[i]=g(Pb[0][i],Pb[1][i],t)    
    return b

# ===================== ITERATE ===================== #
theta = 0.5
time_total = 1 #s
Tm = 5
dt = time_total/Tm
u = g(Pb[0,:],Pb[1,:],0)
U = []
for m in range(Tm):
    t2 = (m+1)*dt
    t1 = m*dt
    A2 = M/dt+theta*At(t2)
    b2 = theta*bt(t2) + (1-theta)*bt(t1) + (M/dt -(1-theta)*At(t1)) @ u

    u = solve(A2, b2)
    U.append(u)
    uu = u_exact(Pb[0,:],Pb[1,:],t2)
    # print(u)
    # print(uu)
    error = max(abs(u - uu))
    red = "\033[31m"
    ok = "\033[0m"
    print(f"{red}error: {error:.5f}{ok}")

    if 1:
        plt.plot(u,'bo')
        plt.plot(uu,'-k.')
        b = [u[i] for i in boundary]
        plt.plot(boundary,b,'rs')

plt.show()
