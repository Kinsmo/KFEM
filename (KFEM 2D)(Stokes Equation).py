"""
author: yunxiao zhang
email: yunxiao9277@gmail.com
date: 2021.01.13
based on He Xiaoming tianyuan FEM Course: chapter 6
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array, arange
from numpy.linalg import solve, det

from kfem_lib.kfem_basis_functions import *
from kfem_lib.kfem_meshing import *
from kfem_lib.kfem_functions import *
from kfem_lib.kfem_norms import *
from kfem_lib.kfem_integration import *

"""
# (01 PROBLEM)
# -∇·T(u,p) = f in domain Ω
# ∇·u=0         in domain Ω
# u=g in        on boudary ∂Ω

# week form:
# a(u,v) + b(v,p) = (f,v)
#          b(u,q) = 0
# 2d domain [0,1]x[0,1]
"""

mu = 2
f1 = lambda x,y: - 2*mu*x**2-2*mu*y**2-mu*exp(-y) + pi**2*cos(pi*x)*cos(2*pi*y)
f2 = lambda x,y: 4*mu*x*y-mu*pi**3*sin(pi*x) + 2*pi*(2-pi*sin(pi*x))*sin(2*pi*y)
f = [f1,f2]

g1 = lambda x,y: x**2*y**2+exp(-y)
g2 = lambda x,y: -2/3*x*y**3+2-pi*sin(pi*x)
g = [g1,g2]

p = lambda x,y: -(2-pi*sin(pi*x))*cos(2*pi*y)
u_exact = g

"===================== (02 MESH) ====================="
a1=0
b1=1
c1=-5
d1=0
N1=10
N2=10
N=2*N1*N2

# P: coordinates of all mesh nodes
# T: indices of all mesh elements
P,T = trangular_2d(a1,b1,c1,d1,N1,N2)

# Pb: coordinates of all finite element nodes, b means basis
# Tb: indices of all finite element elements, b means basis
Pb,Tb = trangular_2d_quadratic(a1,b1,c1,d1,N1,N2)

"# ===================== (03 BASIS TYPE) ===================== #"
# Taylor-Hood finite enelemts: mixing quadratic and linear basis function

phi = basis_function_quadratic_2d
Nlb = 6
Nb = (2*N1+1)*(2*N2+1)

psi = basis_function_linear_2d
Nlbp = 3
Nbp=(N1+1)*(N2+1)

"# ===================== (04 BOUNDARY) ===================== #"
boundary = np.concatenate((arange(2*N2+1),arange(2*N2+1)+2*N1*(2*N2+1),arange(1,2*N1)*(2*N2+1),arange(1,2*N1)*(2*N2+1)+2*N2))
# number of boundary nodes
Nbn = 4*(N1+N2) 

"# ===================== (05 COMPUTE A) ===================== #"
def compute_A(c,r,s,p,q,phi_trial,psi_test,Nb_trial,Nb_test,Nlb_trial,Nlb_test):
    progress = 0
    A = np.zeros((Nb_test,Nb_trial))
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
                progress +=1
                print_progress_bar(progress,N*Nlb_trial*Nlb_test,prefix='Compute A:')
                i = Tb[beta,n]
                if c == -1: 
                    # for psi = linear
                    j = T[alpha,n]
                else:
                    # for trial function
                    j = Tb[alpha,n] 

                ff = lambda x,y: c*phi_trial(x,y,x1,y1,x2,y2,x3,y3,alpha,[r,s])*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q]) 
                A[i,j] +=  gauss_quad_2d(ff,vertices_triangle)
    return A

# chap 6 page 42
def A():
    A1 = compute_A(mu,1,0,1,0,phi,phi,Nb, Nb,Nlb, Nlb) 
    A2 = compute_A(mu,0,1,0,1,phi,phi,Nb, Nb,Nlb, Nlb) 
    A3 = compute_A(mu,1,0,0,1,phi,phi,Nb, Nb,Nlb, Nlb) 
    A5 = compute_A(-1,0,0,1,0,psi,phi,Nbp,Nb,Nlbp,Nlb) 
    A6 = compute_A(-1,0,0,0,1,psi,phi,Nbp,Nb,Nlbp,Nlb) 
    O = np.zeros((Nbp,Nbp))

    A11 = 2*A1+A2
    A12 = A3
    A13 = A5
    A21 = A3.T
    A22 = 2*A2+A1
    A23 = A6
    A31 = A5.T
    A32 = A6.T
    A33 = O

    A = np.vstack(( np.hstack((A11,A12,A13)), 
                    np.hstack((A21,A22,A23)), 
                    np.hstack((A31,A32,A33))))
    return A

"# ===================== (06 COMPUTE B) ===================== #"
def compute_b(f,p,q,psi_test):
    b = np.zeros(Nb)
    for n in range(N):
        x1 = P[0,T[0,n]]
        y1 = P[1,T[0,n]]
        x2 = P[0,T[1,n]]
        y2 = P[1,T[1,n]]
        x3 = P[0,T[2,n]]
        y3 = P[1,T[2,n]]
        vertices_triangle = [[x1,x2,x3],[y1,y2,y3]]
        for beta in range(Nlb):
            i = Tb[beta,n]
            ff = lambda x,y: f(x,y)*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q])
            b[i] +=  gauss_quad_2d(ff,vertices_triangle)
    return b

def b():
    b1 = compute_b(f[0],0,0,phi)
    b2 = compute_b(f[1],0,0,phi)
    o = np.zeros(Nbp)
    b = np.hstack((b1,b2,o)) 
    return b

A = A()
b = b()

"# ===================== (07 BOUNDARY) ===================== #"
for k in range(Nbn):
    i = boundary[k]
    A[i,:]=0
    A[i,i]=1
    b[i]=g[0](Pb[0][i],Pb[1][i])  
    A[Nb+i,:]=0
    A[Nb+i,Nb+i]=1
    b[Nb+i]=g[1](Pb[0][i],Pb[1][i])   

"# ===================== (08 FIX P) ===================== #"
# Fix p at one point in the domain
ip = 0
A[2*Nb+ip,:]=0
A[2*Nb+ip,2*Nb+ip]=1  
b[2*Nb+ip] = p(P[0,ip],P[1,ip])


"# ===================== (09 SOLVE) ===================== #"
u = solve(A, b)

uu1 = u_exact[0](Pb[0,:],Pb[1,:])
uu2 = u_exact[1](Pb[0,:],Pb[1,:])
pp = p(P[0,:],P[1,:])
uu = np.hstack((uu1,uu2,pp))

"# ===================== (10 ERROR) ===================== #"
L2 = norm(u-uu,2)
Linf = norm(u-uu,'inf')

L1 = norm(u[:2*Nb]-uu[:2*Nb],1)
L2 = norm(u[:2*Nb]-uu[:2*Nb],2)
Linf = norm(u[:2*Nb]-uu[:2*Nb],'inf')
print(f'L1: {L1}\nL2: {L2}\nLinf: {Linf}')
"""
L1: 53.62103237052404
L2: 7.8678374114111955
Linf: 3.2349005733587872
"""

"# ===================== (11 PLOT )===================== #"

if 1:
    plt.plot(u,'bo')
    plt.plot(uu,'-k.')
    b = [u[i] for i in boundary]
    plt.plot(boundary,b,'rs')
    plt.plot(2*Nb+ip,u[2*Nb+ip],'g*')
    
    plt.show()

if 0:
    plt.plot(uu,'-k.')
    b = [uu[i] for i in boundary]
    plt.plot(boundary,b,'rs')
    plt.show()


def plot2d(Nb,Pb,u,uu):
    X = []
    Y = []
    for i in range(Nb):
        x = Pb[0][i]
        y = Pb[1][i]
        X.append(x)
        Y.append(y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, uu,c='k',marker='s',s=40)
    ax.scatter(X, Y, u,c='b',marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

if 1:
    plot2d(Nb,Pb,u[:Nb],uu[:Nb])
    plot2d(Nb,Pb,u[Nb:2*Nb],uu[Nb:2*Nb])
    plot2d(Nbp,P,u[2*Nb:],uu[2*Nb:])
    plt.show()