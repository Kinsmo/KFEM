# chapter 5 page 40 example 1 of tianyuan fem course
# author: yunxiao zhang
# email: yunxiao9277@gmail.com
# date: 2021.1.13

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array, arange
from numpy.linalg import norm, solve, det
from mpl_toolkits.mplot3d import Axes3D 

from kfem_lib.kfem_basis_functions import *
from kfem_lib.kfem_meshing import *
from kfem_lib.kfem_functions import *
from kfem_lib.kfem_norms import *
from kfem_lib.kfem_integration import *

"""
# (01 PROBLEM)
# −∇·σ(u) = f   in 2d domain Ω [0,1]x[0,1]
# u = g         on boudary ∂Ω
# a(u,v) = (f,v)
"""

la = 1
mu = 2
f1 = lambda x,y: -(la+2*mu)*(-pi**2*sin(pi*x)*sin(pi*y)) \
                 -(la+mu)*(2*x-1)*(2*y-1) \
                 -mu*(-pi**2*sin(pi*x)*sin(pi*y))
f2 = lambda x,y: -(la+2*mu)*(2*x*(x-1)) \
                 -(la+mu)*pi**2*cos(pi*x)*cos(pi*y) \
                 -mu*(2*y*(y-1))
f = [f1,f2]

g1 = lambda x,y: sin(pi*x)*sin(pi*y)
g2 = lambda x,y: x*(x-1)*y*(y-1)
g = [g1,g2]

u_exact = g

"(02 MESH )"
a1=0
b1=1
c1=0
d1=1
N1=10
N2=10
N=2*N1*N2
P,T = trangular_2d(a1,b1,c1,d1,N1,N2)

"(03 BASIS FUNCTION TYPE)"
basis_function_type = "linear"

if basis_function_type=="linear":
    psi_test    = basis_function_linear_2d
    psi_trail   = basis_function_linear_2d
    Nb          = (N1+1)*(N2+1)
    Nlb_trial   = 3
    Nlb_test    = 3
    Pb          = P
    Tb_trail    = T
    Tb_test     = Tb_trail

"(04 BOUNDARY)"
boundary = np.concatenate((arange(N2+1),arange(N2+1)+N1*(N2+1),arange(1,N1)*(N2+1),arange(1,N1)*(N2+1)+N2))
# number of boundary node
Nbn = 2*(N1+N2) 

"(05 COMPUTE A)" 
def compute_A(c_in,r,s,p,q):
    c=c_in
    A = np.zeros((Nb,Nb))
    progress=0
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
                i = Tb_test[beta,n]
                j = Tb_trail[alpha,n]
                ff = lambda x,y: c*psi_trail(x,y,x1,y1,x2,y2,x3,y3,alpha,[r,s])*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q]) 
                A[i,j] += gauss_quad_2d(ff,vertices_triangle)
    return A

def A():
    A1 = compute_A(la,1,0,1,0)
    A2 = compute_A(mu,1,0,1,0)
    A3 = compute_A(mu,0,1,0,1)
    A4 = compute_A(la,0,1,1,0)
    A5 = compute_A(mu,1,0,0,1)
    A6 = compute_A(la,1,0,0,1)
    A7 = compute_A(mu,0,1,1,0)
    A8 = compute_A(la,0,1,0,1)
    A11 = A1+2*A2+A3
    A12 = A4+A5
    A21 = A6+A7
    A22 = A8+2*A3+A2
    A = np.vstack((np.hstack((A11,A12)),np.hstack((A21,A22))))
    return A

"(06 COMPUTE B)" 
def compute_b(f_in,p,q):
    f=f_in
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
            ff = lambda x,y: f(x,y)*psi_test(x,y,x1,y1,x2,y2,x3,y3,beta,[p,q])
            b[i] += gauss_quad_2d(ff,vertices_triangle)
    return b

def b():
    b1 = compute_b(f[0],0,0)
    b2 = compute_b(f[1],0,0)
    b = np.hstack((b1,b2)) 
    return b

A = A()
b = b()

"(07 BOUNDARY)" 
for k in range(Nbn):
    i = boundary[k]
    A[i,:]=0
    A[i,i]=1
    b[i]=g[0](Pb[0][i],Pb[1][i])  
    A[Nb+i,:]=0
    A[Nb+i,Nb+i]=1
    b[Nb+i]=g[1](Pb[0][i],Pb[1][i])  

"(08 SOLVE)"
u = solve(A, b)
uu1 = u_exact[0](Pb[0,:],Pb[1,:])
uu2 = u_exact[1](Pb[0,:],Pb[1,:])
uu = np.hstack((uu1,uu2))
#print(u)
#print(uu)

L1 = norm(u-uu,1)
L2 = norm(u-uu,2)
Linf = norm(u-uu,'inf')

red = "\033[31m"
ok = "\033[0m"
print(f'{red}L1 norm: {L1:.5f}\nL2 norm: {L2:.5f}\nL_inf norm: {Linf:.5f}{ok}')

"(09 PLOT)" 
todo = ['plot','plot 2d']

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


if 'x' in todo:
    plt.plot(u,'bo')
    plt.plot(uu,'-k.')
    plt.plot(boundary,[u[i] for i in boundary],'rs')

    
if 'plot 2d' in todo:
    plot2d(Nb,Pb,u[:Nb],uu[:Nb])
    plot2d(Nb,Pb,u[Nb:],uu[Nb:])

if todo != []:
    plt.show()