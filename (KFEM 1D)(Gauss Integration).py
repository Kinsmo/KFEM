"""
author: yunxiao zhang
email: yunxiao9277@gmail.com
date: 2020.12.13
based on He Xiaoming FEM Course
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array
from scipy.integrate import quad
from scipy.optimize import fmin
from numpy.linalg import solve, det
from math import floor, ceil
from kfem_lib.kfem_integration import gauss_quad

"""
# ===================== 01 PROBLEM ===================== #
# chap 1 page 128 example 3
#  -(cu')' = f
# u'(a)=ra
# u(b)=gb
# [0,1]
"""

c = lambda x: exp(x)
f = lambda x: -exp(x)*(cos(x)-2*sin(x)-x*cos(x)-x*sin(x))
g = lambda x: x*cos(x)

#c = lambda x: 1
#f = lambda x: -2
#g = lambda x: array(x)**2


"(# ===================== 02 MESH ===================== #)"
a = 0
b = 1
N = 4 # number of mesh elements
Nm = N+1 # number of mesh nodes
h = (b-a)/N


# P = np.arange(a,b+h,h) DONT USE THIS, +h is not exact bound h, use NP.LINSPACE instead
P = np.linspace(a,b,Nm)# matrix, coordinates of all mesh nodes [global node number]

T = np.array([[i for i in range(N)],[i+1 for i in range(N)]]) # matrix, global node indices of the mesh nodes of all the mesh elements, T[mesh number, global node number]


"(# ===================== 03 BASIS FUNCTIONS ===================== #)"    

def linear(x,x1,x2,i,o):
    """
    x: coordinate
    x1: left coordinate
    x2: right coordinate
    i: index of basis function
    o: order of basis function
    """
    def linear_r(x,i,o):
        if o==0:
            if i==0: return 1-x
            if i==1: return x
        if o==1:
            if i==0: return -1
            if i==1: return 1 
    h = x2-x1
    x_r = (x-x1)/h
    if o==0: return linear_r(x_r,i,o)
    if o==1: return 1/h*linear_r(x_r,i,o)

def quadratic(x,x1,x2,i,o):
    # at local frame
    """
    x: coordinate
    x1: left coordinate
    x2: right coordinate
    i: index of basis function
    o: derivative order of basis function
    """
    def quadratic_r(x,i,o):
        # at reference frame
        if o==0:
            if i==0: return 2*x**2-3*x+1
            if i==1: return 2*x**2-x
            if i==2: return -4*x**2+4*x
        if o==1:
            if i==0: return 4*x-3
            if i==1: return 4*x-1
            if i==2: return -8*x+4
    h = x2-x1
    x_r = (x-x1)/h
    if o==0: return quadratic_r(x_r,i,o)
    if o==1: return 1/h*quadratic_r(x_r,i,o)

"(# ===================== 04 BASIS TYPE ===================== #)"
basis_function_type = "quadratic"

if basis_function_type=="linear":
    psi_test = linear
    psi_trail = linear
    Nb = N+1 # scalar, total number of basis functions
    Nlb_trial = 2 # 
    Nlb_test = 2 # the number of local basis functions for test
    # For the linear finite elements we use here, Pb = P and Tb_test = T
    Pb = P # matrix, coordinates of all finite element nodes
    Tb_trail = T
    Tb_test = T# global node indices of the finite element nodes of all the mesh elements
    # reference to local

if basis_function_type=="quadratic":
    psi_test = quadratic
    psi_trail = quadratic
    Nb = 2*N+1
    Nlb_trial = 3
    Nlb_test = 3
    Pb = [a+k*h/2 for k in range(Nb)]
    Tb_trail = np.array([[i*2 for i in range(N)],
                [i*2+2 for i in range(N)],
                [i*2+1 for i in range(N)]])
    Tb_test = Tb_trail

"(# ===================== 05 COMPUTE A ===================== #)"
A = np.zeros((Nb,Nb))
b = np.zeros(Nb)

# algorithm 4: compute A
for n in range(N):
    x1 = P[T[0,n]]
    x2 = P[T[1,n]]
    for alpha in range(Nlb_trial):
        for beta in range(Nlb_test):
            i = Tb_test[beta,n]
            j = Tb_trail[alpha,n]
            ff = lambda x: c(x)*psi_trail(x,x1,x2,alpha,1)*psi_test(x,x1,x2,beta,1)
            A[i,j] += gauss_quad(ff,x1,x2)


"(# ===================== 06 COMPUTE B ===================== #)"
# algorithm 5: compute load vector b
for n in range(N):
    x1 = P[T[0,n]]
    x2 = P[T[1,n]]
    for beta in range(Nlb_test):
        i = Tb_test[beta,n]
        ff = lambda x: f(x)*psi_test(x,x1,x2,beta,0)
        b[i] += gauss_quad(ff,x1,x2)

"(# ===================== 07 BOUNDARY ===================== #)"
# type of boundary = Dirichlet, Neumann, Robin......
# Dirichlet: u(a)=b
# Neumann: u'(a)=b
# Robin: u(a)+qu'(a)=p
Nbn = 2 # number of boundary node
# boundary = [['Dirichlet','Nuemann'],[0,-1],[0,(cos(1)-sin(1))]]
boundary = [['Dirichlet','Robin'],[-1,0],[0,[-1,-1]]]

# algorithm 6: handle boundary
for k in range(Nbn):
    if boundary[0][k] == 'Dirichlet':
        i = boundary[1][k]
        A[i,:]=0
        A[i,i]=1
        b[i]=g(Pb[i])
    if boundary[0][k] == 'Nuemann':
        i = boundary[1][k]
        ri = boundary[2][k]
        b[i] += ri*c(Pb[i])
    if boundary[0][k] == 'Robin':
        i = boundary[1][k]
        qi = boundary[2][k][0]
        pi = boundary[2][k][1]
        A[i,i] += qi*c(Pb[i])
        b[i] += pi*c(Pb[i])

"(# ===================== 08 SOLVE ===================== #)"
# solve Ax=b
# d = det(A)
u = solve(A, b)
u_exact = g
uu = u_exact(Pb)

# error compare with chap 1 page 121
error = max(abs(u - uu))
red = "\033[31m"
ok = "\033[0m"
print(f"{red}error: {error:.5f}{ok}")

"(# ===================== 09 PLOT ===================== #)"
todo = ['plot','plot base functions']
#todo = []
# exact u for plot
a = 0
b = 1
hh = 0.01
xx = np.arange(a,b+hh,hh)
uu = u_exact(xx)

if 'plot' in todo:
    plt.plot(Pb,u,'bo',label="KFEM Solution")
    plt.plot(xx,uu,'-r',label="Exact Solution")
    #plt.show()

"(12 plot interpolation and base functions)"
ll = 0
if 'plot base functions' in todo:
    for n in range(N):
        x1 = P[T[0,n]]
        x2 = P[T[1,n]]
        xx = [x for x in np.linspace(x1,x2,100)]
        plt.vlines(x2,0,u_exact(x2),colors='0.9',linestyles='-')
        for beta in range(Nlb_test):
            i = Tb_test[beta,n]
            label = "Basis Functions" if ll == 0 else None
            ll = 1
            if 'plot base functions' in todo:
                plt.plot(xx,0.1*np.array(vectorize(psi_test)(xx,x1,x2,beta,0)),'-',color='0.5',label=label)
        #psi_sum = [np.array(vectorize(psi_test)(xx,x1,x2,beta,0)*u[Tb[beta,n]] for beta in range(Nlb_test)]
        psi_sum = np.sum([np.array(vectorize(psi_test)(xx,x1,x2,beta,0))*u[Tb_test[beta,n]] for beta in range(Nlb_test)],axis=0)
        # plt.plot(xx,psi_sum)   

plt.legend()        
plt.savefig("(KFEM 1D)(Gauss Integration)")    
plt.show()