# chapter 3 of tianyuan fem course
# author: yunxiao zhang
# email: yunxiao9277@gmail.com
# date: 2020.12.16
# version: 1.0

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, exp, zeros, sin, cos, vectorize, array
from scipy.integrate import quad
from scipy.optimize import fmin
from numpy.linalg import solve, det
from math import floor, ceil

def trangular_2d(a,b,c,d,N1,N2):
    h1=(b-a)/N1
    h2=(d-c)/N2
    N=2*N1*N2
    Nm=(N1+1)*(N2+1)
    Nl=3 # number of local mesh nodes
    rn=N2+1
    cn=N1+1
    re=N2
    ce=N1
    # j global node index
    j = lambda cn,rn: cn*(N2+1)+rn

    # P coordinates of all mesh nodes
    P=zeros((2,Nm))
    for ci in range(cn):
        for ri in range(rn):
            P[0,j(ci,ri)] = a+ci*h1
            P[1,j(ci,ri)] = c+ri*h2
            
    # T global node indices of the mesh nodes of all the mesh elements
    T=zeros((3,N),dtype=int)
    for ci in range(ce):
        for ri in range(re):
            n = 2*(ci*N2+ri) # element index
            T[0,n]=j(ci,ri)
            T[1,n]=j(ci+1,ri)
            T[2,n]=j(ci,ri+1)
            n = n+1 # element index
            T[0,n]=j(ci,ri+1)
            T[1,n]=j(ci+1,ri)
            T[2,n]=j(ci+1,ri+1)
    return P,T

def trangular_2d_quadratic(a,b,c,d,N1,N2):
    h1=(b-a)/N1
    h2=(d-c)/N2
    N=2*N1*N2
    Nb=(2*N1+1)*(2*N2+1)
    Nbl=6 # number of local nodes
    rn=2*N2+1
    cn=2*N1+1
    re=N2
    ce=N1
    # j global node index
    j = lambda cn,rn: cn*(2*N2+1)+rn

    # Pb coordinates of all finite element nodes
    Pb=zeros((2,Nb))
    for ci in range(cn):
        for ri in range(rn):
            Pb[0,j(ci,ri)] = a+ci*h1/2
            Pb[1,j(ci,ri)] = c+ri*h2/2
            
    # Tb global node indices of the finite element nodes of all the finite elements
    j = lambda cn,rn: cn*(2*N2+1)+rn
    Tb=zeros((6,N),dtype=int)
    for ci in range(ce):
        for ri in range(re):
            n = 2*(ci*N2+ri) # element index
            Tb[0,n]=j(2*ci,2*ri)
            Tb[1,n]=j(2*ci+2,2*ri)
            Tb[2,n]=j(2*ci,2*ri+2)
            Tb[3,n]=j(2*ci+1,2*ri)
            Tb[4,n]=j(2*ci+1,2*ri+1)
            Tb[5,n]=j(2*ci,2*ri+1)

            n = n+1 # element index
            Tb[0,n]=j(2*ci,2*ri+2)
            Tb[1,n]=j(2*ci+2,2*ri)
            Tb[2,n]=j(2*ci+2,2*ri+2)
            Tb[3,n]=j(2*ci+1,2*ri+1)
            Tb[4,n]=j(2*ci+2,2*ri+1)
            Tb[5,n]=j(2*ci+1,2*ri+2)
    return Pb,Tb

def rectangular_2d(a,b,c,d,N1,N2):
    h1=(b-a)/N1
    h2=(d-c)/N2
    N=N1*N2
    Nm=(N1+1)*(N2+1)
    Nl=4
    rn=3
    cn=3
    re=2
    ce=2
    j = lambda cn,rn: cn*(N2+1)+rn

    P=zeros((2,Nm))
    for ci in range(cn):
        for ri in range(rn):
            P[0,j(ci,ri)] = a+ci*h1
            P[1,j(ci,ri)] = c+ri*h2
            
    T=zeros((4,N),dtype=int)
    for ci in range(ce):
        for ri in range(re):
            n = ci*N2+ri
            T[0,n]=j(ci,ri)
            T[1,n]=j(ci+1,ri)
            T[2,n]=j(ci+1,ri+1)
            T[3,n]=j(ci,ri+1)
    return P,T

if __name__ == "__main__":
    Pb,Tb = trangular_2d_quadratic(0,1,0,1,2,2)
    print(Pb)
    print(Tb)
