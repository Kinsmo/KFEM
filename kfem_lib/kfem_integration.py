import numpy as np
from numpy import sqrt

"(1D)"
def get_gauss_ref(gauss_n):
    if gauss_n==4:
        gauss_coeff=[0.3478548451,0.3478548451,0.6521451549,0.6521451549]
        gauss_point=[0.8611363116,-0.8611363116,0.3399810436,-0.3399810436]
    elif gauss_n==8:
        gauss_coeff=[0.1012285363,0.1012285363,0.2223810345,0.2223810345,0.3137066459,0.3137066459,0.3626837834,0.3626837834]
        gauss_point=[0.9602898565,-0.9602898565,0.7966664774,-0.7966664774,0.5255324099,-0.5255324099,0.1834346425,-0.1834346425]
    elif gauss_n==2:
        gauss_coeff=[1.,1]
        gauss_point=[-1/np.sqrt(3),1/np.sqrt(3)]
    return np.array(gauss_coeff),np.array(gauss_point)

def get_gauss_ref_local(gauss_coeff,gauss_point,x1,x2):
    gauss_coeff_local=(x2-x1)*gauss_coeff/2
    gauss_point_local=(x2-x1)*gauss_point/2+(x2+x1)/2
    return gauss_coeff_local,gauss_point_local

def gauss_quad(f,x1,x2,gauss_n=4):
    gauss_coeff,gauss_point = get_gauss_ref(gauss_n)
    gauss_coeff_local,gauss_point_local = get_gauss_ref_local(gauss_coeff,gauss_point,x1,x2)

    result=0
    for i in range(gauss_n):
        result += gauss_coeff_local[i]*f(gauss_point_local[i])
    return result


"(2D trangle)"
def get_gauss_ref_2d(gauss_n):
    if gauss_n==4:
        gauss_coeff=[(1-1/sqrt(3))/8,(1-1/sqrt(3))/8,(1+1/sqrt(3))/8,(1+1/sqrt(3))/8]
        gauss_point=[[(1/sqrt(3)+1)/2,(1-1/sqrt(3))*(1+1/sqrt(3))/4],
        [(1/sqrt(3)+1)/2,(1-1/sqrt(3))*(1-1/sqrt(3))/4],
        [(-1/sqrt(3)+1)/2,(1+1/sqrt(3))*(1+1/sqrt(3))/4],
        [(-1/sqrt(3)+1)/2,(1+1/sqrt(3))*(1-1/sqrt(3))/4]]
    elif gauss_n==9:
        gauss_coeff=[64/81*(1-0)/8,100/324*(1-sqrt(3/5))/8,100/324*(1-sqrt(3/5))/8,
        100/324*(1+sqrt(3/5))/8,100/324*(1+sqrt(3/5))/8,40/81*(1-0)/8,
        40/81*(1-0)/8,40/81*(1-sqrt(3/5))/8,40/81*(1+sqrt(3/5))/8]
        gauss_point=[[(1+0)/2,(1-0)*(1+0)/4],
        [(1+sqrt(3/5))/2,(1-sqrt(3/5))*(1+sqrt(3/5))/4],
        [(1+sqrt(3/5))/2,(1-sqrt(3/5))*(1-sqrt(3/5))/4],
        [(1-sqrt(3/5))/2,(1+sqrt(3/5))*(1+sqrt(3/5))/4],
        [(1-sqrt(3/5))/2,(1+sqrt(3/5))*(1-sqrt(3/5))/4],
        [(1+0)/2,(1-0)*(1+sqrt(3/5))/4],
        [(1+0)/2,(1-0)*(1-sqrt(3/5))/4],
        [(1+sqrt(3/5))/2,(1-sqrt(3/5))*(1+0)/4],
        [(1-sqrt(3/5))/2,(1+sqrt(3/5))*(1+0)/4]]
    elif gauss_n==3:
        gauss_coeff=[1/6,1/6,1/6]
        gauss_point=[[1/2,0],[1/2,1/2],[0,1/2]]
    return np.array(gauss_coeff),np.array(gauss_point)

def get_gauss_ref_local_2d(gauss_coeff,gauss_point,vertices_triangle):
    x1=vertices_triangle[0][0]
    y1=vertices_triangle[1][0]
    x2=vertices_triangle[0][1]
    y2=vertices_triangle[1][1]
    x3=vertices_triangle[0][2]
    y3=vertices_triangle[1][2]
    Jacobi=abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    gauss_coeff_local = gauss_coeff*Jacobi
    gauss_point_local = gauss_point
    gauss_point_local[:,0] = x1+(x2-x1)*gauss_point[:,0]+(x3-x1)*gauss_point[:,1]
    gauss_point_local[:,1] = y1+(y2-y1)*gauss_point[:,0]+(y3-y1)*gauss_point[:,1]
    return gauss_coeff_local,gauss_point_local

def gauss_quad_2d(f,vertices_triangle,gauss_n=3):
    gauss_coeff,gauss_point = get_gauss_ref_2d(gauss_n)
    gauss_coeff_local,gauss_point_local = get_gauss_ref_local_2d(gauss_coeff,gauss_point,vertices_triangle)
    result=0
    for i in range(gauss_n):
        result += gauss_coeff_local[i]*f(gauss_point_local[i][0],gauss_point_local[i][1])
    return result

if __name__ == "__main__":
    f = lambda x,y:1
    vertices_triangle = [[0,1,0],[0,0,1]]
    a = gauss_quad_2d(f,vertices_triangle)
    print(a)