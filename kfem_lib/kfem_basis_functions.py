# all kinds of basis functions
# author: yunxiao zhang
# email: yunxiao9277@gmail.com
# date: 2021.1.18
# version: 3.0

# ===================== BASIS FUNCTIONS ===================== #     
# linear_1d()
# quadratic_1d()
# linear_2d() trangular mesh
# quadratic_2d() trangular mesh
# bilinear_2d() 

# ===================== 1D ===================== #     
def basis_function_linear_1d(x,x1,x2,i,o):
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

def basis_function_quadratic_1d(x,x1,x2,i,o):
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

# ===================== 2D ===================== #  
   
def basis_function_linear_2d(x,y,x1,y1,x2,y2,x3,y3,i,o):
    """
    x: coordinate
    x1: left coordinate
    x2: right coordinate
    i: index of basis function
    o: order of basis function
    """
    def linear_r(x,y,i,o):
        # (0,0) (1,0) (0,1)
        if o==[0,0]:
            if i==0: return 1-x-y
            if i==1: return x
            if i==2: return y
        if o==[1,0]:
            if i==0: return -1
            if i==1: return 1 
            if i==2: return 0
        if o==[0,1]:
            if i==0: return -1
            if i==1: return 0 
            if i==2: return 1


    J = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    x_r = ((y3-y1)*(x-x1)-(x3-x1)*(y-y1))/J
    y_r = (-(y2-y1)*(x-x1)+(x2-x1)*(y-y1))/J

    if o==[0,0]: 
        return linear_r(x_r,y_r,i,o)
    if o==[1,0]: 
        r = linear_r(x_r,y_r,i,[1,0])*(y3-y1)/J+\
            linear_r(x_r,y_r,i,[0,1])*(y1-y2)/J
        return r
    if o==[0,1]: 
        r = linear_r(x_r,y_r,i,[1,0])*(x1-x3)/J+\
            linear_r(x_r,y_r,i,[0,1])*(x2-x1)/J
        return r

def basis_function_bilinear_2d(x,y,x1,y1,x2,y2,x3,y3,x4,y4,i,o):
    """
    x: coordinate
    x1: left coordinate
    x2: right coordinate
    i: index of basis function
    o: order of basis function
    """
    def bilinear_r(x,y,i,o):
        # (0,0) (1,0) (0,1)
        if o==[0,0]:
            if i==0: return (1-x-y+x*y)/4
            if i==1: return (1+x-y-x*y)/4
            if i==2: return (1+x+y+x*y)/4
            if i==3: return (1-x+y-x*y)/4
        if o==[1,0]:
            if i==0: return (-1+y)/4
            if i==1: return (1-y)/4
            if i==2: return (1+y)/4
            if i==3: return (-1-y)/4
        if o==[0,1]:
            if i==0: return (-1+x)/4
            if i==1: return (-1-x)/4
            if i==2: return (1+x)/4
            if i==3: return (1-x)/4
        if o==[1,1]:
            if i==0: return 1/4
            if i==1: return -1/4
            if i==2: return 1/4
            if i==3: return -1/4
            
            
    h1=x2-x1
    h2=y4-y1
    x_r = (2*x-2*x1-h1)/h1
    y_r = (2*y-2*y1-h2)/h2

    if o==[0,0]: 
        return bilinear_r(x_r,y_r,i,o)
    if o==[1,0]: 
        r = bilinear_r(x_r,y_r,i,[1,0])*2/h1
        return r
    if o==[0,1]: 
        r = bilinear_r(x_r,y_r,i,[0,1])*2/h2
        return r
    if o==[1,1]: 
        r = bilinear_r(x_r,y_r,i,[0,1])*4/(h1*h2)
        return r

def basis_function_quadratic_2d(x,y,x1,y1,x2,y2,x3,y3,i,o):
    """
    x: coordinate
    x1: left coordinate
    x2: right coordinate
    i: index of basis function
    o: order of basis function
    """
    def quadratic_r(x,y,i,o):
        # (0,0) (1,0) (0,1)
        if o==[0,0]:
            if i==0: return 2*x**2+2*y**2+4*x*y-3*y-3*x+1
            if i==1: return 2*x**2-x
            if i==2: return 2*y**2-y
            if i==3: return -4*x**2-4*x*y+4*x
            if i==4: return 4*x*y
            if i==5: return -4*y**2-4*x*y+4*y
            
        if o==[1,0]:
            if i==0: return 4*x+4*y-3
            if i==1: return 4*x-1
            if i==2: return 0
            if i==3: return -8*x-4*y+4
            if i==4: return 4*y
            if i==5: return -4*y
        if o==[0,1]:
            if i==0: return 4*y+4*x-3
            if i==1: return 0
            if i==2: return 4*y-1
            if i==3: return -4*x
            if i==4: return 4*x
            if i==5: return -8*y-4*x+4
        if o==[2,0]:
            if i==0: return 4
            if i==1: return 4
            if i==2: return 0
            if i==3: return -8
            if i==4: return 0
            if i==5: return 0
        if o==[0,2]:
            if i==0: return 4
            if i==1: return 0
            if i==2: return 4
            if i==3: return 0
            if i==4: return 0
            if i==5: return -8
        if o==[1,1]:
            if i==0: return 4
            if i==1: return 0
            if i==2: return 0
            if i==3: return -4
            if i==4: return 4
            if i==5: return -4
    J = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    x_r = ((y3-y1)*(x-x1)-(x3-x1)*(y-y1))/J
    y_r = (-(y2-y1)*(x-x1)+(x2-x1)*(y-y1))/J

    if o==[0,0]: 
        return quadratic_r(x_r,y_r,i,o)
    if o==[1,0]: 
        r = quadratic_r(x_r,y_r,i,[1,0])*(y3-y1)+\
            quadratic_r(x_r,y_r,i,[0,1])*(y1-y2)
        return r/J
    if o==[0,1]: 
        r = quadratic_r(x_r,y_r,i,[1,0])*(x1-x3)+\
            quadratic_r(x_r,y_r,i,[0,1])*(x2-x1)
        return r/J
    if o==[2,0]: 
        r = quadratic_r(x_r,y_r,i,[2,0])*(y3-y1)**2+\
            quadratic_r(x_r,y_r,i,[1,1])*(y3-y1)*(y1-y2)+\
            quadratic_r(x_r,y_r,i,[0,2])*(y1-y2)**2
        return r/J**2
    if o==[0,2]: 
        r = quadratic_r(x_r,y_r,i,[2,0])*(x1-x3)**2+\
            quadratic_r(x_r,y_r,i,[1,1])*(x1-x3)*(x2-x1)+\
            quadratic_r(x_r,y_r,i,[0,2])*(x2-x1)**2
        return r/J**2
    if o==[1,1]: 
        r = quadratic_r(x_r,y_r,i,[2,0])*(x1-x3)*(y3-y1)+\
            quadratic_r(x_r,y_r,i,[1,1])*(x1-x3)*(y1-y2)+\
            quadratic_r(x_r,y_r,i,[1,1])*(x2-x1)*(y3-y1)+\
            quadratic_r(x_r,y_r,i,[0,2])*(x2-x1)*(y1-y2)
        return r/J**2            
