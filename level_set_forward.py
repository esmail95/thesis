from dolfin import *
from dolfin_adjoint import *

import shutil
import os
import csv
import time

import heaviside


#Mesh and functional spaces
N = 145
delta = 1.0  # The aspect ratio of the domain, 1 high and \delta wide

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N, 'crossed'))
U = VectorElement("CG", mesh.ufl_cell(), 2)
P = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U*P) 





def g1(x):
    x0 = 0; x1 = 0; r_int = 0.7; r_ext = 0.9
    return between(x[0], (0, 1)) and between(x[1], (0, 1))\
    and (x[0] - x0)*(x[0] - x0) + (x[1] - x0)*(x[1] - x0) - r_int*r_int 

def g2(x):
    x0 = 0; x1 = 0; r_int = 0.7; r_ext = 0.9
    return between(x[0], (0, 1)) and between(x[1], (0, 1))\
    and (x[0] - x0)*(x[0] - x0) + (x[1] - x0)*(x[1] - x0) - r_ext*r_ext

#Pipe bend equation
def levelset(x):
    return (g1(x) or g2(x))

#Boundary of pipe bend subdomain, through level set function threshold g(x) = 0
def internalboundary_1(x):
        near_tol=1e-2
        return near(g1(x), 0, eps = near_tol) 


def internalboundary_2(x):
        near_tol=1e-2;r_ext = 0.9; r_int = 0.7
        return near(g2(x), 0, eps = near_tol)



#Define circle subdomain through level set function
class MySubDomain(SubDomain):
    def inside(self, x, on_boundary):
        near_tol=1e-2;r_ext = 0.9; r_int = 0.7
        return g1(x)>0 and g2(x)<0
mysubdomain = MySubDomain()


#Define measure with subdomains and meshfunction
mymeshfunction = MeshFunction('size_t', mesh, 2, 0)
mymeshfunction.set_all(0)
mysubdomain.mark(mymeshfunction, 1)
dx = dx(domain = mesh, subdomain_data = mymeshfunction)



l = 0.2
gbar = 1.0
u0 = Constant((0.0,0.0))

# pressure fixation point
class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.8) and near(x[1],0.0)

# define the boundary condition on velocity
class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 0.2
        gbar = 1.0
        if x[0] == 0.0:
            if (0.8 - l/2) < x[1] < (0.8 + l/2):
                t = x[1] - 0.8
                values[0] = gbar*(1 - (2*t/l)**2)
        if x[1] == 0.0: 
            if (0.8 - l/2) < x[0] < (0.8 + l/2):
                t = x[0] - 0.8
                values[1] = -gbar*(1 - (2*t/l)**2)
    def value_shape(self):
        return (2,)
# Define variational problem
w = Function(W)
w_adjoint = Function(W)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)


f = Constant((0.0, 0.0))
a = (inner(grad(u), grad(v)) + inner(grad(p), v) + inner(div(u), q))*dx\
    + ((inner(p,q)) + inner(u,v))*dx(0)
L = inner(f,v)*dx + inner(u0,v)*dx(0)

bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=1), 'on_boundary')
bc2 = DirichletBC(W.sub(1), 0.0, PinPoint(), 'pointwise')
bc3 = DirichletBC(W.sub(0), u0, internalboundary_1)
bc4 = DirichletBC(W.sub(0), u0, internalboundary_2)
bcs = [bc1,bc2,bc3, bc4]
solve(a == L, w, bcs = bcs)

# Split the mixed solution 
(u, p) = w.split(True)
    

#save results
subdomainfile = File("results_levelset_geometry/subdomains.pvd")
subdomainfile << mymeshfunction
ufile = File("results_levelset_geometry/velocity.pvd")
ufile << u
pfile = File("results_levelset_geometry/pressure.pvd")
pfile << p