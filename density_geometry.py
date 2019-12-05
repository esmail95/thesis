from dolfin import *
from dolfin_adjoint import *

import shutil
import os
import csv
import time

import heaviside

# import the Python interface to IPOPT
try:
    import pyipopt
except ImportError:
    print("""This example depends on IPOPT and pyipopt. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

# delete existing folder
if os.path.isdir('results_dens_geometry'):
    shutil.rmtree('results_dens_geometry')

t0 = time.process_time()
# define constants and variables
mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.1) # q value that controls difficulty/discrete-valuedness of solution
h = 0.1

def alpha(eps):
    """Inverse permeability as a function of porosity eps"""
    return alphabar + (alphaunderbar - alphabar) * eps * (1 + q) / (eps + q)

# define mesh and function spaces
N = 100
delta = 1.0  # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(0.08*pi) * delta  # want the fluid to occupy 0.2513 of the domain

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N, 'crossed'))
A = FunctionSpace(mesh, "CG", 1)        # control function space
U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

#  Create files for storing solution
PorosityField = File("results_dens_geometry/PorosityField.pvd")
eps_viz = Function(A, name="PorosityField")
ufile = File("results_dens_geometry/velocity.pvd")
pfile = File("results_dens_geometry/pressure.pvd")
Sensitivity = File('results_dens_geometry/Sensitivity.pvd')
dj_viz = Function(A, name="Sensitivity")



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

# define forward problem
def forward(eps):
    """Solve the forward problem for a given fluid distribution eps(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(eps) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=1), 'on_boundary')
    bc2 = DirichletBC(W.sub(1), 0.0, PinPoint(), 'pointwise')
    bcs = [bc1,bc2]
    solve(lhs(F) == rhs(F), w, bcs=bcs)

    return w
    
    # define the ``__main__`` section
if __name__ == "__main__":
    eps = interpolate(heaviside.approx_heaviside_interpolate(interpolate(\
            Expression('peakwidth*sens/2-sens*fabs(center-pow(pow(x[0]/lx,2)+pow(x[1]/ly,2),0.5))',\
            degree=2,peakwidth=0.2,center=0.8,sens=1.0,lx=delta,ly=1.0), A), h), A)
    eps_viz.assign(eps)
    PorosityField << eps_viz
    w   = forward(eps)
    #(u, p) = split(w) doesn't work
    (u, p) = w.split(True) #this works

    # Save to file
    eps_viz.assign(eps)
    PorosityField << eps_viz
    ufile << u
    pfile << p
    
    # save results
    def eval_cb(j, eps):
        eps_viz.assign(eps)
        PorosityField << eps_viz
        # w_viz.assign(w)
        # ForwardStates << w_viz
        with open("results_dens_geometry/Results.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([j,assemble(eps*Constant(1)*dx)])   
    def derivative_cb(j, dj, eps):
        dj_viz.assign(dj)
        Sensitivity << dj_viz

    J = assemble(inner(alpha(eps) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(eps)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb, derivative_cb_post=derivative_cb)

     # define control constraints
        # bound constraints
    lb = 0.0
    ub = 1.0
        # volume constraints
    volume_constraint = UFLInequalityConstraint((V/delta - eps)*dx, m)

    # solve the optimisation problem with q = 0.1
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 250}

    solver = IPOPTSolver(problem, parameters=parameters)
    eps_opt = solver.solve()
    
    
   