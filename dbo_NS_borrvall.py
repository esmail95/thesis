from dolfin import *
from dolfin_adjoint import *
import pyipopt
import dolfin 
import numpy 
import csv
import time
# Turn off redundant output in parallel
parameters["std_out_all_processes"] = False
# Constants definition
mu = Constant(1./50.) # viscosity  Constant(1./400.)
nu = Constant(0.01)
alphaunderbar = 2.5 * mu / (100**2) # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2) # parameter for \alpha
q = Constant(0.1) # q value that controls difficulty/discretevaluedness

Re = Constant(1)

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)
# Define the mesh (rectangle 1 high and delta wide)
N = 100
delta = 1.5 # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(1.0/3) * delta # want the fluid to occupy 1/3 of the

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0),
Point(delta, 1.0), N, N))
A = FunctionSpace(mesh, "CG", 1) # control function space
U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h) # mixed Taylor-Hood function
InvPermeability = File('output/alpha.pvd')


Gradient = File('output/Gradient.pvd')
dj_viz = Function(A, name="Gradient")

pFile = File(mesh.mpi_comm(),"output/p.pvd")

# Define boundary condition on velocity
class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0
        if x[0] == 0.0 or x[0] == delta:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
    
    def value_shape(self):
        return (2,)

class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], delta) and near(x[1],0.25)) or (near(x[0], delta) and near(x[1],0.75))



def solve_navier_stokes(W, bcs):
    # Define variational forms
    w = Function(W)
    v, q = TestFunctions(W)
    u, p = split(w)
    F = (alpha(rho) * inner(u, v)* dx + mu*inner(grad(u), grad(v))*dx + Re*inner(dot(grad(u),u), v)*dx - p*div(v)*dx + q*div(u)*dx)

    # Solve the problem
    solve(F == 0, w, bcs)

     
    return w

def forward(rho):
    """Solve the forward problem for a given fluid distribution eps(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=1), 'on_boundary')
    bc2 = DirichletBC(W.sub(1), 0.0, PinPoint(), 'pointwise')
    bcs = [bc1,bc2]
    solve(lhs(F) == rhs(F), w, bcs=bcs)

    return w
start = time.time()
if __name__ == "__main__":
    rho = interpolate(Constant(float(V)/delta), A)
    # rho = Function(A)
    # hdf5file=HDF5File(mesh.mpi_comm(), 'finalporosity.h5', 'r')
    # hdf5file.read(rho, '/porosity')
    bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    pinpoint = PinPoint()
    bc2 = DirichletBC(W.sub(1), 0.0, pinpoint, 'pointwise') 
    bc = [bc1, bc2]

    # w = solve_navier_stokes(W, bcs=bc)
    w = forward(rho)
    (u, p) = w.split(True)
   

# Saving parameters
    uFile= File("output/u.pvd")
    pFile = File("output/p.pvd")
    PorosityField = File("output/PorosityField.pvd")
    rho_viz = Function(A, name="PorosityField")
    rho_viz.assign(rho)
    alpha1 = project(alpha(rho), A)
    #alpha_viz.assign(alpha1)
    InvPermeability << alpha1
    PorosityField << rho_viz
    # uFile << u
    # pFile << p
    
    # save results
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        PorosityField << rho_viz
        alpha1 = project(alpha(rho), A)
        #alpha_viz.assign(alpha1)
        InvPermeability << alpha1
        uFile << u
        pFile << p
        # w_viz.assign(w)
        # ForwardStates << w_viz
        with open("output/Objective.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([j])  
        
        with open("output/Fluid volume.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([assemble(rho*Constant(1)*dx)])   
    def derivative_cb(j, dj, rho):
        dj_viz.assign(dj)
        Gradient << dj_viz

    J = assemble(0.5*inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx) #assemble(0.7*inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb, derivative_cb_post=derivative_cb)

     # define control constraints
        # bound constraints
    lb = 0.0
    ub = 1.0
        # volume constraints
    volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

    # solve the optimisation problem with q = 0.1
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 250}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()
end = time.time()
runtime = end - start

with open('output/Time.csv', mode='a') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([runtime])
uFile_opt =File('output/u_opt.pvd')
uFile_opt << u