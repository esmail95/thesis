from dolfin import *
from dolfin_adjoint import *
import pyipopt
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
# Turn off redundant output in parallel
parameters["std_out_all_processes"] = False
# Constants definition
mu = Constant(1.0) # viscosity
nu = Constant(0.001)
alphaunderbar = 2.5 * mu / (100**2) # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2) # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discretevaluedness

Re = Constant(100)

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

uFile = File(mesh.mpi_comm(),"output/u.pvd")
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


def forward(rho):
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    # Stokes equation
    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx + inner(grad(p), v) * dx + inner(div(u), q) * dx)
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    solve(lhs(F) == rhs(F), w, bc)
    return w
def solve_navier_stokes(W, nu, bcs):
    # Define variational forms
    w = Function(W)
    v, q = TestFunctions(W)
    u, p = split(w)
    F = (alpha(rho) * inner(u, v)* dx + inner(grad(u), grad(v))*dx + Re*inner(v, dot(grad(u),u))*dx - p*div(v)*dx - q*div(u)*dx)
    # Solve the problem
    solve(F == 0, w, bcs)
    uFile.write(u)
    pFile.write(p)
    return w

if __name__ == "__main__":
    rho = interpolate(Constant(float(V)/delta), A)
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
# Solve forward
    w = forward(rho)
    u, p = w.split()
# Store to file    
    # Solve Navier-Stokes
    w = solve_navier_stokes(W, nu, bcs=bc)
    u, p = w.split()

# Saving parameters
    controls = File("output/control_iterations_guess.pvd")
    allctrls = File("output/allcontrols.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz
    def derivative_cb(j, dj, rho):
        dj_viz.assign(dj)
        Gradient << dj_viz

    # Optimization problem
    # Define the Functional and Reduced Functional
    u, p = split(w)
    J = assemble(inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb,derivative_cb_post=derivative_cb)
    # Bound constraints
    lb = 0.0
    ub = 1.0

    volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)
    # Solve the optimisation problem with q = 0.01
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 50}
    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()
    rho_opt_xdmf = XDMFFile(MPI.comm_world,
    "output/control_solution_guess.xdmf")
    rho_opt_xdmf.write(rho_opt)
    # Reset the dolfin-adjoint state, and configure the new problem, Update values of q and rho
    q.assign(0.1)
    rho.assign(rho_opt)
    set_working_tape(Tape())
    rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
    rho_intrm.write(rho)
    w = solve_navier_stokes(W, nu, bcs = bc)
    (u, p) = split(w)
# Define the reduced functionals
    controls = File("output/control_iterations_final.pvd")
    rho_viz = Function(A, name="ControlVisualisation")

    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz
    def derivative_cb(j, dj, rho):
        dj_viz.assign(dj)
        Gradient << dj_viz

    J = assemble(inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
    
# Optimization problem starting with q = 0.1
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint, derivative_cb_post=derivative_cb)
    parameters = {'maximum_iterations': 200}
    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()
    rho_opt_final = XDMFFile(MPI.comm_world,  "output/control_solution_final.xdmf")
    rho_opt_final.write(rho_opt)
