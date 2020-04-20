from dolfin import *
from dolfin_adjoint import *
import shutil
import os
import csv
import time

import heaviside

# Next we import the Python interface to IPOPT. If IPOPT is
# unavailable on your system, we strongly :doc:`suggest you install it
# <../../download/index>`; IPOPT is a well-established open-source
# optimisation algorithm.
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

if os.path.isdir('output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init'): 
    shutil.rmtree('output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init')
t0 = time.time()
# Next we define some constants, and define the inverse permeability as
# a function of :math:`\rho`.

mu = Constant(1./50.)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution
h = 0.1

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

# Next we define the mesh (a rectangle 1 high and :math:`\delta` wide)
# and the function spaces to be used for the control :math:`\rho`, the
# velocity :math:`u` and the pressure :math:`p`. Here we will use the
# Taylor-Hood finite element to discretise the Stokes equations
# :cite:`taylor1973`.

N = 100
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), N, N))
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

uFile = File(mesh.mpi_comm(),"output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/u.pvd")
Gradient = File('output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Gradient.pvd')
dj_viz = Function(A, name="Gradient")

pFile = File(mesh.mpi_comm(),"output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/p.pvd")

# Define the boundary condition on velocity
class PinPoint(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], delta) and near(x[1],0.25)) or (near(x[0], delta) and near(x[1],0.75))


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

# Next we define a function that given a control :math:`\rho` solves the
# forward PDE for velocity and pressure :math:`(u, p)`. (The advantage
# of formulating it in this manner is that it makes it easy to conduct
# :doc:`Taylor remainder convergence tests
# <../../documentation/verification>`.)


def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    # w = Function(W)
    # (u, p) = TrialFunctions(W)
    # (v, q) = TestFunctions(W)

    # F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
    #      inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    # bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    # solve(lhs(F) == rhs(F), w, bcs=bc)

    w = Function(W)
    u, p = split(w)
    test = TestFunction(W)
    v, q = split(test)
    Re = Constant(2.0)

    F = (alpha(rho) * inner(u, v) * dx + mu*inner(grad(u), grad(v)) * dx  + Re*inner(dot(grad(u), u), v)*dx +  inner(grad(p),v)* dx  + inner(div(u), q) * dx) #+ Re*inner(dot(grad(u), u), v)*dx 
          
    bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=1), 'on_boundary')
    bc2 = DirichletBC(W.sub(1), 0.0, PinPoint(), 'pointwise')
    bcs = [bc1] #bc2
    solve(F == 0, w, bcs=bcs)

    (u, p) = w.split(True)
    uFile.write(u)
    pFile.write(p)
    return w

# Now we define the ``__main__`` section. We define the initial guess
# for the control and use it to solve the forward PDE. In order to
# ensure feasibility of the initial control guess, we interpolate the
# volume bound; this ensures that the integral constraint and the bound
# constraint are satisfied.

if __name__ == "__main__":
    #rho = interpolate(Constant(float(V)/delta), A)
    rho = Function(A)
    hdf5file=HDF5File(mesh.mpi_comm(), 'finalporosity.h5', 'r')
    hdf5file.read(rho, '/porosity')
    # rho = interpolate(heaviside.approx_heaviside_interpolate(interpolate(\
    #         Expression('peakwidth*fabs(distance-pow(pow((2*x[1] - center),2),0.5)) + height',\
    #         degree=2,peakwidth=-1.2, distance=0.5, center=1, height=0.2), A), h), A)
    w   = forward(rho)
    (u, p) = split(w)
   

# With the forward problem solved once, :py:mod:`dolfin_adjoint` has
# built a *tape* of the forward model; it will use this tape to drive
# the optimisation, by repeatedly solving the forward model and the
# adjoint model for varying control inputs.
#
# As in the :doc:`Poisson topology example
# <../poisson-topology/poisson-topology>`, we will use an evaluation
# callback to dump the control iterates to disk for visualisation. As
# this optimisation problem (:math:`q=0.01`) is solved only to generate
# an initial guess for the main task (:math:`q=0.1`), we shall save
# these iterates in ``output/control_iterations_guess.pvd``.

    controls = File("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/control_iterations_guess.pvd")
    allctrls = File("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/allcontrols.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz
        with open("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Objective.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([j])  
        
        with open("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Fluid volume.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([assemble(rho*Constant(1)*dx)]) 
    def derivative_cb(j, dj, rho):
        dj_viz.assign(dj)
        Gradient << dj_viz

# Now we define the functional and :doc:`reduced functional
# <../maths/2-problem>`:

    J = assemble(inner(1.6*alpha(rho) * u, u) * dx +mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb,derivative_cb_post=derivative_cb)

# The control constraints are the same as the :doc:`Poisson topology
# example <../poisson-topology/poisson-topology>`, but this time we use
# the UFLInequalityConstraint class to demonstrate the ease of implementing
# inequality constraints with UFL.

    # Bound constraints
    lb = 0.0
    ub = 1.0

    # We want V - \int rho dx >= 0, so write this as \int V/delta - rho dx >= 0
    volume_constraint = UFLInequalityConstraint((V/delta - rho)*dx, m)

# Now that all the ingredients are in place, we can perform the initial
# optimisation. We set the maximum number of iterations for this initial
# optimisation problem to 20; there's no need to solve this to
# completion, as its only purpose is to generate an initial guess.

    # Solve the optimisation problem with q = 0.01
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 20} #20

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    rho_opt_xdmf = XDMFFile(MPI.comm_world, "output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/control_solution_guess.xdmf")
    rho_opt_xdmf.write(rho_opt)

# With the optimised value for :math:`q=0.01` in hand, we *reset* the
# dolfin-adjoint state, clearing its tape, and configure the new problem
# we want to solve. We need to update the values of :math:`q` and
# :math:`\rho`:

    q.assign(0.1)#0.1
    rho.assign(rho_opt)
    set_working_tape(Tape())

# Since we have cleared the tape, we need to execute the forward model
# once again to redefine the problem. (It is also possible to modify the
# tape, but this way is easier to understand.) We will also redefine the
# functionals and parameters; this time, the evaluation callback will
# save the optimisation iterations to
# ``output/control_iterations_final.pvd``.

    rho_intrm = XDMFFile(MPI.comm_world, "intermediate-guess-%s.xdmf" % N)
    rho_intrm.write(rho)

    w = forward(rho)
    (u, p) = split(w)

    # Define the reduced functionals
    controls = File("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/control_iterations_final.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz
        with open("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Objective.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([j])  
        
        with open("output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Fluid volume.csv", mode='a') as employee_file:
            employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([assemble(rho*Constant(1)*dx)]) 
    def derivative_cb(j, dj, rho):
        dj_viz.assign(dj)
        Gradient << dj_viz

    J = assemble(1.6*inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb,derivative_cb_post=derivative_cb)

# We can now solve the optimisation problem with :math:`q=0.1`, starting
# from the solution of :math:`q=0.01`:

    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {'maximum_iterations': 230}

    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    rho_opt_final = XDMFFile(MPI.comm_world, "output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/control_solution_final.xdmf")
    rho_opt_final.write(rho_opt)


t1 = time.time()
with open('output_mu=50_Re=2_1.6alpha_gradu_q=0.01_till_it=20_then_q=0.1_intermediate_init/Time.csv', mode='a') as employee_file:
    employee_writer = csv.writer(employee_file,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow([t1-t0])
# hdf5file=HDF5File(mesh.mpi_comm(), 'finalporosity.h5', 'w')
# hdf5file.flush()
# hdf5file.write(rho, '/porosity')