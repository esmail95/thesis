from dolfin import*

import ufl
import shutil
import os
import csv
import time

import reinit #hope to avoid this
from ufl import replace

# turn off redundant output in parallel
parameters['std_out_all_processes'] = False

#reduce info visualization
set_log_level(30)

# delete existing folder
if os.path.isdir('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2'):
    shutil.rmtree('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2')


# parameters
delta = 1.5
[lx,ly] = [delta, 1.0]

#Bent pipe
# Vmax = Constant(0.08*pi) * delta

#Borrvall
Vmax = Constant(1.0/3) * delta
mu = Constant(1./100.)

beta_list = [1e-2,1e-3,1e-4,1e-5]
beta = beta_list[0]
tol_list = [1e-2,5e-3,5e-4,5e-5]
tol = tol_list[0]
penal = 10
num_steps = 100
relOpt = 0.5
J_list = []

# LSF mesh
[Nx,Ny] = [150,100] #[100,100] 
mesh = Mesh(RectangleMesh(MPI.comm_world,Point(0.0, 0.0),Point(lx,ly),Nx,Ny,'crossed'))

#---------------------------------------------------------
# Define fixed subdomains, inlet and outlet sections
#---------------------------------------------------------  

#Borrvall
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (1.0/6.0, 2.0/6.0)) or between(x[1], (4.0/6.0, 5.0/6.0))) and near(x[0],0.0)  
inlet=Inlet()

class Inlet1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0/6.0, 2.0/6.0)) and near(x[0],0.0)  and on_boundary
inlet1=Inlet1()

class Inlet2(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (4.0/6.0, 5.0/6.0)) and near(x[0],0.0)  and on_boundary
inlet2=Inlet2()

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[1], (1.0/6.0, 2.0/6.0)) or between(x[1], (4.0/6.0, 5.0/6.0))) and near(x[0],delta) 
outlet=Outlet()

class Fixed(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0],0) or near(x[0],delta) or near(x[1],0) or near(x[1],1)) and on_boundary
fixed=Fixed()

class Fixed2(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0],0) and (between(x[1], (0.0, 1.0/6.0)) or between(x[1], (2.0/6.0, 4.0/6.0)) or between(x[1], (5.0/6.0, 1.0)))) or (near(x[0],delta) and (between(x[1], (0.0, 1.0/6.0)) or between(x[1], (2.0/6.0, 4.0/6.0)) or between(x[1], (5.0/6.0, 1.0)))) or near(x[1],0) or near(x[1],1) and on_boundary
fixed2=Fixed2()




#---------------------------------------------------------
# Define meshfunctions and function spaces
#---------------------------------------------------------  

#Functional spaces for forward, adjoint, LS, shape sensitivity
U = VectorElement("CG", mesh.ufl_cell(), 2)
P = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U*P)
A = FunctionSpace(mesh,'CG',1)
#Avec = VectorFunctionSpace(mesh,'CG',1,2)
Avec = VectorFunctionSpace(mesh, "Lagrange", 1)

#Meshfunctions
domain_markers = MeshFunction('size_t',mesh,mesh.topology().dim())
facet_markers = MeshFunction('size_t',mesh,mesh.topology().dim()-1) 



#---------------------------------------------------------
# Save results
#---------------------------------------------------------  
uFile = File(mesh.mpi_comm(),"results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/u.pvd")

pFile = File(mesh.mpi_comm(),"results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/p.pvd")

u_adjoint = File(mesh.mpi_comm(),"results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/u_adjoint.pvd")

p_adjoint = File(mesh.mpi_comm(),"results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/p_adjoint.pvd")

VelFile = File(mesh.mpi_comm(),"results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/Vel.pvd")

LevelSetFunction = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/LevelSetFunction.pvd')
phi_viz = Function(A)

DomainMarkers = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/DomainMarkers.pvd')
dom_viz = MeshFunction('size_t',mesh,mesh.topology().dim())

FacetMarkers = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/FacetMarkers.pvd')
facet_viz = MeshFunction('size_t',mesh,mesh.topology().dim()-1)

Gradient_shape = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/Gradient_shape.pvd')
g_viz = Function(A)

Gradient_top_smooth = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/Gradient_top_smooth.pvd')

Gradient_top_1 = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/Gradient_top_1.pvd')

Sensitivity = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/Sensitivity.pvd')
dj_viz = Function(A)

ForwardStates = File('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing_2/ForwardStates.pvd')
w_viz = Function(A)



#---------------------------------------------------------
# LSF declaration and initialization
#---------------------------------------------------------  
phi = Function(A)

#Straight pipe
# phi0 = Expression('peakwidth*sens/2-sens*fabs(center-x[0]/lx-x[1]/ly)',\
#         degree=2,peakwidth=0.2,center=0.8,sens=1.0,lx=lx,ly=ly)

#Borrvall
phi0 = Expression('peakwidth*fabs(distance-pow(pow((2*x[1] - center),2),0.5)) + height',\
            degree=2,peakwidth=-1.2, distance=0.5, center=1, height=0.2) 

phi.assign(interpolate(phi0,A))

#phi_viz.assign(phi)
#LevelSetFunction << phi_viz

# Reinitialization
phi = reinit.reinit(phi,A,min(lx/Nx,ly/Ny))


#---------------------------------------------------------
# LOOP FOR FORWARD, ADJOINT, HJ AND DESIGN UPDATE
#--------------------------------------------------------- 

#Iteration parameters
[it_max,it,it_start,factor,stop] = [250,0,0,1,False]

while it<it_max and stop==False:
    print('****************************************')
    print('ITERATION: '+str(it))
    print('****************************************')
    t0 = time.process_time()
    #Fluid subdomain
    class Fluid(SubDomain):
        def __init__(self,phi):
            self.phi = phi
            (SubDomain).__init__(self)
        def inside(self, x, on_boundary):
            tol = 1E-10
            return self.phi(x)>= 0 - tol

    #Domain/facet markers
    domain_markers.set_all(1)
    omega_f = Fluid(phi)
    omega_f.mark(domain_markers,0)
    dom_viz = domain_markers
    DomainMarkers << dom_viz

    print('Subdomains defined')

#---------------------------------------------------------
# Direct mesh mapping
#--------------------------------------------------------- 
    facet_markers.set_all(5)
    for fid in facets(mesh):
        domains = []
        for cid in cells(fid):
            domains.append(domain_markers[cid])
        if (domains == [0]):
            facet_markers[fid] = 2
        domains = list(set(domains))
        if (len(domains) > 1):
            facet_markers[fid] = 2

    inlet = Inlet()
    inlet.mark(facet_markers,3)
    outlet = Outlet()
    outlet.mark(facet_markers,4)

    

#---------------------------------------------------------
# Measures and normal
#--------------------------------------------------------- 
    
    n = FacetNormal(mesh)
    dx = Measure('dx',domain=mesh,subdomain_data=domain_markers)
    ds = Measure('ds',domain=mesh,subdomain_data=facet_markers)
    dS = Measure('dS',domain=mesh,subdomain_data=facet_markers)
    
    

#---------------------------------------------------------
# Boundary conditions
#---------------------------------------------------------   

    #Borrvall
    #Boundary conditions forward

    #Pressure fixation point
    class PinPoint(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], delta) and near(x[1],0.25)) or (near(x[0], delta) and near(x[1],0.75))

    #Velocity
    class InflowOutflow(UserExpression):
        def eval(self, values, x):
            values[1] = 0.0
            values[0] = 0.0
            l = 1.0/6.0
            gbar = 3.0

            if x[0] == 0.0 or x[0] == delta:
                if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                    t = x[1] - 1.0/4
                    values[0] = gbar*(1 - (2*t/l)**2)
                if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                    t = x[1] - 3.0/4
                    values[0] = gbar*(1 - (2*t/l)**2)

        def value_shape(self):
            return (2,)
    
    

    u0 = Constant((0.0,0.0))
    inflowoutflow = InflowOutflow(degree=1)
    pinpoint = PinPoint()
    uin1 = Expression(("(1-(2*(x[1]-0.25)/1/6)*(2*(x[1]-0.25)/1/6))","0"),degree=2)
    uin2 = Expression(("(1-(2*(x[1]-0.75)/1/6)*(2*(x[1]-0.75)/1/6))","0"),degree=2)


    #Boundary conditions adjoint
    u0_adjoint = Constant((0.0,0.0))

#---------------------------------------------------------
# Forward problem statement
#---------------------------------------------------------   
    #Forward problem
    w =  Function(W)
    u, p = split(w)
    test = TestFunction(W)
    v, q = split(test)

    u0 = Constant((0.0,0.0))
    rho = Constant(1.0)

    a1 = (mu*inner(grad(u), grad(v)) - p*div(v) + rho*inner(dot(grad(u), u), v) + div(u)* q)*dx(0) 
    a2 =  ((inner(p,q)) + inner(u,v))*dx(1) 
    a = a1 + a2

    bc1 = DirichletBC(W.sub(0), inflowoutflow, 'on_boundary')
    bc3 = DirichletBC(W.sub(0), u0, facet_markers, 2)
    # bc2 = DirichletBC(W.sub(1), 0.0, outlet) #pinpoint, 'pointwise'
    # bc1 = DirichletBC(W.sub(0), uin1, inlet1)
    bc1a = DirichletBC(W.sub(0), u0, fixed2)
    bcs = [bc1, bc1a, bc3]
    
#---------------------------------------------------------
# Adjoint problem statement: Lagrangian
#---------------------------------------------------------
    #Adjoint problem
    w_adjoint = Function(W)
    u_adj, p_adj = split(w_adjoint)
    J = mu * inner(grad(u), grad(u)) * dx 
    L = replace(a, {test:w_adjoint}) + J

    bc1_adj = DirichletBC(W.sub(0), u0_adjoint, inlet)
    bc2_adj = DirichletBC(W.sub(0), u0_adjoint, fixed) 
    bc3_adj = DirichletBC(W.sub(0), u0_adjoint, facet_markers, 2)
    bc0 = [bc1_adj, bc2_adj, bc3_adj]
    #solve(derivative(L, w)==0, w_adjoint, bcs=bc0)
    
#---------------------------------------------------------
# Solve Forward and Adjoint problems
#---------------------------------------------------------
    def solve_state_and_adjoint():
        print('Solving forward problem...')
        solve(a==0,w,bcs=bcs)
        print('Solving adjoint problem...')
        solve(derivative(L, w)==0, w_adjoint,bcs=bc0)
        (u, p) = w.split(True)
        (u_adj, p_adj) = w_adjoint.split(True)
        uFile.write(u)
        pFile.write(p)
        u_adjoint.write(u_adj)
        p_adjoint.write(p_adj)
    
    solve_state_and_adjoint()  

#---------------------------------------------------------
# Steepest descent. Shape derivative: Duan and Challis
#---------------------------------------------------------

    G_shape = - mu*inner(grad(u), grad(u)) + mu*inner(grad(u), grad(u_adj)) 
    #--------------------------------------------------------------------------    
    #Schmidt: mu*inner(grad(u), grad(u)) + mu*inner(grad(u), grad(u_adj)) 
    #Duan: - mu*inner(grad(u), grad(u)) + mu*inner(grad(u), grad(u_adj))
    #--------------------------------------------------------------------------  
    lam_v = - assemble(G_shape('-')*dS(2))/assemble(Constant(1)*dS(2))
    lam = conditional(lt(assemble(Constant(1)*dx(0)),Vmax),0,penal*lam_v)
    G_shape_vol = -G_shape - lam
    G = project(- G_shape - lam, A)
    Gradient_shape << G
#---------------------------------------------------------
# Topological derivative: He, Chiu, Osher
#---------------------------------------------------------   
    def sign(phi):
        eps = Constant(0.01)
        sign = phi/sqrt(phi*phi + eps*eps)
        return sign

    G_top_1 = - sign(phi)* G_shape
    G_top = project(G_top_1, A)    

#---------------------------------------------------------
# Laplace filtering
#---------------------------------------------------------
    
    class Fixed1(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0],0) or near(x[0],delta)) and on_boundary
    fixed1=Fixed1()

    print('Laplace filtering for shape gradient ...')
    Vel = Function(Avec)
    xi = TestFunction(Avec)
    n_phi = grad(phi)
    F = inner(grad(Vel),grad(xi))*dx + inner(G*n_phi('-'),xi('+'))*dS(2) - inner(G*n,xi)*ds(2) #- inner(G*n('-'),xi('+'))*dS(2)
    #--------------------------------------------------------------------------  
    #douan:inner(grad(Vel),grad(xi))*dx + inner(G*grad(phi),xi)*dx 
    #schmidt:inner(grad(Vel),grad(xi))*dx - inner(G*grad(phi),xi)*dx
    #--------------------------------------------------------------------------  


    #BCs for advective term smoothing in HJ, inlet and outlet do not change

    bc_smooth1 = DirichletBC(Avec,[0,0],inlet)
    bc_smooth2 = DirichletBC(Avec,[0,0],outlet)
    bc_smooth3 = DirichletBC(Avec,[0,0], fixed1)
    bcs = [bc_smooth1,bc_smooth2] #, bc_smooth3 

    solve(F == 0, Vel, bcs=bc_smooth3)
    VelFile << Vel


    print('Laplace filtering for topological derivative ...')
    G_top_s = TrialFunction(A)
    yi = TestFunction(A)
    R = inner(grad(G_top_s),grad(yi))*dx  - inner(G_top_1,yi)*dx 
    #--------------------------------------------------------------------------  
    #douan: dot(grad(G_top_s),grad(yi))*dx  - inner(G_top_1,yi)*dx 
    #schmidt: dot(grad(G_top_s),grad(yi))*dx  + inner(G_top_1,yi)*dx 
    #--------------------------------------------------------------------------  

    #BCs for advective term smoothing in HJ, inlet and outlet do not change
    bc_smooth1 = DirichletBC(A, 0.0, inlet)
    bc_smooth2 = DirichletBC(A, 0.0, outlet)
    bc_smooth3 = DirichletBC(A, 0.0, fixed)
    G_top_s = Function(A)
    bcs = [bc_smooth1,bc_smooth2,bc_smooth3] 

    solve(lhs(R)==rhs(R), G_top_s, bcs=bcs)
    Gradient_top_smooth << G_top_s

#Smoothing accelerates algorithm
#---------------------------------------------------------
# Modified Hamilton-Jacobi: Burger
#---------------------------------------------------------
    print('Solving Hamilton-Jacobi equation...')
    Vel_normal = project(dot(Vel,-grad(phi)/norm(project(grad(phi),Avec))),A)
    t = 0
    dt = beta*(min(lx/Nx,ly/Ny)/max(abs(Vel_normal.vector().get_local()[:])))  #dt to satisfy CFL condition    
    #omega = Constant(1)  
    omega = norm(Vel)/norm(G_top_s)
    #omega = Constant(50)

    for n in range(num_steps):
        t += dt
        phi_new = phi - dt*inner(Vel,grad(phi)) + G_top_s*omega*dt 
        phi.assign(project(phi_new,A))
     
#---------------------------------------------------------
# Objective function
#---------------------------------------------------------  
    V_occ = assemble(Constant(1)*dx(0))
    J = float(assemble(mu*inner(grad(u),grad(u))*dx))
    print('Objective value: ', J)
    print('Occupied volume: ', V_occ)
    J_list.append(J)

#---------------------------------------------------------
# Stopping criterion
#---------------------------------------------------------
    
    if (it>50) and (it_start>5)\
        and (abs((J_list[it-2]-J_list[it])/J_list[it])<tol)\
        and (abs((J_list[it-4]-J_list[it])/J_list[it])<tol)\
        and (abs((J_list[it-3]-J_list[it-1])/J_list[it-1])<tol)\
        and (abs((J_list[it-5]-J_list[it-1])/J_list[it-1])<tol):   
        print('Converged')        

    elif (it+1 == it_max):
        print('Maximum number of iterations reached')
        it+=1
        stop = True
    
    if stop == False:
        phi_viz.assign(phi)
        #LevelSetFunction << phi_viz         
        reinit_it = 3*factor
        if it == reinit_it:
            #Reinitialization
            print('Reinitializing level-set function...')
            phi.assign(reinit.reinit(phi,A,min(lx/Nx,ly/Ny)))
            factor+=1   
        
        # go to next iteration
        it_start+=1
        it+=1
        t1 = time.process_time()
        with open('results_ls_top_NS_borrvall_delta=1.5_adjoint1_top_smoothing/Time.csv', mode='a') as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            employee_writer.writerow([t1-t0])

       