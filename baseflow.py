from fenics import *
from ffc.plot import PointSecondDerivative
from mshr import *

def navier_stokes(mesh, boundaries, nu, pin, pout, inflow_marker, outflow_marker, no_slip_marker):

    # discrete function space 
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    TH = MixedElement([P2, P1])
    W  = FunctionSpace(mesh, TH)

    up = Function(W)
    (u,p) = split(up)
    (v,q) = TestFunctions(W)

    # boundary conditions 
    bcu_noslip = DirichletBC(W.sub(0), Constant((0.0,0.0,0.0)), boundaries, no_slip_marker[0])
    
    dx = Measure('dx', domain = mesh)
    ds = Measure('ds', domain = mesh,  subdomain_data = boundaries)

    n    = FacetNormal(mesh)
    nu   = Constant(nu)
    f    = Constant((0,0,0))

    a    = Constant(0.5*nu)*inner(D(u), D(v))*dx
    a   +=  - div(v)*p*dx +  q*div(u)*dx + dot(dot(u, nabla_grad(u)),v)*dx
    
    L = dot(f, v)*dx

    for im in inflow_marker:
        a += -Constant(nu)*dot(nabla_grad(u)*n, v)*ds(im)
        L += - Constant(pin)*dot(n,v)*ds(im)
    
    for om in outflow_marker:
        a += -Constant(nu)*dot(nabla_grad(u)*n, v)*ds(om)
        L += - Constant(pout)*dot(n,v)*ds(om)

    
    # solve 
    F = a - L
    dF = derivative(F,up)

    PETScOptions.set("mat_mumps_icntl_4", 1) # level of printing from the solver (0-4)
    PETScOptions.set("mat_mumps_icntl_14", 300) # percentage increase in the working space wrt memory
    problem = NonlinearVariationalProblem(F, up, bcu_noslip, dF )
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 5E-8
    prm['newton_solver']['relative_tolerance'] = 5E-14
    prm['newton_solver']['maximum_iterations'] = 100
    prm['newton_solver']['relaxation_parameter'] = 1.0
    prm["newton_solver"]["linear_solver"] = "mumps"
    solver.solve()

    u, p = up.split(deepcopy=True)

    return u, p

def make_pipe_mesh(radius, nelem):
    # define a cylinder domain
    cylinder = Cylinder(Point(0,0,0), Point(1,0,0), radius,radius)
    geometry = cylinder
    # define the mesh 
    mesh = generate_mesh(geometry, nelem)
    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    # mark the inflow, outflow and the walls
    class Noslip(SubDomain):
        def inside(self, x, on_boundary):
             return on_boundary and x[1]**2 + x[2]**2 > 0.95*radius**2
    class Inflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],0)

    class Outflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0],1)

    velocity = Noslip()
    velocity.mark(boundaries, 1)

    pressure_inflow = Inflow()
    pressure_inflow.mark(boundaries,2)

    pressure_outflow = Outflow()
    pressure_outflow.mark(boundaries,3)
    return mesh, boundaries


def D(u):
    grad_u = grad(u)
    return grad_u + grad_u.T

pin  = 2.0
pout = 1.0
mu   = 1.0
radius  = 1.0
nelem = 15

poise_case = 0
artery_case = 1

if poise_case:
    # Make pipe mesh so we can solve for Poiseuille flow
    mesh, boundaries = make_pipe_mesh(radius, nelem)
    inflow_marker = [2]
    outflow_marker = [3]
    no_slip_marker = [1]

if artery_case:
    # Get artery mesh
    mesh = Mesh('Case_test_71.xml.gz')
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
    inflow_marker = [1]
    outflow_marker = [2,3]
    no_slip_marker = [0]

## Solve for NS flow in mesh domain
u,p = navier_stokes(mesh, boundaries, mu, pin, pout, inflow_marker, outflow_marker, no_slip_marker)


file  = File("Plots/u.pvd")
file << u

file  = File("Plots/p.pvd")
file << p
