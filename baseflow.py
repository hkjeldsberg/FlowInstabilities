from fenics import *
from mshr import * 

def baseflow(radius, nelem,nu, pin,pout):

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
             return on_boundary and x[1]**2 + x[2]**2 > 0.99 and x[0] > 0 and x[1] < 1    
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

    # discrete function space 
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) 
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    TH = MixedElement([P2, P1]) 
    W  = FunctionSpace(mesh, TH) 

    up = Function(W) 
    (u,p) = split(up) 
    (v,q) = TestFunctions(W) 

    # boundary conditions 
    bcu_noslip = DirichletBC(W.sub(0), Constant((0.0,0.0,0.0)), boundaries, 1) 
    bcu_inflow = DirichletBC(W.sub(1),Constant(pin), boundaries, 2) 
    bcu_outflow= DirichletBC(W.sub(1),Constant(pout), boundaries, 3)

    dx = Measure('dx', domain = mesh) 
    ds = Measure('ds',domain = mesh,  subdomain_data = boundaries) 

    n    = FacetNormal(mesh) 
    nu   = Constant(nu) 
    f    = Constant((0,0,0)) 

    a   =Constant(0.5*nu)*inner(D(u), D(v))*dx 
    a += dot(dot(u, nabla_grad(u)),v)*dx - div(v)*p*dx +  q*div(u)*dx
    a += -Constant(nu)*dot(nabla_grad(u)*n, v)*ds(2) -Constant(nu)*dot(nabla_grad(u)*n, v)*ds(3)
    L = - Constant(pin)*dot(n,v)*ds(2) -Constant(pout)* dot(n,v)*ds(3) 

    L += dot(f, v)*dx 

    # solve 
    F = a - L 
    dF = derivative(F,up)

    PETScOptions.set("mat_mumps_icntl_14",300)


    problem = NonlinearVariationalProblem(F, up,[bcu_noslip,bcu_inflow, bcu_outflow], dF )
    solver = NonlinearVariationalSolver(problem) 
    solver.parameters['newton_solver']['linear_solver'] = 'mumps' 
    solver.parameters['newton_solver']['maximum_iterations'] = 10 

    solver.solve()
    u,p = up.split(deepcopy = True)
    
    return boundaries, u, p

def D(u):
    grad_u = grad(u)
    return grad_u + grad_u.T

pin  = 8.0
pout = 0.0
mu   = 1.0   
radius  = 1.0
nelem = 15
boundaries, u,p = baseflow(radius, nelem, mu, pin, pout) 

file  = File("u.pvd") 
file << u 

file  = File("p.pvd") 
file << p 
