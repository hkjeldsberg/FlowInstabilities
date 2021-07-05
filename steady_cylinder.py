from dolfin import *
from mshr import *


def create_mesh(radius, length, N):
    cylinder = Cylinder(Point(0, 0, 0), Point(length, 0, 0), radius, radius)

    mesh = generate_mesh(cylinder, N)
    print_mesh_info(mesh)
    return mesh


def print_mesh_info(mesh):
    xmin = mesh.coordinates()[:, 0].min()
    xmax = mesh.coordinates()[:, 0].max()
    ymin = mesh.coordinates()[:, 1].min()
    ymax = mesh.coordinates()[:, 1].max()
    zmin = mesh.coordinates()[:, 2].min()
    zmax = mesh.coordinates()[:, 2].max()
    print("Mesh dimensions:")
    print("xmin, xmax: {}, {}".format(xmin, xmax))
    print("ymin, ymax: {}, {}".format(ymin, ymax))
    print("zmin, zmax: {}, {}".format(zmin, zmax))
    print("Number of cells: {}".format(mesh.num_cells()))


def set_boundaries(mesh, length, radius):
    class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] ** 2 + x[2] ** 2 > 0.95 * radius ** 2

    class Inlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0)

    class Outlet(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], length)

    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # Mark boundaries: wall, inlet and outlet
    wall = Wall()
    inlet = Inlet()
    outlet = Outlet()
    wall.mark(boundaries, 1)
    inlet.mark(boundaries, 2)
    outlet.mark(boundaries, 3)

    return boundaries


def solve_for_baseflow(radius, length, N, p_in, p_out, nu, rho):
    # Create mesh
    mesh = create_mesh(radius, length, N)

    # Set boundaries
    boundaries = set_boundaries(mesh, length, radius)

    # Create mixed element space, P2 for velocity, P1 for pressure
    U = VectorElement("CG", mesh.ufl_cell(), 2)
    P = FiniteElement("CG", mesh.ufl_cell(), 1)
    ME = MixedElement([U, P])
    W = FunctionSpace(mesh, ME)

    # Define test and trial functions
    up = Function(W)
    (u, p) = split(up)
    (v, q) = TestFunctions(W)

    # Set boundary conditions
    bcu_wall = DirichletBC(W.sub(0), Constant((0, 0, 0)), boundaries, 1)
    bcp_in = DirichletBC(W.sub(1), Constant(p_in), boundaries, 2)
    bcp_out = DirichletBC(W.sub(1), Constant(p_out), boundaries, 3)
    bcs = [bcu_wall, bcp_in, bcp_out]

    # Variational form of steady NS-equations
    f = Constant((0, 0, 0))
    nu = Constant(nu)
    rho = Constant(rho)
    n = FacetNormal(mesh)

    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    a = nu * inner(grad(u), grad(v)) * dx
    a += -nu * dot(dot(grad(u), n), v) * ds(2) - nu * dot(dot(grad(u), n), v) * ds(3)
    a += dot(dot(u, nabla_grad(u)), v) * dx
    a += -1 / rho * div(v) * p * dx + div(u) * q * dx
    L = inner(f, v) * dx
    L += -p_in * dot(n, v) * ds(2) - p_out * dot(n, v) * ds(3)

    # Solve equations
    F = a - L
    J = derivative(F, up)

    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.solve()

    u, p = up.split(deepcopy=True)

    return u, p, boundaries


if __name__ == '__main__':
    # Define parameters
    p_in = 8.0
    p_out = 0.0
    nu = 1.0
    rho = 1.0
    radius = 1.0
    N = 15
    length = 1
    u, p, boundaries = solve_for_baseflow(radius, length, N, p_in, p_out, nu, rho)

    # Save solutions
    File("velocity.pvd") << u
    File("pressure.pvd") << p
    File("boundaries.pvd") << boundaries
