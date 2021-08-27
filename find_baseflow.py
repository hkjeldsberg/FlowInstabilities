from os import path

from fenics import *
from mshr import Cylinder, generate_mesh


def solve_navier_stokes(mesh, boundaries, nu, pin, pout, inflow_marker, outflow_marker, no_slip_marker):
    """
    Setup and solve the incompressible Navier-Stokes equations given a pressure drop throughout
    the domain.

    Args:
        mesh (Mesh): Mesh of problem domain
        boundaries (MeshFunction): Function determining the boundaries of the mesh
        nu (float): Kinematic viscosity
        pin (float): Predefined pressure value at inlet(s)
        pout (float): Predefined pressure value at outlet(s)
        inflow_marker (list): ID(s) corresponding to inlet(s) of mesh
        outflow_marker (list): ID(s) corresponding to outlet(s) of mesh
        no_slip_marker (list): ID(s) corresponding to wall(s) of mesh

    Returns:
        u (Function): Velocity field solution
        p (Function): Pressure field solution
    """
    # Set up discrete function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH)

    up = Function(W)
    (u, p) = split(up)
    (v, q) = TestFunctions(W)

    # Set boundary conditions
    bcu_no_slip = DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), boundaries, no_slip_marker[0])

    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    n = FacetNormal(mesh)
    f = Constant((0, 0, 0))

    a = Constant(nu) * inner(grad(u), grad(v)) * dx
    a += dot(dot(u, nabla_grad(u)), v) * dx
    a += - div(v) * p * dx + div(u) * q * dx

    L = dot(f, v) * dx

    for im in inflow_marker:
        L += - Constant(pin) * dot(n, v) * ds(im)

    for om in outflow_marker:
        L += - Constant(pout) * dot(n, v) * ds(om)

    # Solve variational problem
    F = a - L
    dF = derivative(F, up)

    PETScOptions.set("mat_mumps_icntl_4", 1)  # Level of printing from the solver (0-4)
    PETScOptions.set("mat_mumps_icntl_14", 300)  # Percentage increase in the working space wrt memory
    problem = NonlinearVariationalProblem(F, up, bcu_no_slip, dF)
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


def make_pipe_mesh(radius, n_elem):
    """
    Setup mesh for the cylinder problem and define
    domain boundaries used to set boundary conditions.

    Args:
        radius (float): Radius of cylinder
        n_elem (int): Mesh resolution

    Returns:
        mesh (Mesh): Mesh of cylinder domain
        boundaries (MeshFunction): Function determining the boundaries of the mesh
    """
    # Define a cylinder domain
    cylinder = Cylinder(Point(0, 0, 0), Point(1, 0, 0), radius, radius)
    geometry = cylinder

    # Define the mesh
    mesh = generate_mesh(geometry, n_elem)
    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    # Mark the inflow, outflow and the walls
    class Noslip(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[1] ** 2 + x[2] ** 2 > 0.95 * radius ** 2

    class Inflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0)

    class Outflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1)

    velocity = Noslip()
    velocity.mark(boundaries, 1)

    pressure_inflow = Inflow()
    pressure_inflow.mark(boundaries, 2)

    pressure_outflow = Outflow()
    pressure_outflow.mark(boundaries, 3)

    return mesh, boundaries


def get_marker_ids(case):
    """
    Determine the IDs which define the inlet(s), outlet(s) and wall(s)
    of the given case.

    Args:
        case (int): Number corresponding to case ID

    Returns:
        inflow_marker (list): ID(s) corresponding to inlet(s) of mesh
        outflow_marker (list): ID(s) corresponding to outlet(s) of mesh
        no_slip_marker (list): ID(s) corresponding to wall(s) of mesh
    """
    if case in [0, 1]:
        inflow_marker = [1]
        outflow_marker = [2, 3]
    elif case == 2:
        inflow_marker = [2]
        outflow_marker = [1, 3]
    elif case == 3:
        inflow_marker = [2]
        outflow_marker = [1]
    else:
        inflow_marker = [1]
        outflow_marker = [2]
    no_slip_marker = [0]

    return inflow_marker, outflow_marker, no_slip_marker


def D(u):
    grad_u = grad(u)
    return grad_u + grad_u.T


if __name__ == '__main__':
    pin = 2.0
    pout = 1.0
    nu = 1.0
    radius = 1.0
    n_elem = 15

    poise_case = 0
    artery_case = 1

    if poise_case:
        # Make pipe mesh so we can solve for Poiseuille flow
        mesh, boundaries = make_pipe_mesh(radius, n_elem)
        inflow_marker = [2]
        outflow_marker = [3]
        no_slip_marker = [1]

    if artery_case:
        # Get artery mesh
        case_names = ["C0015_healthy", "C0015_terminal", "C0019", "C0065_healthy", "C0065_saccular"]
        case = 3
        mesh_name = path.join("models", case_names[case] + ".xml.gz")
        mesh = Mesh(mesh_name)
        boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
        inflow_marker, outflow_marker, no_slip_marker = get_marker_ids(case)

    # Solve for NS flow in mesh domain
    u, p = solve_navier_stokes(mesh, boundaries, nu, pin, pout, inflow_marker, outflow_marker, no_slip_marker)

    file = File("Baseflow/u.pvd")
    file << u

    file = File("Baseflow/p.pvd")
    file << p
