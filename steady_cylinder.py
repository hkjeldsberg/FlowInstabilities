import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from mshr import *


def main():
    # Define parameters
    p_in = 100
    p_out = 0
    nu = 1.0
    rho = 1.0
    radius = 1
    length = 5

    # Solution and simulation arrays
    errors = []
    h_values = []
    solutions = []
    DOFs = []
    N_values = [20, 30, 50]  # Resolution for mesh generation

    # Compute max velocity, Reynolds number and check ratio between length and radius of pipe
    U_max = (p_in - p_out) * radius ** 2 / (length * nu * rho * 4)
    Re = U_max * radius / nu
    ratio = length / radius
    if ratio <= 1 / 48 * Re:
        print(ratio, 1 / 48 * Re, Re)
        print("Ratio = length / radius must be larger than 1/48*Re for the Hagen–Poiseuille law to be valid.")
        exit()

    print("Reynolds number: {:.3f}".format(Re))

    # Solve for multiple grid resolutions
    for N in N_values:
        u, p, boundaries, error, h, u_e, DOF = solve_for_baseflow(radius, length, N, p_in, p_out, nu, rho)
        errors.append(error)
        h_values.append(h)
        DOFs.append(DOF)
        solutions.append(u.copy(deepcopy=True))

    # Plot L2 - error and velocity profiles
    plot_l2_error(h_values, errors)
    plot_velocity(solutions, u_e, N_values, DOFs, Re)

    # Save latest solution
    File("velocity.pvd") << u
    File("pressure.pvd") << p
    File("boundaries.pvd") << boundaries


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
    print("xmin, xmax: {:.2f}, {:.2f}".format(xmin, xmax))
    print("ymin, ymax: {:.2f}, {:.2f}".format(ymin, ymax))
    print("zmin, zmax: {:.2f}, {:.2f}".format(zmin, zmax))
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
    h = mesh.hmin()

    # Set boundaries
    boundaries = set_boundaries(mesh, length, radius)

    # Create mixed element space, P2 for velocity, P1 for pressure
    U = VectorElement("CG", mesh.ufl_cell(), 2)
    P = FiniteElement("CG", mesh.ufl_cell(), 1)
    ME = MixedElement([U, P])
    W = FunctionSpace(mesh, ME)
    DOF = W.sub(0).dim()

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
    mu = nu * rho
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

    # Compute L2 error for x component
    error_l2, u_e = compute_l2_error(length, mesh, mu, p_in, p_out, radius, u)

    return u, p, boundaries, error_l2, h, u_e, DOF


def compute_l2_error(length, mesh, mu, p_in, p_out, radius, u):
    # Compare numerical solution with exact solution (x component)
    delta_p = p_in - p_out
    u_e = Expression("delta_p / (L * mu * 4) * (R * R - x[1] * x[1] - x[2] * x[2] )",
                     delta_p=delta_p, mu=mu, L=length, R=radius, degree=2)
    V = FunctionSpace(mesh, 'CG', 2)

    u_exact = interpolate(u_e, V)
    u_computed = interpolate(u.sub(0), V)
    error_L2 = errornorm(u_exact, u_computed, 'L2', degree_rise=1)

    return error_L2, u_e


def plot_l2_error(h_values, errors):
    plt.plot(h_values, errors, 'ro-', linewidth=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.title("Poiseuille flow ($L_2$ error)")
    plt.xlabel("Characteristic edge size - $\Delta x$")
    plt.ylabel("Error - $L_2$")
    plt.show()


def plot_velocity(solutions, u_e, N_values, DOFs, Re):
    # Plot velocity x component of numerical solution
    z_values = np.linspace(0, 0.99, 25)
    for i, u in enumerate(solutions):
        u_values = [u.sub(0)(0, z, 0) * 1000 for z in z_values]
        plt.plot(z_values, u_values, label="DOFs: {}, N = {}".format(DOFs[i], N_values[i]), marker="o")

    # Plot velocity x component of analytical solution
    u_exact_values = [u_e(0, z, 0) * 1000 for z in z_values]
    plt.plot(z_values, u_exact_values, "--", color="black", label="Analytical solution")
    plt.title("Velocity profile, Re={}".format(Re))
    plt.xlabel("Radius $r$ [m]")
    plt.ylabel("Velocity $v_x$ [mm/s]")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
