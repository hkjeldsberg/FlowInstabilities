import argparse
import time
from os import path, listdir

import numpy as np
from dolfin import *
from slepc4py import SLEPc

from find_baseflow import solve_navier_stokes, make_pipe_mesh, get_marker_ids, D


def main(case, delta_p, nu):
    """
    Find the baseflow for the given problem, and define and solve
    the eigenvalue problem for the flow pertubation

    Args:
        case (str): Case number (or string) for the problem
        delta_p (float): Pressure drop in mmHg
        nu (float): Kinematic viscosity
    """
    print('Running case: ' + case)

    if case.find('poise') > -1:
        print('Running poise case')
        # Define parameters
        p_in = 2.0
        p_out = 1.0
        radius = 1
        length = 5
        n_elem = 10  # Resolution for mesh generation

        # Make pipe mesh so we can solve for Poiseuille flow
        mesh, boundaries = make_pipe_mesh(radius, n_elem)
        inflow_marker = [2]
        outflow_marker = [3]
        no_slip_marker = [1]

        # Compute max velocity, Reynolds number and check ratio between length and radius of pipe
        U_max = (p_in - p_out) * radius ** 2 / (length * nu * 4)
        Re = U_max * radius / nu
        ratio = length / radius
        if ratio <= 1 / 48 * Re:
            print(ratio, 1 / 48 * Re, Re)
            print("Ratio = length / radius must be larger than 1/48*Re for the Hagen–Poiseuille law to be valid.")
            exit()

        results_folder = 'Eigenmodes/poiseuille/'

        print("Reynolds number: {:.3f}".format(Re))

    else:

        # Get artery mesh
        case_names = ["C0015_healthy", "C0015_terminal", "C0019", "C0065_healthy", "C0065_saccular"]
        case = int(case)

        print('Running artery case ' + case_names[case])
        mesh_name = path.join("models", case_names[case] + ".xml.gz")
        mesh = Mesh(mesh_name)
        boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
        inflow_marker, outflow_marker, no_slip_marker = get_marker_ids(case)

        # Rescale mesh from mm to m 
        coords = mesh.coordinates()
        coords *= 1.0e-3

        mmHg = 133.0
        p_in = delta_p * mmHg
        p_out = 0.0

        results_folder = path.join('Eigenmodes', case_names[case], str(int(delta_p)) + 'mmHg')

    print('Results will be stored in ' + results_folder)

    # Check if baseflow is already computed
    baseflow_file_exists = False
    if path.exists(results_folder):
        str_match = max([ffile.find('u0') for ffile in listdir(results_folder)])
        baseflow_file_exists = (str_match > -1)

    if baseflow_file_exists:
        print('Using previously computed baseflow')
        # Set up mesh
        mesh = Mesh()
        h5 = HDF5File(mesh.mpi_comm(), path.join(results_folder, 'mesh.h5'), 'r')
        h5.read(mesh, '/mesh', False)

        P2 = VectorFunctionSpace(mesh, 'CG', 2, 3)

        uf = HDF5File(mesh.mpi_comm(), path.join(results_folder, 'u0.h5'), "r")
        u0 = Function(P2)
        uf.read(u0, "/u")
        uf.close()

    else:
        print('Computing baseflow')
        u0, p = solve_navier_stokes(mesh, boundaries, nu, p_in, p_out, inflow_marker, outflow_marker, no_slip_marker)

        file = HDF5File(mesh.mpi_comm(), path.join(results_folder, 'mesh.h5'), "w")
        file.write(p.function_space().mesh(), "/mesh")
        file.close()

        file = HDF5File(mesh.mpi_comm(), path.join(results_folder, 'u0.h5'), "w")
        file.write(u0, "/u")
        file.close()

        file = HDF5File(mesh.mpi_comm(), path.join(results_folder, 'p0.h5'), "w")
        file.write(p, "/p")
        file.close()

    File(path.join(results_folder, 'baseflow.pvd')) << interpolate(u0, VectorFunctionSpace(mesh, 'CG', 1, 3))

    # Setup eigenvalue matrices and solver
    print('Setting up eigenvalue problem, storing results in ' + results_folder)
    A, B, W = get_eigenvalue_matrices(mesh, nu, u_init=u0)

    E = setup_slepc_solver(A, B, target_eigenvalue=-1.0e-5, max_it=5, n_eigenvalues=6)

    # Solve for eigenvalues
    print('Solving eigenvalue problem')
    tic = time.perf_counter()
    E.solve()
    toc = time.perf_counter()
    print('Done solving, process took %4.0f s' % float(toc - tic))

    # Write the results to terminal and a .text file
    eigenvalue_results = path.join(results_folder, 'Eigenmodes')

    nev, ncv, mpd = E.getDimensions()  # number of requested eigenvalues and Krylov vectors
    n_converged = E.getConverged()

    print('****** nu:', nu, '******')

    lines = []
    lines.append('Parameters: nu %1.4f' % nu)
    lines.append("Number of iterations of the method: %d" % E.getIterationNumber())
    lines.append("Number of requested eigenvalues: %d" % nev)
    lines.append("Stopping condition: tol=%.4g, maxit=%d" % E.getTolerances())
    lines.append("Number of converged eigenpairs: %d" % n_converged)
    lines.append("Solution method: %s" % E.getType())
    lines.append('Solver time: %0.1fs' % float(toc - tic))

    lines.append(' ')

    lines.append("      lambda   (residual)    lambda_num  lamda_den")
    lines.append("---------------------------------------------------\n")

    with open(eigenvalue_results, "w") as f:
        f.write('\n'.join(lines))
        print('\n'.join(lines))

    file_u, file_p, file_e = (File(path.join(results_folder, name + '.pvd')) for name in
                              ('eigvecs', 'eigpressures', 'eigvals'))

    if n_converged == 0:
        with open(eigenvalue_results, "w+") as f:
            f.write('No converged values :( ')

    if n_converged > 0:
        # Create the results vectors
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()

        for i in range(n_converged):
            k = E.getEigenpair(i, vr, vi)
            lambda_ = 1.0 / k.real

            u_r, u_im = Function(W), Function(W)
            E.getEigenpair(i, u_r.vector().vec(), u_im.vector().vec())
            u, p, c = u_r.split()

            # Store eigenmode
            fFile = HDF5File(MPI.comm_world, path.join(results_folder, 'eigvecs' + '_n' + str(i) + ".h5"), "w")
            fFile.write(u, "/f")
            fFile.close()

            fFile = HDF5File(MPI.comm_world, path.join(results_folder, 'eigenmode' + '_n' + str(i) + ".h5"), "w")
            fFile.write(u_r, "/f")
            fFile.close()

            # Plot eigenvector and corresponding eigenvalue
            u.rename('eigvec', '0.0')
            file_u << (u, float(i))

            p.rename('eigpressure', '0.0')
            file_p << (p, float(i))

            lambda_i = interpolate(Constant(lambda_), W.sub(2).collapse())
            lambda_i.rename('eigval', '0.0')
            file_e << (lambda_i, float(i))

            if k.imag != 0.0:
                line = " %9f%+9f j" % (1.0 / k.real, 1.0 / k.imag)
            else:
                line = " %12f" % (1.0 / k.real)

            print(line)
            with open(eigenvalue_results, "a") as f:
                f.write(line + '\n')


def setup_slepc_solver(A, B, target_eigenvalue, max_it, n_eigenvalues):
    """
    Setup and solve the eigenvalue problem using SLEPc, given
    the right and left hand side of the equation.

    Args:
        A (PETScMatrix): Assembled matrix system (LHS)
        B (PETScMatrix): Assembled matrix system (RHS)
        target_eigenvalue (float): Target eigenvalue
        max_it (int): Maximum iterations for SLEPc solver
        n_eigenvalues (int): Number of eigenvalues to compute

    Returns:
        E (EPS): Eigenvalue problem solver, containing eigenproblem solution
    """
    E = SLEPc.EPS()
    E.create()

    # A and B are both symmetric, but B is singular
    # For this reason, the problem is a Generalized Indefinite Eigenvalue Problem (GIEP)
    E.setOperators(A, B)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # Specify solvers and target values
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(target_eigenvalue)

    E.setFromOptions()

    st = E.getST()
    st.setType('sinvert')

    # Since B is singular, we need to change the linear solver to avoid zero pivots
    ksp = st.getKSP()
    ksp.setType('gmres')
    pc = ksp.getPC()
    pc.setType('hypre')
    pc.setFactorSolverType('mumps')

    E.setTolerances(tol=1.0e-8, max_it=max_it)

    n_krylov_space = int(3.0 * n_eigenvalues)

    E.setDimensions(n_eigenvalues, n_krylov_space)
    return E


def get_eigenvalue_matrices(mesh, nu, u_init):
    """
    Define eigenvalue problem matrices A and B used to
    find the flow perturbation and corresponding eigenvalues

    Args:
        mesh (Mesh): Mesh of problem domain
        nu (float): Kinematic viscosity
        u_init (Function): Baseflow solution

    Returns:
        A (PETScMatrix): Left hand side of the eigenvalue problem
        B (PETScMatrix): Right hand side of the eigenvalue problem
        W (FunctionSpace): FEniCS function space for the given mesh
    """
    dim = np.shape(mesh.coordinates())[1]
    if dim == 2:
        zero = Constant((0.0, 0.0))
    if dim == 3:
        zero = Constant((0.0, 0.0, 0.0))

    # Make a mixed space
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    LM = FiniteElement('R', mesh.ufl_cell(), 0)

    TH = MixedElement([P2, P1, LM])
    W = FunctionSpace(mesh, TH)

    # Set up boundary conditions for space
    def all_boundaries(x, on_boundary):
        return on_boundary

    bc = DirichletBC(W.sub(0), zero, all_boundaries)

    # Define variational problem
    (u, p, c) = TrialFunctions(W)
    (v, q, d) = TestFunctions(W)

    a = EIG_A_cyl(u, p, v, q, nu)

    V = W.sub(0).collapse()
    u_zero_i = interpolate(u_init, V)

    b = EIG_B_cyl(u, v, u_zero_i)

    dummy = v[0] * dx
    A = PETScMatrix()
    assemble_system(a, dummy, bc, A_tensor=A)
    B = PETScMatrix()
    assemble_system(b, dummy, [], A_tensor=B)

    A, B = A.mat(), B.mat()

    return A, B, W


def EIG_A_cyl(u, p, v, q, nu):
    """
    Set up operators for eigenvalue problem (Left hand side)
    """
    eig_a = nu * inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
    return eig_a


def EIG_B_cyl(u, v, u0):
    """
    Set up operators for eigenvalue problem (Right hand side)
    """
    eig_b = 0.5 * inner(dot(D(u0), v), u) * dx
    return eig_b


def read_command_line():
    """
    Read arguments from commandline and return all values in a parser dictionary.
    If no arguments are given, default values will be returned.
    """
    parser = argparse.ArgumentParser()
    # Problem parameters
    parser.add_argument('--case', help='Which case to run. Options: poise, 0 (C0015_healthy), ' +
                                       '1 (C0015_terminal), 2 (C0019), 3 (C0065_healthy), 4 (C0065_healthy)',
                        default='0', type=str, choices={"0", "1", "2", "3", "4", "poise"})
    parser.add_argument('--delta_p', help='Pressure drop in mmHg', default=0.001, type=float)
    parser.add_argument('--nu', help='Kinematic viscosity', default=3e-6, type=float)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = read_command_line()

    main(args.case, args.delta_p, args.nu)
