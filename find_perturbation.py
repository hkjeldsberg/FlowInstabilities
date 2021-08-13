import numpy as np
from dolfin import *
from mshr import *
from baseflow import navier_stokes, make_pipe_mesh, get_marker_ids
from os import path


def main(case, delta_p):
    print('Running case: ' + case)
    
    if case.find('poise')>-1:
        print('Running poise case')
        # Define parameters
        p_in = 2.0
        p_out = 1.0
        nu = 1.0
        rho = 1.0
        radius = 1
        length = 5
        N = 10  # Resolution for mesh generation


        # Make pipe mesh so we can solve for Poiseuille flow
        mesh, boundaries = make_pipe_mesh(radius, N)
        inflow_marker = [2]
        outflow_marker = [3]
        no_slip_marker = [1]

        # Compute max velocity, Reynolds number and check ratio between length and radius of pipe
        U_max = (p_in - p_out) * radius ** 2 / (length * nu * rho * 4)
        Re = U_max * radius / nu
        ratio = length / radius
        if ratio <= 1 / 48 * Re:
            print(ratio, 1 / 48 * Re, Re)
            print("Ratio = length / radius must be larger than 1/48*Re for the Hagen–Poiseuille law to be valid.")
            exit()

        results_folder = 'aneurysm/Eigenmodes/poiseuille/'

        print("Reynolds number: {:.3f}".format(Re))

    else:

        # Get artery mesh
        case_names = ["C0015_healthy", "C0015_terminal", "C0019", "C0065_healthy", "C0065_saccular"]
        case  = int(case)

        print('Running artery case ' + case_names[case])
        mesh_name = path.join("models", case_names[case] + ".xml.gz")
        mesh = Mesh('aneurysm/' + mesh_name)
        boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1, mesh.domains())
        inflow_marker, outflow_marker, no_slip_marker = get_marker_ids(case)

        # Rescale mesh from mm to m 
        coords = mesh.coordinates()
        coords *= 1.0e-3

        nu = 3.0e-6 # kinematic visc of blood is 3 cP
        mmHg = 133.0
        p_in = delta_p*mmHg
        p_out = 0.0

        results_folder = 'aneurysm/Eigenmodes/' + case_names[case] + '/' + str(int(delta_p)) + 'mmHg/'

    print('Results will be stored in ' + results_folder)


    # Check if baseflow is already computed
    import os as os
    str_match = max([ffile.find('u0') for ffile in os.listdir(results_folder)])
    baseflow_file_exists = (str_match > -1)    
    
    if baseflow_file_exists:
        print('Using previously computed baseflow')
        ## Set up mesh
        mesh = Mesh()   
        h5 = HDF5File(mesh.mpi_comm(), results_folder + 'mesh.h5', 'r')
        h5.read(mesh, '/mesh', False)

        P2 = VectorFunctionSpace(mesh, 'CG', 2, 3)
        P1 = FunctionSpace(mesh, 'CG', 1)

        uf = HDF5File(mesh.mpi_comm(), results_folder + 'u0.h5', "r")
        u0 = Function(P2)
        uf.read(u0, "/u")
        uf.close()

    else: 
        print('Computing baseflow')
        u0, p= navier_stokes(mesh, boundaries, nu, p_in, p_out, inflow_marker, outflow_marker, no_slip_marker)

        file = HDF5File(mesh.mpi_comm(), results_folder + 'mesh.h5', "w")
        file.write(p.function_space().mesh(), "/mesh")
        file.close()

        file = HDF5File(mesh.mpi_comm(), results_folder + 'u0.h5', "w")
        file.write(u0, "/u")
        file.close()

        file = HDF5File(mesh.mpi_comm(), results_folder + 'p0.h5', "w")
        file.write(p, "/p")
        file.close()

    
    
    File(results_folder + 'baseflow.pvd') << interpolate(u0, VectorFunctionSpace(mesh, 'CG', 1, 3))

    ## Setup eigenvalue matrices and solver
    print('Setting up eigenvalue problem, storing results in ' + results_folder)
    A, B, W = get_eigenvalue_matrices(mesh, nu, u_init=u0)

    E = setup_slepc_solver(A, B, target_eigvalue=-1.0e-5, max_it=5, n_eigvals=6)

    # Solve for eigenvalues
    print('Solving eigenvalue problem')
    import time
    tic = time.perf_counter()
    E.solve()
    toc = time.perf_counter()
    print('Done solving, process took %4.0f s' % float(toc - tic))

    # Write the results to terminal and a .text file
    eigvalue_results = results_folder + 'Eigenmodes'

    nev, ncv, mpd = E.getDimensions()  # number of requested eigenvalues and Krylov vectors
    nconv = E.getConverged()

    print('****** nu:', nu, '******')

    lines = []
    lines.append('Parameters: nu %1.4f' % (nu))
    lines.append("Number of iterations of the method: %d" % E.getIterationNumber())
    lines.append("Number of requested eigenvalues: %d" % nev)
    lines.append("Stopping condition: tol=%.4g, maxit=%d" % E.getTolerances())
    lines.append("Number of converged eigenpairs: %d" % nconv)
    lines.append("Solution method: %s" % E.getType())
    lines.append('Solver time: %0.1fs' % float(toc - tic))

    lines.append(' ')

    lines.append("      lambda   (residual)    lambda_num  lamda_den")
    lines.append("---------------------------------------------------\n")

    with open(eigvalue_results, "w") as ffile:
        ffile.write('\n'.join(lines))
        print('\n'.join(lines))

    file_u, file_p, file_e = (File(results_folder + name + '.pvd')
                              for name in ('eigvecs', 'eigpressures', 'eigvals'))

    if nconv == 0:
        with open(eigvalue_results, "w+") as ffile:
            ffile.write('No converged values :( ')
    if nconv > 0:
        # Create the results vectors
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()

        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            lamda = 1.0 / k.real

            u_r, u_im = Function(W), Function(W)
            E.getEigenpair(i, u_r.vector().vec(), u_im.vector().vec())
            u, p, c = u_r.split()

            # Store eigenmode
            fFile = HDF5File(MPI.comm_world, results_folder + 'eigvecs' + '_n' + str(i) + ".h5", "w")
            fFile.write(u, "/f")
            fFile.close()

            fFile = HDF5File(MPI.comm_world, results_folder + 'eigenmode' + '_n' + str(i) + ".h5", "w")
            fFile.write(u_r, "/f")
            fFile.close()

            # Plot eigenvector and corresponding eigenvalue
            u.rename('eigvec', '0.0')
            file_u << (u, float(i))

            p.rename('eigpressure', '0.0')
            file_p << (p, float(i))

            lamda_i = interpolate(Constant(lamda), W.sub(2).collapse())
            lamda_i.rename('eigval', '0.0')
            file_e << (lamda_i, float(i))

            if k.imag != 0.0:
                line = " %9f%+9f j" % (1.0 / k.real, 1.0 / k.imag)
            else:
                line = " %12f" % (1.0 / k.real)

            print(line)
            with open(eigvalue_results, "a") as ffile:
                ffile.write(line + '\n')


def create_mesh(radius, length, N):
    cylinder = Cylinder(Point(0, 0, 0), Point(length, 0, 0), radius, radius)
    mesh = generate_mesh(cylinder, N)
    return mesh


def setup_slepc_solver(A, B, target_eigvalue, max_it, n_eigvals):
    ## Solve eigenvalue problem using slepc
    from slepc4py import SLEPc
    E = SLEPc.EPS()
    E.create()

    # A and B are both symmetric, but B is singular
    # For this reason, the problem is a Generalized Indefinite Eigenvalue Problem (GIEP)
    E.setOperators(A, B)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # Specify solvers and target values
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(target_eigvalue)

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

    n_krvecs = int(3.0 * n_eigvals)

    E.setDimensions(n_eigvals, n_krvecs)
    return E


def D(u):
    gradu = grad(u)
    return gradu + gradu.T


def get_eigenvalue_matrices(mesh, nu, u_init):
    dim = np.shape(mesh.coordinates())[1]
    if dim == 2: zero = Constant((0.0, 0.0))
    if dim == 3: zero = Constant((0.0, 0.0, 0.0))

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

    a = EIG_A_cyl(u, p, c, v, q, d, nu)

    V = W.sub(0).collapse()
    u_zero_i = interpolate(u_init, V)

    b = EIG_B_cyl(u, p, c, v, q, d, u_zero_i)

    dummy = v[0] * dx
    A = PETScMatrix()
    assemble_system(a, dummy, bc, A_tensor=A)
    B = PETScMatrix()
    assemble_system(b, dummy, [], A_tensor=B)

    A, B = A.mat(), B.mat()

    return A, B, W


## Set up operators for eigenvalue problem ##


def EIG_A_cyl(u, p, c, v, q, d, mu):
    eiga = (0.5 * mu * inner(D(u), D(v)) * dx - div(v) * p * dx - q * div(u) * dx
            + c * q * dx + d * p * dx
            )
    return eiga


def EIG_B_cyl(u, p, c, v, q, d, u0):
    return 0.5 * inner(dot(D(u0), v), u) * dx


                    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Problem parameters, nu is allowed to be a list
    parser.add_argument('--case', help='which case to run. \n Options: poise, 0 (C0015_healthy), 1 (C0015_terminal), 2 (C0019), 3 (C0065_healthy), 4 (C0065_healthy)', default='poise', type=str)
    parser.add_argument('--delta_p', help='pressure drop in mmHg', default=5, type=float)

    args = parser.parse_args()
    
    main(args.case, args.delta_p)