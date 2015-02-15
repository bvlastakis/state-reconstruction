import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
import os
import pandas
import scipy.optimize
import scipy.linalg

from design import designW, designQ, designCW
from regression import reg_inversion, reg_cvx, reg_minimize, reg_speedy
from utilities import complexStack, complexUnstack, interpDispGrid


BASIS = 4

class CV_Measurement(object):
    """Organizes an imported or simulated data file of a continuous variable measurement
    which contains a list displacements and 'raw' heterodyne integrated voltages or
    average measurement probabilities.
    """

    def __init__(self):

        self.file_dirname = None
        self.file_name = None

        self.total_measurements = None
        self.displacements = None
        self.data_frame = None
        self.data_raw = None
        self.data_to_fit = None
        self.data_norm = None

        self.basis = None
        self.design_matrix = None

    def importData(self, file_path):
        """Imports data from file path.

        Note: Current data format must be tab delimited and the first column
        should contain values for real displacements and the first row should
        contain values for imaginary displacements. (See 'sample_code/X_E.txt').
        """

        #split path into its directory and file name
        file_dirname, file_name = os.path.split(file_path)
        self.file_dirname = file_dirname
        self.file_name = file_name
        #read file as float
        data_frame = pandas.read_csv(file_path, delimiter = '\t', index_col=0)
        data_frame.columns = data_frame.columns.astype(np.float64)

        self.data_frame = data_frame
        self.total_measurements = sum(data_frame.count())
        self.data_raw = data_frame.values.T
        self.data_to_fit = self.data_raw

        #calculate displacement grid
        displace_real = data_frame.index.values.astype('cfloat')
        displace_imag = data_frame.columns.values.astype('cfloat')

        mesh_Real, mesh_Imag = np.meshgrid(displace_real, displace_imag)
        self.displacements = mesh_Real + 1j * mesh_Imag
        self.data_title = 'Measured data'

    def addNoiseData(self, noise_amp):
        """Adds noise to the 'data_raw'. This could be useful for testing the
        sensitivity to noisey measurements.
        """
        noise = np.random.normal(0, noise_amp, self.data_raw.shape)
        self.data_to_fit = self.data_raw + noise

    def calcDesignMatrix(self):
        pass

    def regression(self, method = 'inversion', tolerance = 1e-6):

        if self.design_matrix is None:
            raise NameError("a design matrix must exist to perform regression")

        N = self.basis

        # real-valued design matrix
        M = self.design_matrix_complex

        W_flat = self.data_to_fit.reshape( (self.data_to_fit.size, 1) )
        # real-valued response vector
        W = complexStack(W_flat)
        self.response_complex = W


        if method is 'inversion':
            # perform linear regression by matrix inversion where we solve
            #   the problem: Min(M*x - W) by solving for invM  = (M.T * M)^-1 * M.T
            #   and calculating x^hat = invM * W.
            self.density_matrix = reg_inversion(M, W, N,)
            return self.density_matrix

        elif method is 'convex':
            # perform convex optimization using CVXPY toolbox and restricting
            #   the calculated density matrix to be postive semi-definite
            self.density_matrix = reg_cvx(M, W, N, tolerance)
            return self.density_matrix

        elif method is 'minimize':
            # performs least squares regression with a constraint that the
            # trace of the density matrix must equal one.
            self.density_matrix = reg_minimize(M, W, N, tolerance)
            return self.density_matrix

        elif method is 'speedy':
            # performs linear regression by matrix inversion then
            # removes negative eigenvalues (Smolin 2011).
            self.density_matrix = reg_speedy(M, W, N)
            return self.density_matrix

        else:
            raise TypeError("method must be either 'inversion', 'convex', 'minimize', or 'speedy'")

    def posEigenvalues(self):
        '''Following the steps defined in Smolin 'Maximum Liklihood, minimum effort' (2011) to turn a Hermitian matrix positive semidefinite.
        '''

        if self.density_matrix is None:
            raise NameError("A regression must be run and density matrix defined to correct positive semidefinite-ness.")

        rho = self.density_matrix
        (evalue, evector) = scipy.linalg.eigh(rho)
        return (evalue, evector)



    def plotData(self, data_frame_plot = False):
        """Plots the imported or simulated data found in 'data_raw'."""
        if self.data_frame is not None and data_frame_plot is True:
            self.data_frame.plot()
        else:
            fig_shape = ( np.max(np.real(self.displacements)),
                                np.max(np.imag(self.displacements)) )
            fig_shape = fig_shape/np.max(fig_shape)
            fig = plt.figure(figsize = 3*fig_shape)
            x, y = np.real(self.displacements), np.imag(self.displacements)
            ax = fig.add_subplot(111) #, projection='3d')
            ax.pcolor(x, y, self.data_to_fit)
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            ax.set_title(self.data_title)

    def plotDesign(self, state = qp.fock_dm(BASIS, 0), title = [''], show = True):
        """Plot the CV representation of a density matrix given the calculated
        design matrix. This can be used as a check to make sure that the creadted
        design matrix creates the expected quasi-probability distribution.
        """
        if self.design_matrix is None:
            raise ValueError("a design matrix must be defined")

        #truncate state to basis defined by the design matrix
        if isinstance(state, qp.Qobj):
            state.data = state.data[:self.basis, :self.basis]
            D_flat = state.data.toarray().reshape((self.basis**2, 1))
        else:
            state = state[:self.basis, :self.basis]
            D_flat = state.reshape((self.basis**2,1))

        #unpack complex arrays into their real counterparts
        D_complex = complexStack(D_flat)
        M_complex = self.design_matrix_complex

        R_complex = np.split( np.dot(M_complex,D_complex), 2)
        R = R_complex[0].reshape(self.displacements.shape)

        #the to-be plotted response variables calculated
        self.design_response = R

        if show is True:
            fig_shape = ( np.max(np.real(self.displacements)),
                                np.max(np.imag(self.displacements)) )
            fig_shape = fig_shape/np.max(fig_shape)
            fig = plt.figure(figsize = 10*fig_shape)
            ax = fig.add_subplot(111)
            ax.contourf(np.real(self.displacements), np.imag(self.displacements),
                              self.design_response, 200)
            ax.set_title(title)

        return self.design_response

    def plotWigner(self, state, dispGrid = None, factor = 3, title = [''], show = True):
        """Plot the wigner function of a given state using a grid of displacements
        that is interpolated from the given displacement grid.
        """
        if not isinstance(state, qp.Qobj):
            state = qp.Qobj(state)
        # if no displacement grid given, interpolate current defined displacements
        if dispGrid is None:
            dispGrid = interpDispGrid(self.displacements, factor)

        real_vec = np.real(dispGrid[0,:])
        imag_vec = np.imag(dispGrid[:,0])

        wPlot = np.pi / 2 * qp.wigner(state, real_vec, imag_vec, g = 2)

        if show is True:
            fig_shape = ( np.max(np.real(self.displacements)),
                                np.max(np.imag(self.displacements)) )
            fig_shape = fig_shape/np.max(fig_shape)
            fig = plt.figure(figsize = 10*fig_shape)
            ax = fig.add_subplot(111) #, projection='3d')
            ax.contourf(np.real(dispGrid), np.imag(dispGrid),
                              wPlot, 200)
            ax.set_title(title)

        return wPlot

    def plotDiagonal(self, state):

        state_diagonal = np.real( state.diagonal() )
        n = range(len(state_diagonal))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(n, state_diagonal[n])
        ax.set_title('Reconstructed photon number probability')
        ax.set_xlabel('Photon number')
        ax.set_ylabel('Probability')

    def padZeros(self):

        self.data_to_fit
        pass


class CV_Wigner(CV_Measurement):
    """Imported data file which corresponds to a Wigner measurement either
    through Ramsey interferometry or selective rotations on same-parity photon
    number states.
    """

    def __init__(self):

        super(CV_Wigner, self).__init__()

    def calcNormalizer(self, method = 'trapezoid', factor = 10, basis = None):
        """ Calculates normalizer of wigner function by either of two methods.
        Trapezoidal integration of the raw data, or an integration of an
        unconstrained fit of the measured wigner function.
        """

        disp_diff = self.displacements[0][0] - self.displacements[1][1]
        d_real, d_imag = np.real(disp_diff), np.imag(disp_diff)

        if method is 'trapezoid':
            #assumes equal spacing along x and y axes
            to_int = self.data_raw
            disp_diff = self.displacements[0][0] - self.displacements[1][1]
            d_real, d_imag = np.real(disp_diff), np.imag(disp_diff)
            normalizer = np.trapz(to_int, axis = 0)
            normalizer = np.trapz(normalizer, axis = 0)
            normalizer = np.pi / (2 * normalizer * d_real * d_imag)
            self.data_norm = normalizer * self.data_raw
            self.data_to_fit = self.data_norm

            return normalizer

        if method is 'fit':

            self.calcDesignMatrix(basis = basis)
            recon_mat = self.regression()
            to_int = self.plotWigner(recon_mat, factor = factor)
            normalizer = np.trapz(to_int, axis = 0)
            normalizer = np.trapz(normalizer, axis = 0)
            normalizer = np.pi * factor**2 / (2 * normalizer * d_real * d_imag)
            self.data_norm = normalizer * self.data_raw
            self.data_to_fit = self.data_norm

            return normalizer

    def calcDesignMatrix(self, basis = None, method = 'iterative'):
        """Calculate the design matrix for the wigner function with a given
        set of displacements.
        """
        self.basis = basis
        if self.basis is None:
            raise ValueError("basis must be defined to calculate design matrix")

        self.design_matrix = designW(basis = self.basis, dispGrid = self.displacements,
                                        method = method)
        # convert the design matrix into a real-valued matrix for regression
        M_shape = self.design_matrix.shape
        M_flat = self.design_matrix.reshape(np.product(M_shape[0:2]),
                                          np.product(M_shape[2:4]))
        M = complexStack(M_flat)
        # real-valued design matrix
        self.design_matrix_complex = M

    def simulateData(self, state = qp.fock(5,0), noise_amp = 0.05, dispGrid = None):
        """Creates a simulated set of Wigner measurements, with a given state,
        noise amplitude, and displacement grid.
        """
        if dispGrid is None:
            dispArray = np.linspace(-4, 4, 81)
            mesh_Real, mesh_Imag = np.meshgrid(dispArray, dispArray)
            dispGrid = mesh_Real + 1j * mesh_Imag

        wigner_no_noise = 0.5 * np.pi * qp.wigner( state, np.real(dispGrid[0,:]),
                                                np.imag(dispGrid[:,0]), g = 2 )
        # calculate and add random noise to produced wigner function
        if noise_amp > 0:
            noise = np.random.normal(0, noise_amp, wigner_no_noise.shape)
        else:
            noise = 0
        sim_wigner = wigner_no_noise + noise

        self.displacements = dispGrid
        self.data_to_fit = sim_wigner
        self.data_title = 'Simulated Wigner function'

    def getObs(self, c_op):

        #retrieve displacement steps in Re and Im directions
        diff = self.displacements[0,0] - self.displacements[1,1]
        #retrieve and reshape raw data
        overlap_data = self.data_to_fit.reshape(self.displacements.shape)

        #create obvservable in continuous variable basis
        #WARNING! Cavity identity requires an infinitely large truncation basis
            #do not build this using a truncated identity
        # obs_discrete = qp.tensor(q_op, c_op)

        obs_continuous = self.plotDesign(state = c_op, show = False)
        obs_continuous = obs_continuous.reshape(self.displacements.shape)

        #Simply do a reimann sum
        toSum = np.multiply(overlap_data, obs_continuous)
        result = np.sum(toSum) * np.real(diff) * np.imag(diff) * 4 / np.pi

        return result


class CV_Qfunction(CV_Measurement):
    """Imported data file which corresponds to a contunuous variable measurement
    of a Husimi Q-function.
    """

    def __init__(self, photon_projection):

        super(CV_Qfunction, self).__init__()

        self.photon_projection = photon_projection

    def calcNormalizer(self):
        pass

    def calcDesignMatrix(self, basis = None, method = 'iterative'):
        self.basis = basis
        if self.basis is None:
            raise ValueError("basis must be defined to calculate design matrix")

        self.design_matrix = designQ(basis = self.basis,
                                        dispGrid = self.displacements,
                                        photon_proj = self.photon_projection,
                                        method = method)

        # convert the design matrix into a real-valued matrix for regression
        M_shape = self.design_matrix[:][:][:][:][self.photon_projection].shape
        M_flat = self.design_matrix.reshape(np.product(M_shape[0:2]),
                                          np.product(M_shape[2:4]))
        M = complexStack(M_flat)
        # real-valued design matrix
        self.design_matrix_complex = M

    def simulateData(self, state = qp.fock(5,0), noise_amp = 0.05, dispGrid = None, photon = None):
        """Creates a simulated set of Wigner measurements, with a given state,
        noise amplitude, and displacement grid.
        """
        if photon is None:
            photon = self.photon_projection
        if dispGrid is None:
            dispArray = np.linspace(-4, 4, 81)
            mesh_Real, mesh_Imag = np.meshgrid(dispArray, dispArray)
            dispGrid = mesh_Real + 1j * mesh_Imag

        if state.type == 'ket' or state.type == 'bra':
            rho = qp.ket2dm(state)
        else:
            rho = state

        qMat = designQ(state.shape[0], dispGrid, photon)
        print qMat.shape
        qMat = qMat[:,:,:,:,photon]
        print qMat.shape
        q_func = np.real(np.sum(np.sum(qMat * rho.data.toarray(), 2), 2))
        # calculate and add random noise to produced wigner function
        noise = np.random.normal(0, noise_amp, q_func.shape)
        sim_q_func = q_func + noise

        self.displacements = dispGrid
        self.data_to_fit = sim_q_func
        self.data_title = 'Simulated Q function'

class CV_QCWigner(CV_Measurement):
    """Imported data file which corresponds to a Wigner measurement through
    sequential qubit then cavity state measurement. Each wigner will correspond
    to a particular qubit state projection.
    """

    def __init__(self, qubit_projection):

        super(CV_QCWigner, self).__init__()
        # projection is a state vector
        self.qubit_projection = qubit_projection

    def calcNormalizer(self, normalizer = None):
        if normlizer is not None:
            self.normalizer = normalizer
        else:
            raise ValueError("Must specify a normalizing constant")

    def calcDesignMatrix(self, basis = None, method = 'iterative'):
        """Calculate the design matrix for the conditional wigner function
        for a given qubit projection and set of displacements
        """
        self.basis = basis
        if self.basis is None:
            raise ValueError("basis must be defined to calculate design matrix")

        self.design_matrix = designCW(basis = self.basis, dispGrid = self.displacements, q_proj = self.qubit_projection, method = method)

        M_shape = self.design_matrix.shape
        M_flat = self.design_matrix.reshape(np.product(M_shape[0:2]),
                                          np.product(M_shape[2:4]))
        #M = complexStack(M_flat)
        # real-valued design matrix
        return M_flat

    def simulateData(self, state = qp.tensor(qp.fock_dm(2,0), qp.fock_dm(5,0)), proj = qp.fock(2,0),noise_amp = 0.05, dispGrid = None):
        """Creates a simulated set of Wigner measurements, with a given state,
        noise amplitude, and displacement grid.
        """

        if dispGrid is None:
            dispArray = np.linspace(-4, 4, 81)
            mesh_Real, mesh_Imag = np.meshgrid(dispArray, dispArray)
            dispGrid = mesh_Real + 1j * mesh_Imag

        basis = state.dims[0][1]
        projector = qp.tensor(proj, qp.identity(basis))
        proj_state = projector.dag() * state * projector
        proj_prob = proj_state.norm()
        wigner_no_noise = 0.5 * np.pi * qp.wigner( proj_state.unit(),
                                                np.real(dispGrid[0,:]),
                                                np.imag(dispGrid[:,0]), g = 2 )
        # calculate and add random noise to produced wigner function
        noise = np.random.normal(0, noise_amp, wigner_no_noise.shape)
        sim_wigner = wigner_no_noise + noise

        self.displacements = dispGrid
        self.data_to_fit = sim_wigner
        self.data_title = 'Simulated Wigner function by ' + 'qubit projection'












