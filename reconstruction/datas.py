import numpy as np
import matplotlib.pyplot as plt
import qutip as qp
import os
import pandas
import scipy.optimize

from design import designW, designQ, designCW
from regression import reg_inversion, reg_cvx, reg_minimize, reg_speedy
from utilities import complexStack, complexUnstack, interpDispGrid

from data import CV_Measurement, CV_QCWigner

BASIS = 20

class CV_Measurements(CV_Measurement):
    """Organizes a list of all imported data files through 'CV_Measurement'
    and prepares the data for density matrix reconstruction.

    Note: this will be important for multiple sets of measurements, especially
    of generalized Q-functions and qubit-cavity Tomography.
    """

    def __init__(self, measurements = {}):
        #import initial list of measurement instances
        self.msmts = measurements
        self.q_ops = { 'I':qp.identity(2), 'X': qp.sigmax(),
                        'Y':qp.sigmay(), 'Z':qp.sigmaz() }

    def appendMsmt(self, key, measurement):
        self.msmts[key] = measurement

    def importMsmts(self, files):
        #import a list of files and their operators
        data = None
        displacements = None
        for key, name in files.iteritems():
            measurement = CV_QCWigner(self.q_ops[key])
            measurement.importData(name)

            temp = measurement.data_to_fit
            temp = temp.reshape( (temp.size,1) )

            if data is None:
                data = temp
            else:
                data = np.vstack( (data,temp) )

            if displacements is None:
                displacements = measurement.displacements
            else:
                displacements = np.vstack( (displacements, measurement.displacements) )

            self.appendMsmt(key, measurement)
            self.data_to_fit = data
            self.displacements = displacements

    def plotData(self):
        #plots the list of imported data
        for key in self.q_ops.iterkeys():
            self.msmts[key].plotData()

    def calcDesignMatrix(self, basis):
        #calculates the design matrix for the entire set of
        dMat = None
        N = basis
        self.basis = basis
        for key in self.q_ops.iterkeys():
            temp = self.msmts[key].calcDesignMatrix(basis = N)
            if dMat is None:
                dMat = temp
            else:
                dMat = np.vstack( (dMat, temp))

        self.design_matrix = dMat
        dMat_stack = complexStack(dMat)
        self.design_matrix_complex = dMat_stack

    def addNoiseData(self, noise_amp):
        """Adds noise to the 'data_raw'. This could be useful for testing the
        sensitivity to noisy measurements.
        """
        noise = np.random.normal(0, noise_amp, self.data_to_fit.shape)
        self.data_to_fit = self.data_to_fit + noise

    def regression(self, method = 'inversion', tolerance = 1e-6):

        if self.design_matrix is None:
            raise NameError("a design matrix must exist to perform regression")

        N = 2 * self.basis

        # real-valued design matrix
        M = self.design_matrix_complex

        W_flat = self.data_to_fit.reshape( (self.data_to_fit.size, 1) )
        print W_flat[0]
        # real-valued response vector
        W = complexStack(W_flat)
        self.response_complex = W


        if method is 'inversion':
            # perform linear regression by matrix inversion where we solve
            #   the problem: Min(M*x - W) by solving for invM  = (M.T * M)^-1 * M.T
            #   and calculating x^hat = invM * W.
            self.density_matrix_inv = reg_inversion(M, W, N,)
            self.rho = self.density_matrix_inv
            return self.density_matrix_inv

        elif method is 'speedy':
            # perform linear regression by matrix inversion where we solve
            #   the problem: Min(M*x - W) by solving for invM  = (M.T * M)^-1 * M.T
            #   and calculating x^hat = invM * W.
            self.density_matrix_inv = reg_speedy(M, W, N,)
            self.rho = self.density_matrix_inv
            return self.density_matrix_inv

        elif method is 'convex':
            # perform convex optimization using CVXPY toolbox and restricting
            #   the calculated density matrix to be postive semi-definite
            self.density_matrix_cvx = reg_cvx(M, W, N, tolerance)
            return self.density_matrix_cvx

        elif method is 'minimize':
            # performs least squares regression with a constraint that the
            # trace of the density matrix must equal one.
            self.density_matrix_min = reg_minimize(M, W, N, tolerance)
            return self.density_matrix_min

        else:
            raise TypeError("method must be either 'inversion', 'convex', or 'minimize'")

    def plotDesign(self, state = None, title = [''], show = True):
        """Plot the CV representation of a density matrix given the calculated
        design matrix. This can be used as a check to make sure that the creadted
        design matrix creates the expected quasi-probability distribution.
        """
        if self.design_matrix is None:
            raise ValueError("a design matrix must be defined")

        if state is None:
            state = qp.tensor(qp.fock_dm(2, 0), qp.fock_dm(self.basis,0))
        #truncate state to basis defined by the design matrix
        if isinstance(state, qp.Qobj):
            D_flat = state.data.toarray().reshape(( (2*self.basis)**2, 1))
        else:
            # state = state[:self.basis, :self.basis]
            D_flat = state.reshape(( (2*self.basis)**2,1))

        # D_flat = state.reshape( ( (2*self.basis)**2, 1) )

        #unpack complex arrays into their real counterparts
        D_complex = complexStack(D_flat)
        M_complex = self.design_matrix_complex

        R_complex = np.split( np.dot(M_complex,D_complex), 2)
        R = R_complex[0].reshape(self.displacements.shape)

        #the to-be plotted response variables calculated
        self.design_response = R

        if show is True:
            #fig_shape = ( np.max(np.real(self.displacements)),
            #                    np.max(np.imag(self.displacements)) )
            #fig_shape = fig_shape/np.max(fig_shape)
            #fig = plt.figure(figsize = 10*fig_shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.contourf(np.real(self.displacements), np.imag(self.displacements),
            #                  self.design_response, 200)
            ax.imshow(self.design_response)
            ax.set_title(title)

        return self.design_response.reshape(self.design_response.size,1)

    def getObs(self, q_op, c_op):

        #retrieve displacement steps in Re and Im directions
        diff = self.displacements[0,0] - self.displacements[1,1]
        #retrieve and reshape raw data
        overlap_data = self.data_to_fit.reshape(self.displacements.shape)

        #create obvservable in continuous variable basis
        #WARNING! Cavity identity requires an infinitely large truncation basis
            #do not build this using a truncated identity
        obs_discrete = qp.tensor(q_op, c_op)

        obs_continuous = self.plotDesign(state = obs_discrete, show = False)
        obs_continuous = obs_continuous.reshape(self.displacements.shape)

        #Simply do a reimann sum (note factor of two! this is due to qc_tomo instead of simply a cavity tomo)
        toSum = np.multiply(overlap_data, obs_continuous)
        result = np.sum(toSum) * np.real(diff) * np.imag(diff) * 2 / np.pi

        return result

