#bootstrap_test.py
import datas as dt
reload(dt)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qp

BASIS = 10

class BS_residual(object):
    """Organizes a set of reconstructions to calculate uncertainty in fitted values. We
    will use a method called 'residual bootstrapping' where we take a dataset, perform
    a reconstruction, determine residuals, resample residuals, and add them to the
    reconstructed data. Performing many iterations will determine uncertainty in the
    original reconstructed dataset.
    """

    def __init__(self, basis = BASIS, method = 'speedy'):

        self.residualGrid = None
        self.basis = basis
        self.method = method
        self.observable = None
        self.space = None

    def iniReconstruction(self, file_list, show = True):
        """Import data, perform reconstruction, and calculate residuals."""

        exp = dt.CV_Measurements()
        exp.importMsmts(file_list)
        self.exp = exp
        # self.data = data
        # if show is True:
        #     data.plotData()

        # #calculate and normalize imported data by forcing integral to zero
        # data.calcNormalizer(method = 'trapezoid')
        #calculate design matrix for reconstruction
        self.exp.calcDesignMatrix(basis = self.basis)
        #perform regression
        self.reconstruction = self.exp.regression(method = self.method)
        # if show is True:
        #     data.plotWigner(self.reconstruction, factor = 10,
        #             title = 'Reconstructed Wigner function')
        #reconstructed response (wigner function)
        self.response = exp.plotDesign(state = self.reconstruction, show = False)
        #residuals of reconstruction
        self.residualGrid = self.exp.data_to_fit - self.response
        self.residual_frame = pd.DataFrame(self.residualGrid.flatten())

    def plotResidual(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        # ax.pcolor(np.real(self.exp.displacements), np.imag(self.data.displacements),
        #             self.residualGrid)
        ax.pcolor(self.residualGrid)
        ax.set_title('Residuals')

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        self.residual_frame.hist(bins = 25)
        ax.set_title('Residual distribution')

        print self.residual_frame.describe()

    def sampleResidual(self, method = 'resample'):
        """Creates a new residual grid based off of two different methods, 'resample'
        and 'regenerate'. Returns a newly sample residual grid
        """

        if method is 'resample':
            """Return a random resampling (without replacement) of measured residuals."""
            r_grid = self.residualGrid
            r_vec = r_grid.flatten()
            s_vec = np.random.choice(r_vec, r_vec.size, replace = False)
            s_grid = s_vec.reshape(r_grid.shape)

            return s_grid

        elif method is 'regenerate':
            """Return a generated random noise of equivalent distribution."""
            r_grid = self.residualGrid
            r_stdev = self.residual_frame.std(axis = 0)
            r_noise = np.random.normal(scale = r_stdev, size = r_grid.size)

            return r_noise

        else:
            raise ValueError("method must be either 'resample' or 'regenerate'")

    def residualBootstrap(self, number_regress = 5, method_reg = None,
                                 method_sam = 'resample'):
        """ Performs repeated reconstructions on an initial reconstruction with sampled
        residuals.
        """
        if method_reg is None:
            method_reg = self.method
        self.bootstrap_frame = pd.DataFrame([])

        if self.observable is not None:
            self.observ_frame = pd.DataFrame([])

        for i in np.arange(number_regress):
            residual_to_add = self.sampleResidual(method = method_sam)

            self.exp.data_to_fit = self.response - residual_to_add
            dM = self.exp.regression(method = method_reg)
            D = pd.DataFrame(np.real(dM.reshape( (1,(2*self.basis)**2) )) )
            self.bootstrap_frame = pd.concat( (self.bootstrap_frame, D),
                                                ignore_index = True )
            if self.observable is not None:
                a = self.observableCalc(recon = dM, operator = self.observable)
                b = self.spaceCalc(recon = dM, space = self.space)
                c = self.detectCalc(recon = dM)
                all = pd.DataFrame(np.array([[a, b, c]]))
                self.observ_frame = pd.concat( (self.observ_frame, all),
                                                ignore_index = True )
            print i

    def plotBootstrap(self):
        """ Plots the resulting uncertainty from residual bootstrapping. """
        basis = self.basis
        fig = plt.figure(figsize = (10,5))
        bp = self.bootstrap_frame.boxplot()

        dd = np.arange((2*basis)**2).reshape((2*basis,2*basis))
        diag_idx = dd.diagonal().copy()

        photon_inv = self.bootstrap_frame.ix[:,diag_idx]

        fig = plt.figure(figsize = (10,5))
        bp = photon_inv.boxplot()


    def observableCalc(self, recon, operator = None):
        """ Calculates a mean observable along with a resulting error estimate. """

        observ = operator.data.toarray()
        density_matrix = recon.reshape((2*self.basis), (2*self.basis))
        mean_value = np.trace(np.dot(density_matrix, observ))

        return np.real(mean_value)

    def spaceCalc(self, recon, space = None):
        density_matrix = recon.reshape( (2*self.basis), (2*self.basis) )

        fidelity = 0
        for op in space:
            op_mat = op.data.toarray()
            fidelity += np.trace(np.dot(density_matrix, op_mat))

        return np.real(fidelity)

    def detectCalc(self, recon):
        density_matrix = recon.reshape( (2*self.basis), (2*self.basis) )
        return np.real(np.trace(density_matrix))






