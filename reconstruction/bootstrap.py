import data as dt
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

	def __init__(self, basis = BASIS, method = 'inversion'):

		self.residualGrid = None
		self.basis = basis
		#regression method
		self.method = method

	def iniReconstruction(self, path, show = True):
		"""Import data, perform reconstruction, and calculate residuals."""

		data = dt.CV_Wigner()
		data.importData(path)
		self.data = data
		if show is True:
			data.plotData()

		#calculate and normalize imported data by forcing integral to zero
		data.calcNormalizer(method = 'trapezoid')
		#calculate design matrix for reconstruction
		data.calcDesignMatrix(basis = self.basis)
		#perform regression
		self.reconstruction = data.regression(method = self.method)
		if show is True:
			data.plotWigner(self.reconstruction, factor = 10,
                	title = 'Reconstructed Wigner function')
		#reconstructed response (wigner function)
		self.response = data.plotDesign(state = self.reconstruction, show = False)
		#residuals of reconstruction
		self.residualGrid = data.data_to_fit - self.response
		self.residual_frame = pd.DataFrame(self.residualGrid.flatten())

	def plotResidual(self):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
		ax.pcolor(np.real(self.data.displacements), np.imag(self.data.displacements),
					self.residualGrid)
		ax.set_title('Residuals')

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
		ax = self.residual_frame.hist(bins = 15)
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
			s_vec = np.random.choice(r_vec, r_vec.size)
			s_grid = s_vec.reshape(r_grid.shape)

			return s_grid

		elif method is 'regenerate':
			"""Return a generated random noise of equivalent distribution."""
			r_grid = self.residualGrid
			r_stdev = residual_frame.std(axis = 0)
			r_noise = np.random.normal(scale = r_stdev, size = r_grid)

			return r_noise

		else:
			raise ValueError("method must be either 'resample' or 'regenerate'")

	def residualBootstrap(self, number_regress = 20, method_reg = None,
								 method_sam = 'resample'):
		""" Performs repeated reconstructions on an initial reconstruction with sampled
		residuals.
		"""
		if method_reg is None:
			method_reg = self.method
		self.bootstrap_frame = pd.DataFrame([])

		for i in np.arange(number_regress):
			residual_to_add = self.sampleResidual(method = method_sam)

			self.data.data_to_fit = self.response + residual_to_add
			dM = self.data.regression(method = method_reg)
			dM_norm = np.real(dM) / np.real(np.trace(dM))
			dM_shape = dM_norm.reshape( (1,self.basis**2) )
			D = pd.DataFrame( dM_shape )
			self.bootstrap_frame = pd.concat( (self.bootstrap_frame, D),
												ignore_index = True )
			print i

	def plotBootstrap(self):
		""" Plots the resulting uncertainty from residual bootstrapping. """
		basis = self.basis
		fig = plt.figure(figsize = (10,5))
		bp = self.bootstrap_frame.boxplot()

		dd = np.arange(basis**2).reshape((basis,basis))
		diag_idx = dd.diagonal().copy()

		photon_inv = self.bootstrap_frame.ix[:,diag_idx]

		fig = plt.figure(figsize = (10,5))
		bp = photon_inv.boxplot()

	def observableCalc(self, operator = None):
		""" Calculates a mean observable along with a resulting error estimate. """
		if not isinstance(operator, qp.Qobj):
			operator = qp.Qobj(operator)
		if operator.shape[0] is not self.basis:
			raise TypeError("operator must have same basis size as density matrix")

		density_matrix = qp.Qobj(self.reconstruction)
		uncertainty_matrix = self.bootstrap_frame.std(axis = 0)
		uncertainty_matrix = uncertainty_matrix.to_dense().reshape((self.basis,self.basis))
		uncertainty_square = qp.Qobj(uncertainty_matrix**2)
		operator_square = qp.Qobj(operator.data.toarray()**2)

		mean_value = (density_matrix * operator).tr()
		std_value = np.sqrt( (uncertainty_square * operator_square).tr() )

		return mean_value, std_value







