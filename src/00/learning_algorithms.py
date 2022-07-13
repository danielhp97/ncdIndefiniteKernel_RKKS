import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import logging
from qpsolvers import solve_qp


__logger = logging.getLogger("Learners")


def kernel_matrix(eig_vals, eig_vecs):
	return np.multiply(eig_vecs, eig_vals.reshape(1, -1)).dot(eig_vecs.T)


class EigenDecomposition(BaseEstimator, TransformerMixin):
	""" Computes the sign of the eigenvalues of kernel matrix. Defining it as a transformer and using caching
	reduces the number of eigendecompositions required, speeding up a grid search."""

	def transform(self, X):
		return X

	def fit_transform(self, X, y=None, **fit_params):
		eigvals, eigvecs = np.linalg.eigh(X)
		return {"kmat": X, "eigvals": eigvals, "eigvecs": eigvecs}


class Model(BaseEstimator, ClassifierMixin):
	def __init__(self, **kwargs):
		self.w_ = None
		self.requires_eigen_decomposition = False

	def set_weights(self, w):
		self.w_ = w

	def weights(self):
		return self.w_


class SquareHingeKernelSVM(Model):
	def __init__(self, C=1.0):
		super().__init__()
		self.C = C
		self.requires_transform = False
		self.transform_mat = None

	def _kernel_matrix(self, eig_vals, eig_vecs):
		# should I change this? ask for a parameter on a fit that is the kernel matrix. and just return the kernel I put
		return np.multiply(eig_vecs, eig_vals.reshape(1, -1)).dot(eig_vecs.T)

	def _fit_transform_mat(self, X):
		raise NotImplementedError

	def spectral_transform(self, X):
		return X @ self.transform_mat

	def fit(self, X, y):
		if self.requires_transform:
			self._fit_transform_mat(X)
			self._fit(self.spectral_transform(X), y)
		else:
			self._fit(X, y)
		return self

	def predict(self, X):
		if self.requires_transform:
			pred = self._predict(self.spectral_transform(X))
		else:
			pred = self._predict(X)
		return pred

	def _fit(self, X, y):
		if 0 in y:
			y[y == 0] = -1
		n = X.shape[0]
		P = np.multiply(X, np.outer(y, y))
		np.fill_diagonal(P, P.diagonal() + 1/self.C)
		one_vec = np.ones((n,))
		q = -1*one_vec
		G = -1*np.eye(n)
		h = np.zeros((n, ))
		sol = solve_qp(P=P, q=q, G=G, h=h)
		beta = np.array(sol['x']).reshape(-1, ) 
		self.w_ = np.multiply(beta, y)
		return self

	def _predict(self, X):
		return np.squeeze(X).dot(self.w_).reshape(-1)

	def get_params(self, deep=True):
		return {"C": self.C}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	@property
	def _pairwise(self):
		return True


class SquareHingeKreinSVM(Model):
	def __init__(self, mse_lambda_p=1.0, mse_lambda_m=1.0, with_bias=False):
		super().__init__()
		self.mse_lambda_p = mse_lambda_p
		self.mse_lambda_m = mse_lambda_m
		self.with_bias = with_bias
		self.requires_eigen_decomposition = True

	def fit(self, X, y):
		if np.any(y == 0):
			y[y == 0] = -1
		if isinstance(X, dict):
			eigvals = X["eigvals"]
			eigvecs = X["eigvecs"]
		else:
			kmat = X
			eigvals, eigvecs = np.linalg.eigh(kmat)

		n = eigvals.shape[0]
		signs = np.sign(eigvals)
		pm_args = (np.argwhere(signs >= 0).reshape(-1), np.argwhere(signs < 0).reshape(-1))
		scale_vec = np.ones((n, ))
		scale_vec[pm_args[0]] /= self.mse_lambda_p
		scale_vec[pm_args[1]] /= self.mse_lambda_m
		regularised_kmat = kernel_matrix(np.multiply(np.absolute(eigvals), scale_vec), eigvecs)
		P = np.multiply(regularised_kmat, np.outer(y, y))
		np.fill_diagonal(P, P.diagonal() + 1.0)
		one_vec = np.ones((n,))
		q = -1*one_vec
		G = -1*np.eye(n)
		h = np.zeros((n, )) # dot
		if self.with_bias:
			A = y.reshape(1, -1).astype(np.float64)
			sol = solve_qp(P= P, q=q, G=G, h=h, A=A, b=np.zeros((1,), dtype=np.float64),
			         options={'show_progress': False})
			beta = np.array(sol['x']).reshape(-1, )
			K = kernel_matrix(eigvals, eigvecs)
			pred = (regularised_kmat*y).dot(beta)
			self.w_ = np.linalg.lstsq(a=K, b=pred, rcond=None)[0]
			active_args = np.argwhere(beta > 0)
			b = y[active_args]*(1 + beta[active_args]) - pred[active_args]
			self.b_ = np.mean(b)
		else:
			sol = solve_qp(P=P, q=q, G=G, h=h)
			beta = np.array(sol).reshape(-1, )
			K = kernel_matrix(eigvals, eigvecs)
			pred = (regularised_kmat*y).dot(beta)
			self.w_ = np.linalg.lstsq(a=K, b=pred, rcond=None)[0]
			self.b_ = 0
		return self

	def predict(self, X):
		fx = np.squeeze(X).dot(self.w_).reshape(-1)
		return fx + self.b_

	def get_params(self, deep=True):
		return {"mse_lambda_p": self.mse_lambda_p, "mse_lambda_m": self.mse_lambda_m}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	@property
	def _pairwise(self):
		return True
