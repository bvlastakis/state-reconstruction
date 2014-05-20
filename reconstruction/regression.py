import numpy as np
import cvxpy as cp
import cvxopt
import scipy

from utilities import complexStack, complexUnstack, interpDispGrid

def reg_inversion(M, W, N):
    #Unconstrained regression using matrix inversion
    invM = np.dot( np.linalg.inv( np.dot(M.T, M) ), M.T )
    #self.inverse_matrix = invM
    x = np.dot(invM, W)

    return complexUnstack(x).reshape( (N, N) )

def reg_cvx(M, W, N, tolerance):
    # Perform convex optimization using CVXPY toolbox and restricting
    #   the calculated density matrix to be postive semi-definite.
    #   Note that the trace of rho is not forced to be one.
    x = cp.SDPVar(2*N, 2*N)
        
    x_flat = cp.Variable(0)

    # reshape x to a flattend real-value vector
    #   this is a possible source of slowdown
    for idx in np.arange(x.size[1]/2):
        x_flat = cp.vstack(x_flat, x[:x.size[1]/2, idx])
    for idx in (x.size[1]/2 + np.arange(x.size[1]/2)):
        x_flat = cp.vstack(x_flat, x[:x.size[1]/2, idx])
        
    objective = cp.Minimize(cp.sum(cp.square(M * x_flat - W)))
    constraints = [ cp.lambda_min(x) >= 0 ]

    p = cp.Problem(objective, constraints)
    #solve problem
    result = p.solve()
    #remove values below tolerance
    x_sym = np.array(x.value)
    x_sym[np.abs(x_sym) < tolerance] = 0

    return complexUnstack(x_sym)

def reg_minimize(M, W, N, tolerance):
    # Perform least squares regression with trace equal one constraint
    # and forcing matrix to be postive semidefinite, which is 
    # based off of Reinier's fitting code.
    x0 = np.eye(N, N) / N
    x0 = x0.reshape((x0.size,1))
    x0 = complexStack(x0)

    ret = scipy.optimize.minimize( error_function, x0, 
                                    method='SLSQP', 
                                    args = (M, W), 
                                    tol = 1e-3 ,
                                    options = {'maxiter':10000, 'disp':True},
                                    constraints = [
                                        dict(type='eq', fun=trace1)] )

    x = ret['x']
    x = x.reshape((2*N**2, 1))
    x_complex = complexUnstack(x).reshape(N, N)
    x_square = np.dot(np.conj(x_complex).T, x_complex)
    x_sym = complexStack(x_square)
    
    x_sym[np.abs(x_sym) < tolerance] = 0

    x_final = complexUnstack(x_sym).reshape((N,N))

    return x_final

def error_function(x, M, W):
    N = np.sqrt(x.size/2).astype(int)

    x = x.reshape( (2*N**2, 1) )
    x_complex = complexUnstack(x).reshape(N, N)
    x_square = np.dot(np.conj(x_complex).T,x_complex)
    x_sym = complexStack(x_square)
    
    x_test = x_sym[:,:N].reshape((2*N**2, 1))

    err = np.sum(np.abs(W - np.dot(M, x_test) ))
    return err

def trace1(x):
    N = np.sqrt(x.size/2).astype(int)

    x = x.reshape( (2*N**2, 1) )
    x_complex = complexUnstack(x).reshape(N, N)
    x_square = np.dot(np.conj(x_complex).T,x_complex)

    return np.trace(x_square) - 1