import numpy as np
import qutip as qp
import matplotlib.pyplot as plt


def designW(basis = 10, dispGrid = np.zeros([10,10]), method = 'iterative'):
    """Returns the design matrix to reconstruct a density matrix for a
    given grid of displacements.


    Parameters
    ----------
    basis : integer
        The truncation number of the density matrix which can be reconstructed
        from the resulting design matrix.

    dispGrid : complex matrix
        An array of complex values which represent the displacement amplitude for
        a set of measurements

    method : string {'iterative', 'operative'}
        Select method 'iterative' or 'operative', where 'iterative' use a
        iterative method to evaluate the Wigner functions for density matrices
        :math:`|m><n|`, while 'operative' uses the displacement operaters created
        by qutip to calculate the matrix. The 'iterative' method is default.

    Returns
    -------

    Wmat : array
        Values representing the design matrix for a density matrix reconstruction
        using measurements of a states Wigner function.

    Note: Iterative method is derived from 'wigner.py' found in the qutip package.

    """

    rho = qp.identity(basis)

    if method is 'operative':
        # max displacement
        maxDisp = np.max(np.abs(dispGrid))
        # truncation basis for calculating operators
        N = int(maxDisp ** 2 + 4 * maxDisp)
        a = qp.destroy(N)
        adag = a.dag()
        P = 1j * np.pi * a * adag
        P = P.expm()

        Wmat = []
        for alpha in dispGrid.flatten():
            disp = qp.displace(N, alpha)
            M = -1 * disp.dag() * P * disp
            vec = M.full()[:basis,:basis]
            Wmat.append(vec)

        #rehsape array in proper format
        Wmat = np.array(Wmat).reshape( (dispGrid.shape[0], dispGrid.shape[1], basis, basis))

        return Wmat

    elif method is 'iterative':

        M = rho.shape[0]

        Wmat = np.zeros(np.append(rho.shape, dispGrid.shape), dtype = complex)

        #initial 'seed' calculation for |0><0|
        Wmat[0][0] = np.exp(-2.0 * np.abs(dispGrid) ** 2)

        for n in range(1,basis):
            # calculate |0><n| and |n><0|
            Wmat[0][n] = (2.0 * dispGrid * Wmat[0][n-1]) / np.sqrt(n)
            Wmat[n][0] = (2.0 * np.conj(dispGrid) * Wmat[n-1][0]) / np.sqrt(n)

        for m in range(1,basis):
            # calculate |n><n|
            Wmat[m][m] = (2.0 * np.conj(dispGrid) * Wmat[m-1][m]
                           - np.sqrt(m) * Wmat[m - 1][m - 1]) / np.sqrt(m)

            for n in range(m + 1, basis):
                # calculate |m><n| and |n><m|
                Wmat[m][n] = (2.0 * dispGrid * Wmat[m][n - 1]
                               - np.sqrt(m) * Wmat[m - 1][n - 1]) / np.sqrt(n)

                Wmat[n][m] = (2.0 * np.conj(dispGrid) * Wmat[n - 1][m]
                               - np.sqrt(m) * Wmat[n - 1][m - 1] ) / np.sqrt(n)

        eps = 1e-12
        Wmat[np.abs(Wmat) < eps] = 0
        # set dimension in correct order
        Wmat = np.rollaxis(Wmat,3)
        Wmat = np.rollaxis(Wmat,3)

        # tile Wmat with qubit operator
        #CWmat = np.kron(Wmat, qubit_projection.full())
        #CWmat = np.kron(qubit_projection.full(), Wmat)

        return Wmat

    else:
        raise TypeError("method must be either 'iterative' or 'operative'")

def designQ(basis = 10, dispGrid = np.zeros([10,10]), photon_proj = 0, method = 'iterative'):
    """Determine the design matrix of a generalized Q-function for a given grid
    of displacements.
    """
    rho = qp.identity(basis)

    if method is 'iterative':

        M = rho.shape[0]
        photon_array = np.arange(photon_proj + 1)
        Q_size = np.append(rho.shape, photon_array.shape)
        Q_size = np.append(Q_size, dispGrid.shape)
        alpha = dispGrid

        Qmat = np.zeros(Q_size,dtype = complex)

        #initial 'seed' calculation for |0><0|, 0 photon
        Qmat[0][0][0] = np.exp( -np.abs(alpha) ** 2)

        for k in np.arange(1,basis):

            Qmat[0][k][0] = (alpha * Qmat[0][k-1][0]) / np.sqrt(k)
            Qmat[k][0][0] = (np.conj(alpha) * Qmat[k-1][0][0]) / np.sqrt(k)

        for k in np.arange(1,basis):
            # calculate |n><n|
            Qmat[k][k][0] = (np.abs(alpha)**2 * Qmat[k-1][k-1][0]) / k

            for l in np.arange(k + 1, basis):
                # calculate |m><n| and |n><m|
                Qmat[l][k][0] = (np.conj(alpha) * Qmat[l-1][k][0]) / np.sqrt(l)

                Qmat[k][l][0] = (alpha * Qmat[k][l-1][0]) / np.sqrt(l)

        for n in np.arange(1, photon_proj+1):

            Qmat[0][0][n] = np.abs(alpha)**2 * Qmat[0][0][n-1] / n

            for k in np.arange(1, basis):
                # calculate |k><0| for n photon
                Qmat[0][k][n] = ( (1./n) * (np.abs(alpha)**2 * Qmat[0][k][n-1] -
                                    alpha * Qmat[0][k-1][n-1] * np.sqrt(k) ) )

                Qmat[k][0][n] = ( (1./n) * (np.abs(alpha)**2 * Qmat[k][0][n-1] -
                                np.conj(alpha) * Qmat[k-1][0][n-1] * np.sqrt(k) ) )

            for k in np.arange(1, basis):
                 for l in np.arange(1, basis):
                    # calculate |k><k|
                    Qmat[l][k][n] = ( (1./(n)) * ( 1.*np.sqrt(l*k) * Qmat[l-1][k-1][n-1]
                            - (alpha) * Qmat[l][k-1][n-1] * np.sqrt(k)
                            - np.conj(alpha) * Qmat[l-1][k][n-1] * np.sqrt(l)
                            + np.abs(alpha)**2 * Qmat[l][k][n-1] ) )


        eps = 1e-12
        Qmat[np.abs(Qmat) < eps] = 0
        # set dimension in correct order
        Qmat = np.rollaxis(Qmat,4)
        Qmat = np.rollaxis(Qmat,4)

        return Qmat

def designCW(basis = 10, dispGrid = np.zeros([10,10]), q_proj = None, method = 'iterative'):
    """Returns the design matrix to reconstruct a density matrix for a
    given grid of displacements.


    Parameters
    ----------
    basis : integer
        The truncation number of the density matrix which can be reconstructed
        from the resulting design matrix.

    dispGrid : complex matrix
        An array of complex values which represent the displacement amplitude for
        a set of measurements

    method : string {'iterative', 'operative'}
        Select method 'iterative' or 'operative', where 'iterative' use a
        iterative method to evaluate the Wigner functions for density matrices
        :math:`|m><n|`, while 'operative' uses the displacement operaters created
        by qutip to calculate the matrix. The 'iterative' method is default.

    Returns
    -------

    Wmat : array
        Values representing the design matrix for a density matrix reconstruction
        using measurements of a states Wigner function.

    Note: Iterative method is derived from 'wigner.py' found in the qutip package.

    """

    rho = qp.identity(basis)

    if method is 'operative':
        # max displacement
        maxDisp = np.max(np.abs(dispGrid))
        # truncation basis for calculating operators
        N = maxDisp ** 2 + 4 * maxDisp
        a = qp.destroy(N)
        adag = a.dag()
        P = 1j * np.pi * a * adag
        P = P.expm()

        Wmat = []
        for alpha in dispGrid.flatten():
            disp = qp.displace(N, alpha)
            M = -1 * disp.dag() * P * disp
            vec = M.full()[:basis,:basis]
            Wmat.append(vec)

        #rehsape array in proper format
        Wmat = np.array(Wmat).reshape( (dispGrid.shape[0], dispGrid.shape[1], basis, basis))

        return Wmat

    elif method is 'iterative':

        M = rho.shape[0]

        Wmat = np.zeros(np.append(rho.shape, dispGrid.shape), dtype = complex)

        #initial 'seed' calculation for |0><0|
        Wmat[0][0] = np.exp(-2.0 * np.abs(dispGrid) ** 2)

        for n in range(1,basis):
            # calculate |0><n| and |n><0|
            Wmat[0][n] = (2.0 * dispGrid * Wmat[0][n-1]) / np.sqrt(n)
            Wmat[n][0] = (2.0 * np.conj(dispGrid) * Wmat[n-1][0]) / np.sqrt(n)

        for m in range(1,basis):
            # calculate |n><n|
            Wmat[m][m] = (2.0 * np.conj(dispGrid) * Wmat[m-1][m]
                           - np.sqrt(m) * Wmat[m - 1][m - 1]) / np.sqrt(m)

            for n in range(m + 1, basis):
                # calculate |m><n| and |n><m|
                Wmat[m][n] = (2.0 * dispGrid * Wmat[m][n - 1]
                               - np.sqrt(m) * Wmat[m - 1][n - 1]) / np.sqrt(n)

                Wmat[n][m] = (2.0 * np.conj(dispGrid) * Wmat[n - 1][m]
                               - np.sqrt(m) * Wmat[n - 1][m - 1] ) / np.sqrt(n)

        eps = 1e-12
        Wmat[np.abs(Wmat) < eps] = 0
        # set dimension in correct order
        Wmat = np.rollaxis(Wmat,3)
        Wmat = np.rollaxis(Wmat,3)

        # tile Wmat with qubit operator
        #CWmat = np.kron(Wmat, q_proj.full())
        CWmat = np.kron(q_proj.full(), Wmat)

        return CWmat

    else:
        raise TypeError("method must be either 'iterative' or 'operative'")



if __name__ == '__main__':
    '''A quick example for plotting a fock state using the generating matrix calculated above.'''

    #Create displacement grid (note that using an iterative approach, the range does not matter)
    d_real = np.linspace(-2, 2, 101)
    m_real, m_imag = np.meshgrid(d_real, d_real)
    dispGrid = m_real + 1j*m_imag

    N = 15 #basis is only needed for the state not the displacements
    state = qp.fock_dm(N, 3) #3 photon fock state

    #create generating matrix
    genMat = designW(basis = N, dispGrid = dispGrid, method = 'iterative')

    #calculate Wigner
    state = state.data.toarray()
    toTrace = np.multiply(genMat, state)
    wigner = np.real(np.trace(toTrace, axis1 = 2, axis2 = 3))
    print np.min(wigner)
    #plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.pcolor(d_real, d_real, wigner)
    fig.savefig('test.png')



