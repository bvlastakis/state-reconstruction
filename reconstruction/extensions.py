import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datas import CV_Measurements
from data import CV_Wigner


class Bell_Cat(CV_Measurements):
    '''Extension of CV_Measurements to perform operations on reconstructed
    bell cat state'''

    def __init__(self, dispAmp):
        super(Bell_Cat, self).__init__()

        self.dispAmp = dispAmp

    def makeOps(self):
        '''Makes a dictionary of qubit and logical qubit operators to use
        for projective measurements.
        '''

        N = self.basis
        #define some cavity states
        evenCat = ( qp.coherent(N, self.dispAmp) +
                        qp.coherent(N, -self.dispAmp) ).unit()
        oddCat = ( qp.coherent(N, self.dispAmp) -
                        qp.coherent(N, -self.dispAmp) ).unit()
        pjCat = ( qp.coherent(N, self.dispAmp) +
                        1j * qp.coherent(N, -self.dispAmp) ).unit()
        mjCat = ( qp.coherent(N, self.dispAmp) -
                        1j *qp.coherent(N, -self.dispAmp) ).unit()

        #Logical qubit operators
        Zp = qp.coherent_dm(N, self.dispAmp)
        Zm = qp.coherent_dm(N, -self.dispAmp)
        Xp = evenCat * evenCat.dag()
        Xm = oddCat * oddCat.dag()
        Yp = pjCat * pjCat.dag()
        Ym = mjCat * mjCat.dag()

        q_ops = { 'I':qp.identity(2),
                        'X':qp.sigmax(),
                        'Y':qp.sigmay(),
                        'Z':qp.sigmaz()
                        }

        c_ops = { 'I':Zp+Zm,
                        'X':Xp-Xm,
                        'Y':Yp-Ym,
                        'Z':Zp-Zm
                        }

        return q_ops,c_ops

    def reduceQubit(self):

        q_ops, c_ops = self.makeOps()

        rho_reduced = 0

        for q_key, q_op in q_ops.iteritems():
            for c_key, c_op in c_ops.iteritems():

                total_op = qp.tensor(q_op, c_op)
                norm = np.trace(np.dot(total_op.full(), self.rho ) )

                rho_reduced += norm * qp.tensor(q_ops[q_key],q_ops[c_key]) / 4

        self.rho_reduced = rho_reduced

    def paulibar(self, fig = None, ax = None, plot = True):

        q_ops, c_ops = self.makeOps()

        #list of strings representing each of 16 pauli observables
        # ticks = [''.join(p) for p in product('IXYZ',repeat = 2)]
        ticks = ['II', 'IX', 'IY','IZ', 'XI', 'YI', 'ZI','XX',
             'XY', 'XZ', 'YX','YY', 'YZ', 'ZX', 'ZY','ZZ']

        ops = {}
        for key1, key2 in ticks:
            ops[key1 + key2] = self.getObs(q_ops[key1], c_ops[key2])

        ops_list = []
        for tick in ticks:
            ops_list.append( ops[tick] )

        if plot is True:
            if ax is None:
                fig, ax = plt.subplots()
            ax.set_ylim([-1,1])
            index = np.arange(len(ops))
            opacity = 1
            rects1 = plt.bar(index, ops_list,
                             alpha=opacity,
                             color='b',
                             label='')

            plt.xlabel('Expectation Value')
            plt.ylabel('Observable')
            plt.title('Pauli Set')
            plt.legend()
            plt.xticks(index + 0.5, ticks )
            plt.tight_layout()
            plt.show()

        return ops, ops_list

    def buildDensityMatrix(self, ops):
        #build density matrix from set of Pauli correlations
        q_ops, c_ops = self.makeOps()

        rho = 0
        for key, val in ops.iteritems():
            rho += 0.25*val * qp.tensor(q_ops[ key[0] ], q_ops[ key[0] ])

        self.rho_overlap = rho
        return rho

class Overlap(CV_Wigner):
    ''' Method for calculating elements of the density matrix using overlap integrals.
    '''
    def __init__():
        super(Overlap, self).__init__()

    def buildDensity():

        N = self.basis
        rho = np.empty((N, N))
        for idx in np.arange(5):
            for idy in np.arange(5):
                c_op = qp.fock(N, idx) * (qp.fock(N, idy)).dag()
        self.getObs(c_op)


        return rho





