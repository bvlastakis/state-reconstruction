import qutip_plus as qp
from data import CV_Measurements
import numpy as np


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

        c_ops = { 'I':qp.identity(N),
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