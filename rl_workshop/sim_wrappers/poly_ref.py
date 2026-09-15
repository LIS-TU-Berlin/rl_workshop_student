import numpy as np
import math

class SecondOrderPolyRef:
    def __init__(self, t0, x0, v0, action_delta, lmbda, xi=1.):
        self.t0 = t0
        self.x0 = x0.copy()
        self.v0 = v0.copy()
        self.coeff = (action_delta-2.*xi*lmbda*v0)/(2.*lmbda*lmbda) #see overleaf notes!

    def eval(self, t, single_row=-1):
        d = t - self.t0
        if single_row==-1:
            if isinstance(d, np.ndarray): #t may be scalar or vector
                return np.multiply.outer(d*d, self.coeff) + np.multiply.outer(d, self.v0) + self.x0
            else:
                return self.coeff*(d*d) + self.v0*d + self.x0
        else:
            return self.coeff[single_row]*(d*d) + self.v0[single_row]*d + self.x0[single_row]

    def eval_vel(self, t):
        d = t - self.t0
        return self.coeff*d + self.v0

    def reshape(self, num_rows):
        self.coeff = self.coeff.reshape(num_rows, -1)
        self.v0 = self.v0.reshape(num_rows, -1)
        self.x0 = self.x0.reshape(num_rows, -1)

    def reset(self, x0, row):
        assert self.x0.ndim==2
        self.coeff[row] *= 0.
        self.v0[row] *= 0.
        self.x0[row] = x0

    def sample_buffer(self, t, t_stop, tau, single_row=-1):
        T = np.arange(t, t_stop+1e-10, tau)
        assert math.fabs(T[-1]-t_stop<1e-10), f'{T[-1]}, {t_stop} not equal'
        return self.eval(T, single_row)
