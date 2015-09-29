from __future__ import division

import numpy as np

def euler(alpha, x_t, r_t, Win, Wrec, brec, bout, u, noise_rec, f_hidden, r):
    for i in xrange(1, r.shape[0]):
        x_t += alpha*(-x_t            # Leak
                      + Wrec.dot(r_t) # Recurrent input
                      + brec          # Bias
                      + Win.dot(u[i]) # Input
                      + noise_rec[i]) # Recurrent noise
        r_t = f_hidden(x_t)

        r[i] = r_t

def euler_no_Win(alpha, x_t, r_t, Wrec, brec, bout, noise_rec, f_hidden, r):
    for i in xrange(1, r.shape[0]):
        x_t += alpha*(-x_t            # Leak
                      + Wrec.dot(r_t) # Recurrent input
                      + brec          # Bias
                      + noise_rec[i]) # Recurrent noise
        r_t = f_hidden(x_t)

        r[i] = r_t
