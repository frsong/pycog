from __future__ import division

import  numpy as np
cimport numpy as np

cimport cython

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def euler(float alpha,
          np.ndarray[np.float32_t, ndim=1] x_t, 
          np.ndarray[np.float32_t, ndim=1] r_t, 
          np.ndarray[np.float32_t, ndim=2] Win, 
          np.ndarray[np.float32_t, ndim=2] Wrec, 
          np.ndarray[np.float32_t, ndim=2] Wout, 
          np.ndarray[np.float32_t, ndim=1] brec, 
          np.ndarray[np.float32_t, ndim=1] bout, 
          np.ndarray[np.float32_t, ndim=2] u, 
          np.ndarray[np.float32_t, ndim=2] noise_rec, 
          f_hidden, 
          np.ndarray[np.float32_t, ndim=2] r):
    cdef Py_ssize_t i
    for i in xrange(1, r.shape[0]):
        x_t += alpha*(-x_t            # Leak
                      + Wrec.dot(r_t) # Recurrent input
                      + brec          # Bias
                      + Win.dot(u[i]) # Input
                      + noise_rec[i]) # Recurrent noise
        r_t = f_hidden(x_t)

        r[i] = r_t

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def euler_no_Win(float alpha,
                 np.ndarray[np.float32_t, ndim=1] x_t, 
                 np.ndarray[np.float32_t, ndim=1] r_t, 
                 np.ndarray[np.float32_t, ndim=2] Wrec, 
                 np.ndarray[np.float32_t, ndim=2] Wout, 
                 np.ndarray[np.float32_t, ndim=1] brec, 
                 np.ndarray[np.float32_t, ndim=1] bout, 
                 np.ndarray[np.float32_t, ndim=2] noise_rec, 
                 f_hidden, 
                 np.ndarray[np.float32_t, ndim=2] r):
    cdef Py_ssize_t i
    for i in xrange(1, r.shape[0]):
        x_t += alpha*(-x_t            # Leak
                      + Wrec.dot(r_t) # Recurrent input
                      + brec          # Bias
                      + noise_rec[i]) # Recurrent noise
        r_t = f_hidden(x_t)

        r[i] = r_t
