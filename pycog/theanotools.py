"""
Theano-specific functions.

"""
import numpy as np

import theano
import theano.tensor as T

#=========================================================================================
# Activation functions and some of their derivatives
#=========================================================================================

def rectify(x):
    return T.switch(x < 0, 0, x)

def d_rectify(x):
    return T.switch(x < 0, 0, 1)

sigmoid = T.nnet.sigmoid

def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

tanh = T.tanh

def d_tanh(x):
    return 1 - tanh(x)**2

def rtanh(x):
    return rectify(tanh(x))

def d_rtanh(x):
    return T.switch(x < 0, 0, d_tanh(x))

def softplus(x):
    return T.log(1 + T.exp(x))

d_softplus = sigmoid

def softmax(x):
    """
    Softmax function.

    Parameters
    ----------

    x : theano.tensor.tensor3
        This function assumes the outputs are the third dimension of x.

    """
    sh = x.shape
    x  = x.reshape((sh[0]*sh[1], sh[2]))
    fx = T.nnet.softmax(x)
    fx = fx.reshape(sh)

    return fx

#-----------------------------------------------------------------------------------------
# Gather all functions into a convenient dictionary.
#-----------------------------------------------------------------------------------------

hidden_activations = {
    'linear':   (lambda x: x, lambda x: 1),
    'rectify':  (rectify,     d_rectify),
    'sigmoid':  (sigmoid,     d_sigmoid),
    'tanh':     (tanh,        d_tanh),
    'rtanh':    (rtanh,       d_rtanh),
    'softplus': (softplus,    d_softplus)
}

output_activations = {
    'linear':  (lambda x: x),
    'rectify': rectify,
    'sigmoid': sigmoid,
    'softmax': softmax
    }

#=========================================================================================
# Loss functions
#=========================================================================================

def binary_crossentropy(y, t):
    return -t*T.log(y + 1e-10) - (1-t)*T.log((1-y) + 1e-10)

def categorical_crossentropy(y, t):
    return -t*T.log(y + 1e-10)

def L2(y, t):
    return (y - t)**2

#=========================================================================================
# Theano
#=========================================================================================

def grad(*args, **kwargs):
    kwargs.setdefault('disconnected_inputs', 'warn')

    return T.grad(*args, **kwargs)

def function(*args, **kwargs):
    kwargs.setdefault('on_unused_input', 'warn')

    return theano.function(*args, **kwargs)

#=========================================================================================
# NumPy-Theano
#=========================================================================================

def shared(x, dtype=theano.config.floatX, **kwargs):
    if x.dtype == dtype:
        return theano.shared(x, **kwargs)
    return theano.shared(np.asarray(x, dtype=dtype), **kwargs)

def shared_scalar(c, dtype=theano.config.floatX, **kwargs):
    return theano.shared(np.cast[dtype](x), **kwargs)

def shared_zeros(shape, dtype=theano.config.floatX, **kwargs):
    return shared(np.zeros(shape), dtype=dtype, **kwargs)

#=========================================================================================
# GPU
#=========================================================================================

def get_processor_type():
    """
    Test whether the GPU is being used, based on the example in

      http://deeplearning.net/software/theano/tutorial/using_gpu.html

    """
    rng = np.random.RandomState(22)
    
    n = 10*30*768
    x = shared(rng.rand(n))
    f = function([], T.exp(x))

    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        return 'cpu'
    return 'gpu'
