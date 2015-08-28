# Excitatory-inhibitory recurrent neural networks for cognitive tasks

## Requirements

This code is written in Python (tested with 2.7) and requires

* [Theano](http://deeplearning.net/software/theano/)

Optional but recommended if you plan to run many trials with the trained networks, outside of Theano, for analysis purposes:

* [Cython](http://cython.org/)

Also optional but recommended for analysis and visualization of the networks (including examples from the paper):

* matplotlib

## Installation

Go to the pycog directory and type

```
python setup.py develop
```

If this fails for some reason, simply add the ``pycog`` path to your ``$PYTHONPATH``. To build the Cython extension, type

```
python setup.py build_ext --inplace
```

## Example

The networks used to generate the figures in the paper were trained using the specifications contained in /examples/models. It's instructive, however, to consider an example that strips away the embellishments such as different sets of trials for gradient and validation datasets.

* TODO

## Note

It is common to see the following warning when running Theano:

```
RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
  rval = __import__(module_name, {}, {}, [module_name])
```

This is almost always innocuous and can be safely ignored.

## Acknowledgments

* On the difficulty of training recurrent neural networks.                                         
  R. Pascanu, T. Mikolov, & Y. Bengio, ICML 2013.                                                  
  https://github.com/pascanur/trainingRNNs

## License

MIT

## Citation

If you find our code helpful to your work, consider giving us a shout-out in your publications:

* Song, H. F., Yang, G. Robert, and Wang, X.-J. "Training Excitatory-Inhibitory Recurrent Neural Networks: A Simple and Flexible Framework for Cognitive Tasks." 2015.
