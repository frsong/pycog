# Theano-based code for training excitatory-inhibitory RNNs

## Requirements

This code is written in Python and requires

- numpy
- http://deeplearning.net/software/theano/

Optional but recommended for analysis and visualization of the networks:

- matplotlib
- scipy

Due to the exploratory nature of this code, we recommend ``installing'' by adding the location of this package to your $PYTHONPATH.

## Example

The networks used to generate the figures in the paper were trained using the specifications contained in /examples/models. It's instructive, however, to consider an example that strips away the embellishments such as different sets of trials for gradient and validation datasets.

## Additional notes

It is common to see the following warning when running Theano:

RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility
  rval = __import__(module_name, {}, {}, [module_name])

This is almost always innocuous and can be safely ignored.

## License

MIT

Please cite our paper in your publications if it helps your research:

  @article{song2015,
    Author  = {Song, H. F., Yang, G. Robert, and Wang, X.-J.},
    Journal = {Unknown},
    Title   = {Training Excitatory-Inhibitory Recurrent Neural Networks: A Simple and Flexible Framework for Cognitive Tasks},
    Year    = {2015}
  }
