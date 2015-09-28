#include <Python.h>
#include <iostream>
#include "theano_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <vector>
#include <algorithm>
//////////////////////
////  Support Code
//////////////////////


    namespace {
    struct __struct_compiled_op_4c4cda07dce00948adec624562fc30f7 {
        PyObject* __ERROR;

        PyObject* storage_V3;
PyObject* storage_V1;
        

        __struct_compiled_op_4c4cda07dce00948adec624562fc30f7() {}
        ~__struct_compiled_op_4c4cda07dce00948adec624562fc30f7(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V3, PyObject* storage_V1) {
            Py_XINCREF(storage_V3);
Py_XINCREF(storage_V1);
            this->storage_V3 = storage_V3;
this->storage_V1 = storage_V1;
            



            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_6:

double __DUMMY_6;

            Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V1);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
            typedef npy_float32 dtype_V1;
            
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
            typedef npy_float32 dtype_V3;
            
{

    py_V1 = Py_None;
    {Py_XINCREF(py_V1);}
    
        V1 = NULL;
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
            V3 = NULL;
            if (py_V3 == Py_None) {
                // We can either fail here or set V3 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            if (!PyArray_Check(py_V3)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V3;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V3),
                             (long int) PyArray_NDIM(tmp),
                             (long int) PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1,
                             (long int) PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1,
                             (long int) PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1,
                             (long int) PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1,
                             (long int) PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1,
                             (long int) PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1
            );
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V3) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V3));
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_4;}
            }
            
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
{
// Op class Elemwise

        npy_float32* V3_iter;
        
                npy_intp V3_n0;
                ssize_t V3_stride0;
                int V3_jump0_0;
                

            if (PyArray_NDIM(V3) < 1) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;}
            }
            
                V3_n0 = PyArray_DIMS(V3)[0];
                V3_stride0 = PyArray_STRIDES(V3)[0] / sizeof(npy_float32);
                V3_jump0_0 = (V3_stride0) - (0);
                //printf("V3_jump0_0 is:");
                //std::cout << V3_jump0_0 << std::endl;
                

        npy_float32* V1_iter;
        
                npy_intp V1_n0;
                ssize_t V1_stride0;
                int V1_jump0_0;
                
    {
        npy_intp dims[1];
        //npy_intp* dims = (npy_intp*)malloc(1 * sizeof(npy_intp));
        dims[0] = V3_n0;

        if (!V1) {
            V1 = (PyArrayObject*)PyArray_EMPTY(1, dims,
                                                    NPY_FLOAT32,
                                                    PyArray_ISFORTRAN(V3));
        }
        else {
            PyArray_Dims new_dims;
            new_dims.len = 1;
            new_dims.ptr = dims;
            PyObject* success = PyArray_Resize(V1, &new_dims, 0, NPY_CORDER);
            if (!success) {
                // If we can't resize the ndarray we have we can allocate a new one.
                PyErr_Clear();
                Py_XDECREF(V1);
                V1 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT32, 0);
            }
        }
        if (!V1) {
            {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;}
        }
    }
    
            if (PyArray_NDIM(V1) < 1) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
        __failure = 5;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_5;}
            }
            
                V1_n0 = PyArray_DIMS(V1)[0];
                V1_stride0 = PyArray_STRIDES(V1)[0] / sizeof(npy_float32);
                V1_jump0_0 = (V1_stride0) - (0);
                //printf("V1_jump0_0 is:");
                //std::cout << V1_jump0_0 << std::endl;
                

            if((PyArray_ISCONTIGUOUS(V3) && PyArray_ISCONTIGUOUS(V1)) || (PyArray_ISFORTRAN(V3) && PyArray_ISFORTRAN(V1))){
                
                    // All output have the same size
                    npy_intp n = PyArray_SIZE(V1);
                    
            dtype_V3 * V3_ptr = (dtype_V3*) PyArray_DATA(V3);
                            
            dtype_V1 * V1_ptr = (dtype_V1*) PyArray_DATA(V1);
                            
                    for(int i=0; i<n; i++){
                        
            dtype_V3& V3_i = V3_ptr[i];
                            
            dtype_V1& V1_i = V1_ptr[i];
                            
                        V1_i = exp(V3_i);;
                    }
                    
            }else{
                {
        V3_iter = (npy_float32*)(PyArray_DATA(V3));
V1_iter = (npy_float32*)(PyArray_DATA(V1));

        for (int ITER_0 = 0; ITER_0<V1_n0; ITER_0++) {
            npy_float32 &V3_i = * ( V3_iter + ITER_0 * V3_jump0_0 );
npy_float32 &V1_i = * ( V1_iter + ITER_0 * V1_jump0_0 );

            
        {
            
            V1_i = exp(V3_i);
            
        }
        
        }
        }
            }
            __label_5:

double __DUMMY_5;

}
__label_4:

        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1,
                         (long int) PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1,
                         (long int) PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1,
                         (long int) PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1,
                         (long int) PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1,
                         (long int) PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_4c4cda07dce00948adec624562fc30f7_executor(__struct_compiled_op_4c4cda07dce00948adec624562fc30f7* self) {
            return self->run();
        }

        static void __struct_compiled_op_4c4cda07dce00948adec624562fc30f7_destructor(void* executor, void* self) {
            delete ((__struct_compiled_op_4c4cda07dce00948adec624562fc30f7*)self);
        }
        
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (3 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 3, got %i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_4c4cda07dce00948adec624562fc30f7* struct_ptr = new __struct_compiled_op_4c4cda07dce00948adec624562fc30f7();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
  PyObject* thunk = PyCObject_FromVoidPtrAndDesc((void*)(&__struct_compiled_op_4c4cda07dce00948adec624562fc30f7_executor), struct_ptr, __struct_compiled_op_4c4cda07dce00948adec624562fc30f7_destructor);
  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
PyMODINIT_FUNC init4c4cda07dce00948adec624562fc30f7(void){
   import_array();
   (void) Py_InitModule("4c4cda07dce00948adec624562fc30f7", MyMethods);
}
