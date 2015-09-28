
        #include <Python.h>
        #include "numpy/arrayobject.h"
        #include "theano_mod_helper.h"

        extern "C"{
        static PyObject *
        run_cthunk(PyObject *self, PyObject *args)
        {
          PyObject *py_cthunk = NULL;
          if(!PyArg_ParseTuple(args,"O",&py_cthunk))
            return NULL;

          if (!PyCObject_Check(py_cthunk)) {
            PyErr_SetString(PyExc_ValueError,
                           "Argument to run_cthunk must be a PyCObject.");
            return NULL;
          }
          void * ptr_addr = PyCObject_AsVoidPtr(py_cthunk);
          int (*fn)(void*) = (int (*)(void*))(ptr_addr);
          void* it = PyCObject_GetDesc(py_cthunk);
          int failure = fn(it);

          return Py_BuildValue("i", failure);
         }
        #if NPY_API_VERSION >= 0x00000008
        typedef void (*inplace_map_binop)(PyArrayMapIterObject *,
                                          PyArrayIterObject *, int inc_or_set);
        
    #if defined(NPY_INT8)
    static void npy_int8_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int8*)mit->dataptr)[0] = (inc_or_set ? ((npy_int8*)mit->dataptr)[0] : 0) + ((npy_int8*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT16)
    static void npy_int16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int16*)mit->dataptr)[0] = (inc_or_set ? ((npy_int16*)mit->dataptr)[0] : 0) + ((npy_int16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT32)
    static void npy_int32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int32*)mit->dataptr)[0] = (inc_or_set ? ((npy_int32*)mit->dataptr)[0] : 0) + ((npy_int32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT64)
    static void npy_int64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int64*)mit->dataptr)[0] = (inc_or_set ? ((npy_int64*)mit->dataptr)[0] : 0) + ((npy_int64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT128)
    static void npy_int128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int128*)mit->dataptr)[0] = (inc_or_set ? ((npy_int128*)mit->dataptr)[0] : 0) + ((npy_int128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_INT256)
    static void npy_int256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_int256*)mit->dataptr)[0] = (inc_or_set ? ((npy_int256*)mit->dataptr)[0] : 0) + ((npy_int256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT8)
    static void npy_uint8_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint8*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint8*)mit->dataptr)[0] : 0) + ((npy_uint8*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT16)
    static void npy_uint16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint16*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint16*)mit->dataptr)[0] : 0) + ((npy_uint16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT32)
    static void npy_uint32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint32*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint32*)mit->dataptr)[0] : 0) + ((npy_uint32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT64)
    static void npy_uint64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint64*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint64*)mit->dataptr)[0] : 0) + ((npy_uint64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT128)
    static void npy_uint128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint128*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint128*)mit->dataptr)[0] : 0) + ((npy_uint128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_UINT256)
    static void npy_uint256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_uint256*)mit->dataptr)[0] = (inc_or_set ? ((npy_uint256*)mit->dataptr)[0] : 0) + ((npy_uint256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT16)
    static void npy_float16_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float16*)mit->dataptr)[0] = (inc_or_set ? ((npy_float16*)mit->dataptr)[0] : 0) + ((npy_float16*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT32)
    static void npy_float32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float32*)mit->dataptr)[0] = (inc_or_set ? ((npy_float32*)mit->dataptr)[0] : 0) + ((npy_float32*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT64)
    static void npy_float64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float64*)mit->dataptr)[0] = (inc_or_set ? ((npy_float64*)mit->dataptr)[0] : 0) + ((npy_float64*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT80)
    static void npy_float80_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float80*)mit->dataptr)[0] = (inc_or_set ? ((npy_float80*)mit->dataptr)[0] : 0) + ((npy_float80*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT96)
    static void npy_float96_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float96*)mit->dataptr)[0] = (inc_or_set ? ((npy_float96*)mit->dataptr)[0] : 0) + ((npy_float96*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT128)
    static void npy_float128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float128*)mit->dataptr)[0] = (inc_or_set ? ((npy_float128*)mit->dataptr)[0] : 0) + ((npy_float128*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_FLOAT256)
    static void npy_float256_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            ((npy_float256*)mit->dataptr)[0] = (inc_or_set ? ((npy_float256*)mit->dataptr)[0] : 0) + ((npy_float256*)it->dataptr)[0];

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX32)
    static void npy_complex32_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex32*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex32*)mit->dataptr)[0].real : 0)
        + ((npy_complex32*)it->dataptr)[0].real;
    ((npy_complex32*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex32*)mit->dataptr)[0].imag : 0)
        + ((npy_complex32*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX64)
    static void npy_complex64_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex64*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex64*)mit->dataptr)[0].real : 0)
        + ((npy_complex64*)it->dataptr)[0].real;
    ((npy_complex64*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex64*)mit->dataptr)[0].imag : 0)
        + ((npy_complex64*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX128)
    static void npy_complex128_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex128*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex128*)mit->dataptr)[0].real : 0)
        + ((npy_complex128*)it->dataptr)[0].real;
    ((npy_complex128*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex128*)mit->dataptr)[0].imag : 0)
        + ((npy_complex128*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX160)
    static void npy_complex160_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex160*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex160*)mit->dataptr)[0].real : 0)
        + ((npy_complex160*)it->dataptr)[0].real;
    ((npy_complex160*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex160*)mit->dataptr)[0].imag : 0)
        + ((npy_complex160*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX192)
    static void npy_complex192_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex192*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex192*)mit->dataptr)[0].real : 0)
        + ((npy_complex192*)it->dataptr)[0].real;
    ((npy_complex192*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex192*)mit->dataptr)[0].imag : 0)
        + ((npy_complex192*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    
    #if defined(NPY_COMPLEX512)
    static void npy_complex512_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            
    ((npy_complex512*)mit->dataptr)[0].real =
        (inc_or_set ? ((npy_complex512*)mit->dataptr)[0].real : 0)
        + ((npy_complex512*)it->dataptr)[0].real;
    ((npy_complex512*)mit->dataptr)[0].imag =
        (inc_or_set ? ((npy_complex512*)mit->dataptr)[0].imag : 0)
        + ((npy_complex512*)it->dataptr)[0].imag;
    

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    static inplace_map_binop addition_funcs[] = {
#if defined(NPY_INT8)
npy_int8_inplace_add,
#endif

#if defined(NPY_INT16)
npy_int16_inplace_add,
#endif

#if defined(NPY_INT32)
npy_int32_inplace_add,
#endif

#if defined(NPY_INT64)
npy_int64_inplace_add,
#endif

#if defined(NPY_INT128)
npy_int128_inplace_add,
#endif

#if defined(NPY_INT256)
npy_int256_inplace_add,
#endif

#if defined(NPY_UINT8)
npy_uint8_inplace_add,
#endif

#if defined(NPY_UINT16)
npy_uint16_inplace_add,
#endif

#if defined(NPY_UINT32)
npy_uint32_inplace_add,
#endif

#if defined(NPY_UINT64)
npy_uint64_inplace_add,
#endif

#if defined(NPY_UINT128)
npy_uint128_inplace_add,
#endif

#if defined(NPY_UINT256)
npy_uint256_inplace_add,
#endif

#if defined(NPY_FLOAT16)
npy_float16_inplace_add,
#endif

#if defined(NPY_FLOAT32)
npy_float32_inplace_add,
#endif

#if defined(NPY_FLOAT64)
npy_float64_inplace_add,
#endif

#if defined(NPY_FLOAT80)
npy_float80_inplace_add,
#endif

#if defined(NPY_FLOAT96)
npy_float96_inplace_add,
#endif

#if defined(NPY_FLOAT128)
npy_float128_inplace_add,
#endif

#if defined(NPY_FLOAT256)
npy_float256_inplace_add,
#endif

#if defined(NPY_COMPLEX32)
npy_complex32_inplace_add,
#endif

#if defined(NPY_COMPLEX64)
npy_complex64_inplace_add,
#endif

#if defined(NPY_COMPLEX128)
npy_complex128_inplace_add,
#endif

#if defined(NPY_COMPLEX160)
npy_complex160_inplace_add,
#endif

#if defined(NPY_COMPLEX192)
npy_complex192_inplace_add,
#endif

#if defined(NPY_COMPLEX512)
npy_complex512_inplace_add,
#endif
NULL};
static int type_numbers[] = {
#if defined(NPY_INT8)
NPY_INT8,
#endif

#if defined(NPY_INT16)
NPY_INT16,
#endif

#if defined(NPY_INT32)
NPY_INT32,
#endif

#if defined(NPY_INT64)
NPY_INT64,
#endif

#if defined(NPY_INT128)
NPY_INT128,
#endif

#if defined(NPY_INT256)
NPY_INT256,
#endif

#if defined(NPY_UINT8)
NPY_UINT8,
#endif

#if defined(NPY_UINT16)
NPY_UINT16,
#endif

#if defined(NPY_UINT32)
NPY_UINT32,
#endif

#if defined(NPY_UINT64)
NPY_UINT64,
#endif

#if defined(NPY_UINT128)
NPY_UINT128,
#endif

#if defined(NPY_UINT256)
NPY_UINT256,
#endif

#if defined(NPY_FLOAT16)
NPY_FLOAT16,
#endif

#if defined(NPY_FLOAT32)
NPY_FLOAT32,
#endif

#if defined(NPY_FLOAT64)
NPY_FLOAT64,
#endif

#if defined(NPY_FLOAT80)
NPY_FLOAT80,
#endif

#if defined(NPY_FLOAT96)
NPY_FLOAT96,
#endif

#if defined(NPY_FLOAT128)
NPY_FLOAT128,
#endif

#if defined(NPY_FLOAT256)
NPY_FLOAT256,
#endif

#if defined(NPY_COMPLEX32)
NPY_COMPLEX32,
#endif

#if defined(NPY_COMPLEX64)
NPY_COMPLEX64,
#endif

#if defined(NPY_COMPLEX128)
NPY_COMPLEX128,
#endif

#if defined(NPY_COMPLEX160)
NPY_COMPLEX160,
#endif

#if defined(NPY_COMPLEX192)
NPY_COMPLEX192,
#endif

#if defined(NPY_COMPLEX512)
NPY_COMPLEX512,
#endif
-1000};
static int
map_increment(PyArrayMapIterObject *mit, PyObject *op,
              inplace_map_binop add_inplace, int inc_or_set)
{
    PyArrayObject *arr = NULL;
    PyArrayIterObject *it;
    PyArray_Descr *descr;
    if (mit->ait == NULL) {
        return -1;
    }
    descr = PyArray_DESCR(mit->ait->ao);
    Py_INCREF(descr);
    arr = (PyArrayObject *)PyArray_FromAny(op, descr,
                                0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&arr, 0);
        if (arr == NULL) {
            return -1;
        }
    }
    it = (PyArrayIterObject*)
            PyArray_BroadcastToShape((PyObject*)arr, mit->dimensions, mit->nd);
    if (it  == NULL) {
        Py_DECREF(arr);
        return -1;
    }

    (*add_inplace)(mit, it, inc_or_set);

    Py_DECREF(arr);
    Py_DECREF(it);
    return 0;
}


static PyObject *
inplace_increment(PyObject *dummy, PyObject *args)
{
    PyObject *arg_a = NULL, *index=NULL, *inc=NULL;
    int inc_or_set = 1;
    PyArrayObject *a;
    inplace_map_binop add_inplace = NULL;
    int type_number = -1;
    int i = 0;
    PyArrayMapIterObject * mit;

    if (!PyArg_ParseTuple(args, "OOO|i", &arg_a, &index,
            &inc, &inc_or_set)) {
        return NULL;
    }
    if (!PyArray_Check(arg_a)) {
        PyErr_SetString(PyExc_ValueError,
                        "needs an ndarray as first argument");
        return NULL;
    }

    a = (PyArrayObject *) arg_a;

    if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
        return NULL;
    }

    if (PyArray_NDIM(a) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL;
    }
    type_number = PyArray_TYPE(a);



    while (type_numbers[i] >= 0 && addition_funcs[i] != NULL){
        if (type_number == type_numbers[i]) {
            add_inplace = addition_funcs[i];
            break;
        }
        i++ ;
    }

    if (add_inplace == NULL) {
        PyErr_SetString(PyExc_TypeError, "unsupported type for a");
        return NULL;
    }
    mit = (PyArrayMapIterObject *) PyArray_MapIterArray(a, index);
    if (mit == NULL) {
        goto fail;
    }
    if (map_increment(mit, inc, add_inplace, inc_or_set) != 0) {
        goto fail;
    }

    Py_DECREF(mit);

    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(mit);

    return NULL;
}
        #endif
static PyMethodDef CutilsExtMethods[] = {
            {"run_cthunk",  run_cthunk, METH_VARARGS|METH_KEYWORDS,
             "Run a theano cthunk."},
            #if NPY_API_VERSION >= 0x00000008
            {"inplace_increment",  inplace_increment,
              METH_VARARGS,
             "increments a numpy array inplace at the passed indexes."},
            #endif
            {NULL, NULL, 0, NULL}        /* Sentinel */
        };
        PyMODINIT_FUNC
        initcutils_ext(void)
        {
          import_array();
          (void) Py_InitModule("cutils_ext", CutilsExtMethods);
        }
    } //extern C
        
