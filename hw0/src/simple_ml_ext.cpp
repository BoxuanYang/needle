#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

// X: m by k
// Y: k by n
// mat_mul(Z, data, theta, batch, n, k);
void mat_mul(float *output, const float *X, const float *Y, int m, int k, int n){

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float val = 0.0f;
            for(int kk = 0; kk < k; kk++){
                // output[i][j] += X[i][kk] * Y[kk][j]
                val += X[i * k + kk] * Y[kk * n + j];
            }
            output[i * n + j] = val;
        }
    }
} 


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format: m by n
     * 
     *     y (const unsigned char *): pointer to y data, of size m
     * 
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format: n by k
     * 
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_iterations = (m + batch - 1) / batch;
    for(int iter = 0 ; iter < num_iterations; iter++){
        // get the subset of data & labels
        
        
        const float *data = &X[iter * batch * n];
        
        // Z = np.exp(data @ theta), batch by k
        float* Z = new float[batch * k];
        mat_mul(Z, data, theta, batch, n, k);
        for(size_t i = 0; i < batch * k; i++){
            Z[i] = exp(Z[i]);
        }
        
        // Z = Z / np.sum(Z, axis=1, keepdims=True)
        for(size_t i = 0; i < batch; i++){
            float summ = 0.0f;
            for(size_t j = 0; j < k; j++){
                summ += Z[i * k + j];
            }
            for(size_t j = 0; j < k; j++){
                Z[i * k + j] = Z[i * k + j] / summ;
            }
        }
        
        // Z = Z - Iy
        for(size_t i = 0; i < batch; i++){
            Z[i * k + y[iter * batch + i]] -= 1.0f;
        }

        // delta_theta = data.T @ (Z - Iy) / batch
        // data: batch by n, Z: batch by k
        // delta_theta: n by k
        
        
        float *delta_theta = new float[n * k];
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < k; j++){
                // delta_theta[i][j] += data.T[i][kk] * Z[kk][j]
                float val = 0.0f;
                for(size_t kk = 0; kk < batch; kk++){
                    val += data[kk * n + i] * Z[kk * k + j];
                }
                delta_theta[i * k + j] = val;
                
                theta[i * k + j] -= lr * delta_theta[i * k + j] / batch;
            }
        } 
        
        
        
        
        
        
        
        
        delete[] Z;
        delete[] delta_theta;
        
        
        
        
        
    }

    /// END YOUR CODE
}




/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
