import numpy as np
import time

# Define the matrices
matrices = [
    np.array([[2, 1], [1, 0]]),
    np.array([[2, 1], [1, 0], [0, 1]]),
    np.array([[2, 1], [-1, 1], [1, 1], [2, -1]]),
    np.array([[1, 1, 0], [-1, 0, 1], [0, 1, -1], [1, 1, -1]]),
    np.array([[1, 1, 0], [-1, 0, 1], [0, 1, -1], [1, 1, -1]]),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1]]),
]

# Perform SVD for each matrix
for i, matrix in enumerate(matrices):
    print(f"\nSVD for Matrix {i+1}:\n")
    start_time = time.time()
    U, S, Vt = np.linalg.svd(matrix)
    end_time = time.time()
    
    # Construct diagonal matrix from singular values
    Sigma = np.diag(S)
    
    # Ensure compatibility of dimensions for multiplication
    U = U[:, :Sigma.shape[0]]
    Vt = Vt[:Sigma.shape[1], :]
    
    # Reconstruct the original matrix using SVD
    reconstructed_matrix = np.dot(U, np.dot(Sigma, Vt))
    
    print("U:\n", U)
    print("Singular Values:\n", S)
    print("Vt:\n", Vt)
    print("Reconstructed Matrix:\n", reconstructed_matrix)
    print("Time taken:", end_time - start_time, "seconds")
    
''' Solution : - 
SVD for Matrix 1:

U:
 [[-0.92387953 -0.38268343]
 [-0.38268343  0.92387953]]
Singular Values:
 [2.41421356 0.41421356]
Vt:
 [[-0.92387953 -0.38268343]
 [ 0.38268343 -0.92387953]]
Reconstructed Matrix:
 [[2.00000000e+00 1.00000000e+00]
 [1.00000000e+00 4.38470705e-18]]
Time taken: 0.00015473365783691406 seconds

SVD for Matrix 2:

U:
 [[-9.12870929e-01  4.80183453e-17]
 [-3.65148372e-01 -4.47213595e-01]
 [-1.82574186e-01  8.94427191e-01]]
Singular Values:
 [2.44948974 1.        ]
Vt:
 [[-0.89442719 -0.4472136 ]
 [-0.4472136   0.89442719]]
Reconstructed Matrix:
 [[ 2.00000000e+00  1.00000000e+00]
 [ 1.00000000e+00 -1.25949234e-17]
 [-1.79128377e-16  1.00000000e+00]]
Time taken: 5.3882598876953125e-05 seconds

SVD for Matrix 3:

U:
 [[-0.63245553 -0.5       ]
 [ 0.31622777 -0.5       ]
 [-0.31622777 -0.5       ]
 [-0.63245553  0.5       ]]
Singular Values:
 [3.16227766 2.        ]
Vt:
 [[-1. -0.]
 [-0. -1.]]
Reconstructed Matrix:
 [[ 2.  1.]
 [-1.  1.]
 [ 1.  1.]
 [ 2. -1.]]
Time taken: 4.57763671875e-05 seconds

SVD for Matrix 4:

U:
 [[-4.36435780e-01  7.07106781e-01  4.08248290e-01]
 [ 4.36435780e-01  7.07106781e-01 -4.08248290e-01]
 [-4.36435780e-01  2.78128094e-16 -8.16496581e-01]
 [-6.54653671e-01  3.21893557e-16 -2.30903117e-16]]
Singular Values:
 [2.64575131 1.         1.        ]
Vt:
 [[-0.57735027 -0.57735027  0.57735027]
 [ 0.          0.70710678  0.70710678]
 [ 0.81649658 -0.40824829  0.40824829]]
Reconstructed Matrix:
 [[ 1.00000000e+00  1.00000000e+00  1.31165958e-16]
 [-1.00000000e+00  2.32001774e-16  1.00000000e+00]
 [-2.21127337e-16  1.00000000e+00 -1.00000000e+00]
 [ 1.00000000e+00  1.00000000e+00 -1.00000000e+00]]
Time taken: 4.5299530029296875e-05 seconds

SVD for Matrix 5:

U:
 [[-4.36435780e-01  7.07106781e-01  4.08248290e-01]
 [ 4.36435780e-01  7.07106781e-01 -4.08248290e-01]
 [-4.36435780e-01  2.78128094e-16 -8.16496581e-01]
 [-6.54653671e-01  3.21893557e-16 -2.30903117e-16]]
Singular Values:
 [2.64575131 1.         1.        ]
Vt:
 [[-0.57735027 -0.57735027  0.57735027]
 [ 0.          0.70710678  0.70710678]
 [ 0.81649658 -0.40824829  0.40824829]]
Reconstructed Matrix:
 [[ 1.00000000e+00  1.00000000e+00  1.31165958e-16]
 [-1.00000000e+00  2.32001774e-16  1.00000000e+00]
 [-2.21127337e-16  1.00000000e+00 -1.00000000e+00]
 [ 1.00000000e+00  1.00000000e+00 -1.00000000e+00]]
Time taken: 3.695487976074219e-05 seconds

SVD for Matrix 6:

U:
 [[-5.47722558e-01  7.18123969e-17  7.07106781e-01]
 [-3.65148372e-01  4.08248290e-01  1.66507414e-16]
 [-5.47722558e-01 -5.09964989e-17 -7.07106781e-01]
 [-3.65148372e-01  4.08248290e-01  1.78236286e-16]
 [-3.65148372e-01 -8.16496581e-01  7.67437979e-17]]
Singular Values:
 [2.23606798 1.41421356 1.        ]
Vt:
 [[-4.08248290e-01 -8.16496581e-01 -4.08248290e-01]
 [-5.77350269e-01  5.77350269e-01 -5.77350269e-01]
 [-7.07106781e-01  2.45660405e-16  7.07106781e-01]]
Reconstructed Matrix:
 [[-1.92849653e-16  1.00000000e+00  1.00000000e+00]
 [ 1.04764717e-16  1.00000000e+00 -2.39821445e-16]
 [ 1.00000000e+00  1.00000000e+00 -6.82925806e-16]
 [ 9.64711525e-17  1.00000000e+00 -2.31527880e-16]
 [ 1.00000000e+00 -5.83724659e-16  1.00000000e+00]]
Time taken: 4.7206878662109375e-05 seconds

'''
