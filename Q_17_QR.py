import numpy as np

def qr_algorithm(matrix, tolerance=0.01):
    n = matrix.shape[0]
    
    while True:
        Q, R = np.linalg.qr(matrix)
        matrix = np.dot(R, Q)
        
        print("Matrix at iteration:")
        print(matrix)
        
        # Check if off-diagonal elements are within tolerance
        off_diag = np.abs(matrix - np.diag(np.diag(matrix)))
        if np.max(off_diag) < tolerance:
            break
    
    eigenvalues = np.diag(matrix)
    return eigenvalues

# Define the matrix
A = np.array([[5, -2], [-2, 8]])

# Compute eigenvalues using the QR algorithm
eigenvalues_QR = qr_algorithm(A)

# Compute eigenvalues using numpy.linalg.eigh
eigenvalues_eigh = np.linalg.eigh(A)[0]

print("Eigenvalues from QR algorithm:", eigenvalues_QR)
print("Eigenvalues from numpy.linalg.eigh:", eigenvalues_eigh)


#Solution:
'''Matrix at iteration:
[[6.79310345 2.48275862]
 [2.48275862 6.20689655]]
Matrix at iteration:
[[ 8.32498352 -1.70863546]
 [-1.70863546  4.67501648]]
Matrix at iteration:
[[8.8504632  0.85165883]
 [0.85165883 4.1495368 ]]
Matrix at iteration:
[[ 8.96973553 -0.38782268]
 [-0.38782268  4.03026447]]
Matrix at iteration:
[[8.99399265 0.17320695]
 [0.17320695 4.00600735]]
Matrix at iteration:
[[ 8.99881222 -0.07705516]
 [-0.07705516  4.00118778]]
Matrix at iteration:
[[8.99976533 0.03425327]
 [0.03425327 4.00023467]]
Matrix at iteration:
[[ 8.99995364 -0.01522425]
 [-0.01522425  4.00004636]]
Matrix at iteration:
[[8.99999084e+00 6.76638245e-03]
 [6.76638245e-03 4.00000916e+00]]
Eigenvalues from QR algorithm: [8.99999084 4.00000916]
Eigenvalues from numpy.linalg.eigh: [4. 9.]
'''
