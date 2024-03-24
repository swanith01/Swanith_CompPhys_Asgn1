import numpy as np

def power_method(matrix, num_iterations=1000, tolerance=1e-6):
    n = matrix.shape[0]
    
    # Initialize a random vector
    x = np.random.rand(n)
    
    for _ in range(num_iterations):
        # Multiply the matrix with the current vector
        x_new = np.dot(matrix, x)
        
        # Normalize the vector, to prevent blowing up of vector components as in class.
        x_new /= np.linalg.norm(x_new)
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
    
    # Compute the dominant eigenvalue
    eigenvalue = np.dot(np.dot(x, matrix), x) / np.dot(x, x)
    
    return eigenvalue, x

# Example usage:
# Define the matrix
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

# Compute the dominant eigenvalue and eigenvector using the power method
eigenvalue_power, eigenvector_power = power_method(A)

print("Dominant Eigenvalue (Power Method):", eigenvalue_power)
print("Dominant Eigenvector (Power Method):", eigenvector_power)

# Compute the eigenvalues and eigenvectors using numpy.linalg.eigh
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Extract the dominant eigenvalue and eigenvector
dominant_index = np.argmax(np.abs(eigenvalues))
dominant_eigenvalue = eigenvalues[dominant_index]
dominant_eigenvector = eigenvectors[:, dominant_index]

print("\nDominant Eigenvalue (numpy.linalg.eigh):", dominant_eigenvalue)
print("Dominant Eigenvector (numpy.linalg.eigh):", dominant_eigenvector)

#Solution:
'''
Dominant Eigenvalue (Power Method): 3.4142135623694
Dominant Eigenvector (Power Method): [ 0.49999886 -0.70710678  0.50000114]

Dominant Eigenvalue (numpy.linalg.eigh): 3.414213562373095
Dominant Eigenvector (numpy.linalg.eigh): [ 0.5        -0.70710678  0.5       ]
'''

