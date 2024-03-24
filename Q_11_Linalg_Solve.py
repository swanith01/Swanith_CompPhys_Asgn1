import numpy as np

# Coefficient matrices
a1 = np.array([[3, -1, 1], [3, 6, 2], [3, 3, 7]])
a2 = np.array([[10, -1, 0], [-1, 10, -2], [0, -2, 10]])
a3 = np.array([[10, 5, 0, 0], [5, 10, -4, 0], [0, -4, 8, -1], [0, 0 ,-1, 5]])
a4 = np.array([[4, 1, 1, 0, 1], [-1, -3, 1, 1, 0], [2, 1, 5, -1, -1], [-1, -1, -1, 4, 0], [0, 2, -1, 1, 4]])

# Right-hand side vectors
b1 = np.array([1, 0, 4])
b2 = np.array([9, 7, 6])
b3 = np.array([6, 25, -11, -11])
b4 = np.array([6, 6, 6, 6, 6])

# Solve the equations
x1 = np.linalg.solve(a1, b1)
x2 = np.linalg.solve(a2, b2)
x3 = np.linalg.solve(a3, b3)
x4 = np.linalg.solve(a4, b4)

# Print the solutions
print("Solution 1:", x1)
print("Solution 2:", x2)
print("Solution 3:", x3)
print("Solution 4:", x4)

#Solution 1: [ 0.03508772 -0.23684211  0.65789474]
#Solution 2: [0.99578947 0.95789474 0.79157895]
#Solution 3: [-0.79764706  2.79529412 -0.25882353 -2.25176471]
#Solution 4: [ 0.78663239 -1.00257069  1.86632391  1.9125964   1.98971722]

