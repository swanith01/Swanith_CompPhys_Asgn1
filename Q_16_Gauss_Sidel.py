import numpy as np

# Define the system of equations
A = np.array([[0.2, 0.1, 1, 1, 0], [0.1, 4, -1, 1, -1], [1, -1, 60, 0, -2], [1, 1, 0, 8, 4], [0, -1, -2, 4, 700]])
b = np.array([1, 2, 3, 4, 5])

# Define the true solution vector
true_solution = np.array([7.859713071, 0.422926408, -0.073592239, -0.540643016, 0.010626163])

# Define the tolerance
tolerance = 0.01

# Maximum number of consecutive iterations with no change
max_stagnation_iterations = 10

# Gauss-Seidel Method with L∞ norm
def gauss_seidel(A, b, true_solution, tolerance, max_stagnation_iterations):
    x = np.random.rand(len(b))  # Generate a random initial guess for x
    print("Initial guess:", x)
    iterations = 0
    stagnation_count = 0
    previous_x = None
    while True:
        for i in range(len(A)):
            x_new = np.zeros_like(x)
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
            x[i] = x_new[i]
        if previous_x is not None and np.allclose(x, previous_x, atol=tolerance):
            stagnation_count += 1
            if stagnation_count >= max_stagnation_iterations:
                print("Gauss-Seidel method appears to have stuck at:", x)
                break  # Stop iterations if stuck
        else:
            stagnation_count = 0
            # Compare L∞ norm with true solution during non-stagnant runs
            max_diff = np.max(np.abs(true_solution - x))
            print(f"Iteration {iterations + 1}: {x}, L∞ norm difference from true solution:", max_diff)
            if max_diff < tolerance:
                break  # Stop iterations if tolerance reached
        previous_x = x.copy()
        iterations += 1
    return x, iterations

# Solve using Gauss-Seidel method with L∞ norm and print the result
solution, iterations = gauss_seidel(A, b, true_solution, tolerance, max_stagnation_iterations)
print("Final Solution:", solution)
print("Iterations:", iterations)

'''
Initial guess: [0.92903739 0.21765947 0.47768978 0.10671764 0.52586402]
Iteration 1: [1.96913316 0.72420904 0.0675288  0.23706799 0.00714286], L∞ norm difference from true solution: 5.890579911561417
Iteration 2: [3.11491152 0.45940092 0.0502381  0.49642857 0.00714286], L∞ norm difference from true solution: 4.744801550140901
Iteration 3: [2.03696621 0.3902381  0.0502381  0.49642857 0.00714286], L∞ norm difference from true solution: 5.822746862657656
Iteration 4: [2.07154762 0.3902381  0.0502381  0.49642857 0.00714286], L∞ norm difference from true solution: 5.788165451952381
Gauss-Seidel method appears to have stuck at: [2.07154762 0.3902381  0.0502381  0.49642857 0.00714286]
Final Solution: [2.07154762 0.3902381  0.0502381  0.49642857 0.00714286]
Iterations: 13
'''
