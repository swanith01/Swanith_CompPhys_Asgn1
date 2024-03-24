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

# Relaxation Method with L∞ norm
def relaxation(A, b, true_solution, tolerance, max_stagnation_iterations, w=1.25):
    x = np.random.rand(len(b))  # Generate a random initial guess for x
    print("Initial guess:", x)
    iterations = 0
    stagnation_count = 0
    previous_x = None
    while True:
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (1 - w) * x[i] + (w / A[i, i]) * (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:]))
        if previous_x is not None and np.allclose(x, previous_x, atol=tolerance):
            stagnation_count += 1
            if stagnation_count >= max_stagnation_iterations:
                print("Relaxation method appears to have stuck at:", x)
                break  # Stop iterations if stuck
        else:
            stagnation_count = 0
            # Compare L∞ norm with true solution during non-stagnant runs
            max_diff = np.max(np.abs(true_solution - x))
            print(f"Iteration {iterations + 1}: {x}, L∞ norm difference from true solution:", max_diff)
            if max_diff < tolerance:
                break  # Stop iterations if tolerance reached
        previous_x = x.copy()
        x = x_new
        iterations += 1
    return x, iterations

# Solve using Relaxation method with L∞ norm and print the result
solution, iterations = relaxation(A, b, true_solution, tolerance, max_stagnation_iterations)
print("Final Solution:", solution)
print("Iterations:", iterations)

'''
Initial guess: [0.98760818 0.64216175 0.38294837 0.35676563 0.08595957]
Iteration 1: [0.98760818 0.64216175 0.38294837 0.35676563 0.08595957], L∞ norm difference from true solution: 6.8721048880498445
Iteration 2: [ 0.97853437  0.46892484 -0.04027231  0.25591836 -0.01369577], L∞ norm difference from true solution: 6.881178700002471
Iteration 3: [ 4.36450056  0.27453863 -0.01321012 -0.15526961  0.01390465], L∞ norm difference from true solution: 3.495212515879021
Iteration 4: [ 6.0402865   0.41634518 -0.05078355 -0.35372171  0.0085411 ], L∞ norm difference from true solution: 1.8194265700982069
Iteration 5: [ 7.00787051  0.39925502 -0.06212772 -0.44927112  0.01049345], L∞ norm difference from true solution: 0.8518425599630817
Iteration 6: [ 7.44474076  0.41679961 -0.06794628 -0.49760631  0.01036116], L∞ norm difference from true solution: 0.4149723088303414
Iteration 7: [ 7.66301874  0.41883738 -0.07100216 -0.51986417  0.01054594], L∞ norm difference from true solution: 0.19669432626090533
Iteration 8: [ 7.76538649  0.42118731 -0.0723142  -0.53077733  0.01057721], L∞ norm difference from true solution: 0.09432657628052699
Iteration 9: [ 7.81473333  0.42206786 -0.0729946  -0.53591661  0.01060524], L∞ norm difference from true solution: 0.04497973693426793
Iteration 10: [ 7.8382193   0.42251595 -0.07330329 -0.53838901  0.01061559], L∞ norm difference from true solution: 0.02149377000739694
Iteration 11: [ 7.84944953  0.42273238 -0.07345514 -0.53956592  0.01062126], L∞ norm difference from true solution: 0.010263537134870937
Relaxation method appears to have stuck at: [ 7.85970674  0.42292629 -0.07359215 -0.54064235  0.01062616]
Final Solution: [ 7.85970674  0.42292629 -0.07359215 -0.54064235  0.01062616]
Iterations: 20

