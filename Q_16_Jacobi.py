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

# Jacobi Method with L∞ norm
def jacobi(A, b, true_solution, tolerance, max_stagnation_iterations):
    x = np.random.rand(len(b))
    print("Initial guess", x)
    iterations = 0
    stagnation_count = 0
    previous_x = None
    while True:
        x_new = np.zeros_like(x)
        for i in range(len(A)):
            x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        if previous_x is not None and np.allclose(x, previous_x, atol=tolerance):
            stagnation_count += 1
            if stagnation_count >= max_stagnation_iterations:
                print("Jacobi method appears to have stuck at:", x)
                break  # Stop iterations if stuck
        else:
            stagnation_count = 0
            # Compare L∞ norm with true solution during non-stagnant runs
            max_diff = np.max(np.abs(true_solution - x_new))
            print(f"Iteration {iterations + 1}: {x_new}, L∞ norm difference from true solution:", max_diff)
            if max_diff < tolerance:
                break  # Stop iterations if tolerance reached
        previous_x = x.copy()
        x = x_new
        iterations += 1
    return x, iterations

# Solve using Jacobi method with L∞ norm and print the result
solution, iterations = jacobi(A, b, true_solution, tolerance, max_stagnation_iterations)
print("Final Solution:", solution)
print("Iterations:", iterations)

'''
Initial guess [0.58255062 0.22000192 0.14596755 0.71169231 0.87072856]
Iteration 1: [ 0.60169978  0.56168719  0.07298181 -0.03568335  0.00380738], L∞ norm difference from true solution: 7.258013287867948
Iteration 2: [4.53266411 0.51307564 0.04946004 0.35267294 0.00835769], L∞ norm difference from true solution: 3.327048959423288
Iteration 3: [ 2.73279731  0.31296959 -0.01671455 -0.13489631  0.00600186], L∞ norm difference from true solution: 5.1269157599927695
Iteration 4: [5.60156953 0.46272597 0.0098696  0.11627821 0.00831304], L∞ norm difference from true solution: 2.2581435371570517
Iteration 5: [ 4.13789799  0.33543687 -0.03537029 -0.26219346  0.00716765], L∞ norm difference from true solution: 3.72181508555704
Iteration 6: [ 6.32010031  0.45505025 -0.01313543 -0.06275068  0.00901924], L∞ norm difference from true solution: 1.5396127643497133
Iteration 7: [ 5.15190543  0.35665612 -0.04745019 -0.35140344  0.00811397], L∞ norm difference from true solution: 2.707807645767198
Iteration 8: [ 6.81594011  0.44921917 -0.02965036 -0.19262718  0.00952481], L∞ norm difference from true solution: 1.0437729572987857
Iteration 9: [ 5.88677809  0.37272691 -0.05579452 -0.41290732  0.00880061], L∞ norm difference from true solution: 1.9729349769427138
Iteration 10: [ 7.15714574  0.4443089  -0.0416075  -0.28683843  0.00987538], L∞ norm difference from true solution: 0.7025673282220648
Iteration 11: [ 6.4200752   0.38484793 -0.06155143 -0.45511952  0.00929778], L∞ norm difference from true solution: 1.4396378722863439
Iteration 12: [ 7.39093081  0.44021459 -0.05027719 -0.35526428  0.01011746], L∞ norm difference from true solution: 0.4687822597055007
Iteration 13: [ 6.8076001   0.39400287 -0.06550802 -0.48395191  0.00965817], L∞ norm difference from true solution: 1.0521129755764864
Iteration 14: [ 7.5502982   0.43683551 -0.05657135 -0.40502945  0.01028399], L∞ norm difference from true solution: 0.30941486797612683
Iteration 15: [ 7.08958626  0.40092807 -0.06821491 -0.50353371  0.00991973], L∞ norm difference from true solution: 0.7701268145629996
Iteration 16: [ 7.65827908  0.43406998 -0.06114698 -0.44127416  0.01039805], L∞ norm difference from true solution: 0.20143399555475128
Iteration 17: [ 7.29507068  0.40617433 -0.07005688 -0.51674266  0.01010982], L∞ norm difference from true solution: 0.5646423874357005
Iteration 18: [ 7.73091053  0.43182213 -0.06447795 -0.46771054  0.01047576], L∞ norm difference from true solution: 0.12880254339001151
Iteration 19: [ 7.44503134  0.41015432 -0.07130228 -0.52557946  0.01024815], L∞ norm difference from true solution: 0.41468173157578736
Iteration 20: [ 7.77933155  0.43000555 -0.06690635 -0.48702229  0.01052838], L∞ norm difference from true solution: 0.08038151812466765
Iteration 21: [ 7.55464038  0.41317779 -0.07213782 -0.53143133  0.01034897], L∞ norm difference from true solution: 0.3050726939021837
Iteration 22: [ 7.81125685  0.42854461 -0.06867941 -0.50115176  0.01056375], L∞ norm difference from true solution: 0.0484562185134525
Iteration 23: [ 7.63488354  0.4154776  -0.07269308 -0.53525706  0.01042256], L∞ norm difference from true solution: 0.2248295322732412
Iteration 24: [ 7.83201189  0.42737455 -0.06997601 -0.51150642  0.01058731], L∞ norm difference from true solution: 0.02913659257369683
Iteration 25: [ 7.69372491  0.41722913 -0.07305771 -0.53771696  0.01047635], L∞ norm difference from true solution: 0.16598815968501412
Iteration 26: [ 7.8452588   0.42644078 -0.07092572 -0.51910743  0.01060283], L∞ norm difference from true solution: 0.021535583083741816
Iteration 27: [ 7.73694536  0.41856467 -0.07329354 -0.53926386  0.01051574], L∞ norm difference from true solution: 0.12276770652999414
Iteration 28: [ 7.85350468  0.42569788 -0.07162249 -0.52469662  0.0106129 ], L∞ norm difference from true solution: 0.01594639136858167
Iteration 29: [ 7.76874662  0.41958414 -0.07344302 -0.54020677  0.01054463], L∞ norm difference from true solution: 0.09096645433603179
Iteration 30: [ 7.85845687  0.42510843 -0.07213455 -0.52881366  0.01061932], L∞ norm difference from true solution: 0.01182935710892452
Iteration 31: [ 7.79218685  0.42036318 -0.07353516 -0.54075532  0.01056585], L∞ norm difference from true solution: 0.06752622362932748
Iteration 32: [ 7.86127084  0.42464183 -0.07251153 -0.53185168  0.01062331], L∞ norm difference from true solution: 0.00879133770222107
Final Solution: [ 7.79218685  0.42036318 -0.07353516 -0.54075532  0.01056585]
Iterations: 31
'''
