import multiprocessing as mp
import numpy as np


def gauss_seidel(A, b, **kwargs):
    # Uses the Gauss-Seidel method to solve A*x=b

    # Get kwargs
    tolerance = kwargs.get("gs_convergence", 1e-10)
    max_iterations = kwargs.get("gs_max_iterations", 10000)
    verbose = kwargs.get("verbose", False)
    
    # Initial guess
    x = np.zeros_like(b, dtype=np.double)

    if verbose:
        print()
        print("Running Gauss-Seidel")
        print("{0:<20}{1:<20}".format("Iteration", "Error"))
    
    #Iterate
    for k in range(max_iterations):
        
        x_old  = x.copy()

        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]

        # Check error
        err = np.max(np.abs((x - x_old)/np.max(np.abs(x))))

        # Print progress on every twentieth iteration
        if verbose and k%20 == 0:
            print("{0:<20}{1:<20.5e}".format(k, err))
            
        #Stop condition 
        if err < tolerance:
            break
            
    return x


def gauss_seidel_segment(args):
    # Does Gauss-Seidel iterations on the specified slice of the problem A*x=b

    A, b, x, i_min, i_max, iterations = args
    
    # Iterate
    for k in range(iterations):
        
        # Loop over rows in the assigned slice
        for i in range(i_min, i_max):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x[(i+1):])) / A[i ,i]

    return x
            

def gauss_seidel_multiprocess(A, b, **kwargs):
    # Uses the Gauss-Seidel method to solve A*x=b with multiple processes

    # Get kwargs
    tolerance = kwargs.get("gs_convergence", 1e-10)
    max_iterations = kwargs.get("gs_max_iterations", 10000)
    verbose = kwargs.get("verbose", False)

    # Options
    sub_iterations = 1
    processes = os.cpu_count()
    processes = 4

    # Create slices of the problem for each process to tackle
    N = b.size
    N_per_process = N//processes
    
    # Do one initial iteration
    x = np.zeros_like(b)

    # Create arg list
    arg_list = []
    for i in range(processes):
        arg_list.append((A, b, x, i*N_per_process, (i+1)*N_per_process, sub_iterations))

    if verbose:
        print()
        print("Running Gauss-Seidel")
        print("{0:<20}{1:<20}".format("Iteration", "Error"))

    # Create pool
    with mp.Pool(processes) as pool:
    
        #Iterate
        for k in range(max_iterations//sub_iterations):

            # Store previous x vector
            x_old = np.copy(x)

            # Send off subprocesses
            results = pool.map(gauss_seidel_segment, arg_list)

            # Reconstruct x
            for i, result in enumerate(results):
                x[arg_list[i][3]:arg_list[i][4]] = result[arg_list[i][3]:arg_list[i][4]]

            # Check error
            err = np.max(np.abs((x - x_old)/np.max(np.abs(x))))

            # Print progress on every outer iteration
            if verbose:
                print("{0:<20}{1:<20.5e}".format(k*sub_iterations, err))

            #Stop condition 
            if err < tolerance:
                break

        return np.array(x)