import numpy as np
import matplotlib.pyplot as plt
from oct2py import Oct2Py
from scipy.sparse import diags
from scipy.linalg import solve


def f(x, y):
    return np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

def gmres(alpha, beta, N, max_iter):
    oc = Oct2Py()
    residuals = oc.gmres_solver(alpha, beta, N, max_iter)
    residuals = np.array(residuals).flatten()
    return residuals

def cg(alpha, beta, N, max_iter):
    oc = Oct2Py()
    residuals = oc.cg_solver(alpha, beta, N, max_iter)
    residuals = np.array(residuals).flatten()
    return residuals

def fom(A, b, m_max = 100):
    r = b
    res = np.linalg.norm(r)
    V = np.empty((A.shape[0], m_max), dtype='float64')
    V[:, 0] = r / res
    H = np.zeros((m_max, m_max), dtype='float64')
    
    r_norms = [1]
        
    for m in range(1, m_max):
        tmp = A @ V[:, m - 1]
        H[:m, m - 1] = V[:, :m].T @ tmp
        w = tmp - V[:, :m] @ H[:m, m - 1]
        hn = np.linalg.norm(w)
        V[:, m] = w[:] / hn
        H[m, m - 1] = hn 

        y = solve(H[:m, :m], V[:, :m].T @ b)
        
        x = V[:, :m] @ y
        r = b - A @ x
        rnorm = np.linalg.norm(r)
        r_norms.append(rnorm / res)            

        if (rnorm / res < 1e-3):
            break
            
    return r_norms

def create_system(N, alpha, beta):
    h = 1 / (N + 1)

    main_diag = 2 * np.ones(N)
    off_diag = -1 * np.ones(N-1)
    T = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N)).toarray() / h**2
    I = np.eye(N)
    A = np.kron(I, T) + np.kron(T, I)

    Ax = np.kron(I, diags([-1, 1], [-1, 1], shape=(N, N)).toarray()) / (2*h)
    Ay = np.kron(diags([-1, 1], [-1, 1], shape=(N, N)).toarray(), I) / (2*h)

    A = A + alpha * Ax + beta * Ay

    x = np.linspace(0, 1, N+2)[1:-1]
    y = np.linspace(0, 1, N+2)[1:-1]
    X, Y = np.meshgrid(x, y)
    f = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2)).flatten()
    b = f

    return A, b


def iterative_solver(A, b, method="jacobi", block_size=3, max_iter=1000, rtol=1e-6):
    x = np.zeros_like(b)
    r_norms = []
    N = int(np.sqrt(len(b)))

    if method == "jacobi":
        D_inv = 1.0 / A.diagonal()
        for _ in range(max_iter):
            r = b - A @ x
            x += D_inv * r
            r_norms.append(np.linalg.norm(r) / np.linalg.norm(b))
            if r_norms[-1] < rtol:
                break

    elif method == "gauss-seidel":
        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(len(b)):
                sum1 = A[i, :i] @ x_new[:i]
                sum2 = A[i, i+1:] @ x[i+1:]
                x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            r = b - A @ x_new
            r_norms.append(np.linalg.norm(r) / np.linalg.norm(b))
            if r_norms[-1] < rtol:
                break
            x = x_new

    elif method == "block-jacobi":
        num_blocks = len(b) // block_size
        for _ in range(max_iter):
            x_new = np.copy(x)
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, len(b))
                
                A_block = A[start:end, start:end]
                b_block = b[start:end] - A[start:end, :] @ x + A_block @ x[start:end]
                
                if np.linalg.cond(A_block) < 1 / np.finfo(A_block.dtype).eps: 
                    x_new[start:end] = np.linalg.solve(A_block, b_block)
            
            r = b - A @ x_new
            r_norms.append(np.linalg.norm(r) / np.linalg.norm(b))
            if r_norms[-1] < rtol:
                break
            x = x_new

    elif method == "block-gauss-seidel":
        num_blocks = len(b) // block_size
        for _ in range(max_iter):
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, len(b))
                
                A_block = A[start:end, start:end]
                b_block = b[start:end] - A[start:end, :] @ x + A_block @ x[start:end]
                
                if np.linalg.cond(A_block) < 1 / np.finfo(A_block.dtype).eps:
                    x[start:end] = np.linalg.solve(A_block, b_block)
            
            r = b - A @ x
            r_norms.append(np.linalg.norm(r) / np.linalg.norm(b))
            if r_norms[-1] < rtol:
                break
    return x, r_norms



def plot_convergence(errors_dict, i, max_iter, name="test"):
    plt.figure(figsize=(8, 5))
    for method, errors in errors_dict.items():
        plt.semilogy(errors[:max_iter], label=method.title(), linestyle='-', marker='')

    plt.xlabel("Iteration")
    plt.ylabel("Relative Residual ||r(k)|| / ||r(0)||")
    plt.title(f"Convergence of Iterative Methods (α = {alpha[i]:.1f}, β = {beta[i]:.1f})")
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}.png")

def run(alpha, beta, N, i, max_iter, name="test"):
    L = 1.0  
    h = L / (N + 1)
    A, b = create_system(N, alpha, beta)

    print(f"Cond: {np.linalg.cond(A)}")

    methods = ["jacobi", "gauss-seidel", "block-jacobi", "block-gauss-seidel"]
    solutions = {}
    errors = {}

    for method in methods:
        solutions[method], errors[method] = iterative_solver(A, b, method, block_size=5)
    errors["fom"] = fom(A, b)
    residuals = gmres(alpha, beta, N, max_iter)
    errors["gmres"] = residuals / residuals[0]
    if alpha == beta:
        residuals = cg(alpha, beta, N, max_iter)
        errors["cg"] = residuals / residuals[0]

    plot_convergence(errors, i, max_iter, name)

if __name__ == "__main__":
    N = 50
    max_iter = 400
    alpha = [0.0, 1.0, 1.0, 3.0]
    beta = [0.0, 0.4, 1.0, 3.0]

    # run([1.0], [0.4], N, 0, max_iter)
    for i in range(len(alpha)):
        run(alpha[i], beta[i], N, i, max_iter, name=f"res{i}")
        print(f"finished {i} iteration")
