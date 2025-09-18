"""Solve the 1D heat equation using finite differences.

I'm using backward Euler in time and central differences in space.
"""
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def initial_condition(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Initial condition."""
    return np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2))

def source_term(x: int | float | np.ndarray, t: int | float | np.ndarray) -> int | float | np.ndarray:
    """Source term."""
    return 0

def main():

    # Parameters and constants
    alpha = 1
    L = 1
    t_final = 0.1
    n_nodos_t = 100
    n_nodos_x = 300
    # non-homogeneous dirichelt boundary conditions
    T_0 = 0
    T_L = 0

    delta_x = L / (n_nodos_x - 1)
    delta_t = t_final / (n_nodos_t - 1)
    # fourier number
    F = alpha * delta_t / (delta_x ** 2)
    x = np.linspace(0, L, n_nodos_x)
    t = np.linspace(0, t_final, n_nodos_t)


    # create array to keep track of temperature at each node
    T = np.zeros((n_nodos_t, n_nodos_x))
    T[0, :] = initial_condition(x)

    # Create matrix A with equations, including boundary nodes
    A = np.zeros((n_nodos_x, n_nodos_x))

    # add equations for internal nodes
    for i in range(1, n_nodos_x - 1):
        A[i, i - 1] = -F
        A[i, i] = 1 + 2 * F
        A[i, i + 1] = -F

    # add equations for boundary nodes
    A[0, 0] = 1
    A[-1, -1] = 1

    # create vector b with known terms, including boundary nodes
    for n in range(1, n_nodos_t):
        # add known terms for internal nodes
        b = np.zeros(n_nodos_x)
        for i in range(1, n_nodos_x - 1):
            b[i] = T[n-1, i] + delta_t * source_term(x[i], t[n])
        # add known terms for boundary nodes
        b[0] = T_0
        b[-1] = T_L

        # solve system of equations
        T[n, :] = np.linalg.solve(A, b)
        print(f"Time step {n}: t = {t[n]:.4f} s, T = {T[n, :]}")

    # plot results
    fig, ax = plt.subplots()
    line, = ax.plot(x, T[0, :], 'r-', lw=2)
    ax.set_xlim(0, L)
    ax.set_ylim(np.min(T), np.max(T))
    ax.set_xlabel('x')
    ax.set_ylabel('Temperature')
    ax.set_title('1D Heat Equation Evolution')

    def update(frame):
        line.set_ydata(T[frame, :])
        ax.set_title(f"1D Heat Equation Evolution\nTime: {t[frame]:.3f} s")
        return line,

    ani = FuncAnimation(fig, update, frames=n_nodos_t, interval=800, blit=True)
    plt.show()

if __name__ == "__main__":
    main()