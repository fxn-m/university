import inspect

import numpy as np

import numdifftools as nd
import matplotlib.pyplot as plt

# function to be initialised as specified by the assignment
initial_func = lambda X: (X[0] - 1) ** 2 + (X[1] - 2) ** 2 + (X[2] - 3) ** 2
# more functions to test the performance of the optimiser
rosen = lambda X: (1 - X[0]) ** 2 + 10 * (X[1] - X[0] ** 2) ** 2
func_0 = lambda X: 3 / 2 * X[0] ** 2 + 3 * X[1] ** 2 + 2 * X[0] * X[1] - 2 * X[0] + 8 * X[1] + 10
func_1 = lambda X: X[0] ** 2 + X[1] ** 2 - 10

# define the function to optimise by commenting out the alternatives below
func = initial_func
# func = rosen
# func = func_0
# func = func_1

# variables to be initialised at 0.5
x1, x2, x3 = 0.5, 0.5, 0.5

if func == initial_func:
    X = [x1, x2, x3]
else:
    X = [x1, x2]

# set parameters for optimisation
# define maximum number of CG iterations
j_max = 1000
# define CG error tolerance
eps = 0.01
# define maximum number of line search iterations
i_max = 100
# define line search error tolerance
eta = 0.1

with open('log.txt', "w") as log_file:
    log_file.write('NON-LINEAR PROGRAMMING LIBRARY\nFelix Newport-Mangell\nlog file\n\n')
    log_file.write('Function to optimise: ' + ('f(x) = {}').format(inspect.getsource(func)))


def plot_func(func):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make Data
    X_1 = np.arange(-5, 5, .025)
    X_2 = np.arange(-5, 5, .025)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    Z = func([X_1, X_2])

    # Plot the surface.
    surf = ax.plot_surface(X_1, X_2, Z, cmap='viridis',
                           linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_zlim(-20, 100)

    return plt

# visualise the function in 3D if the number of variables = 2
if len(X) == 2:
    plt = plot_func(func=func)
    plt.savefig('surf.png')


def contour_plot(func):
    X_1 = np.arange(-15, 15, .025)
    X_2 = np.arange(-15, 15, .025)
    X_1, X_2 = np.meshgrid(X_1, X_2)
    Z = func([X_1, X_2])

    # plt.style.use('_mpl-gallery-nogrid')
    levels = np.linspace(Z.min(), Z.max(), 15)
    # plot
    fig, ax = plt.subplots()

    ax.contourf(X_1, X_2, Z, levels=levels)

    return plt


# visualise the function as a contour plot if the number of variables = 2
if len(X) == 2:
    plt = contour_plot(func=func)
    plt.savefig('contour.png')

def alpha_GS(func, X, d, eta, i_max):
    alpha_U = 10
    alpha_L = 0
    alpha_1 = ''
    alpha_2 = ''

    # define golden ratio phi
    phi = (np.sqrt(5) - 1) / 2

    i = 0
    e = 1

    with open('log.txt', "a") as log_file:
        log_file.write('\n\nStart Golden Search routine\n-----------------------------------------')

    # loop until convergence (error < eta)
    while e >= eta and i < i_max:

        # evaluate function at bounds
        fL = func(X + alpha_L * d)
        fU = func(X + alpha_U * d)

        # determine step interval D
        D = phi * (alpha_U - alpha_L)

        # define interior points alpha_1 and alpha_2
        alpha_1 = alpha_U - D
        alpha_2 = alpha_L + D

        # determine the position of the min and update bounding brackets
        f1 = func(X + alpha_1 * d)
        f2 = func(X + alpha_2 * d)

        if f1 < f2:
            alpha_U = alpha_2
            alpha_2 = alpha_1
            alpha_1 = alpha_U - phi * (alpha_U - alpha_L)
            fU = f2
            f2 = f1
            f1 = func(X + alpha_1 * d)  # calculate new interior point

            e = 2 * abs((alpha_U - alpha_L) / ((alpha_U + alpha_L)))
            i += 1  # update iteration count

            continue

        if f2 < f1:
            alpha_L = alpha_1
            alpha_1 = alpha_2
            alpha_2 = alpha_L + phi * (alpha_U - alpha_L)
            fL = f1
            f1 = f2
            f2 = func(X + alpha_2 * d)  # calculate new interior point

            e = 2 * abs((alpha_U - alpha_L) / ((alpha_U + alpha_L)))
            i += 1  # update iteration count

    alpha = (alpha_U + alpha_L) / 2

    print('\n\nAlpha at end of line search:', alpha)

    with open('log.txt', "a") as log_file:
        log_file.write('\n\nAlpha at end of line search : ' + str(alpha))

    return alpha


def beta_HS(gs, j):
    beta = np.inner(gs[j], (gs[j] - gs[j - 1])) / np.inner(ds[j - 1], gs[j] - gs[j - 1])
    return beta


def beta_PR(gs, j):
    beta = np.inner(gs[j], (gs[j] - gs[j - 1])) / np.inner(gs[j - 1], gs[j - 1])
    return beta


def beta_FR(gs, j):
    beta = np.inner(gs[j], gs[j]) / np.inner(gs[j - 1], gs[j - 1])
    return beta


def cg(func, X, eps, j_max):
    Xs = [np.array(X)]
    gs = []
    ds = []
    alphas = []
    betas = []

    delta = eps + 1
    j = 0

    while abs(delta) > eps and j < j_max:

        # accessing the appropriate X
        X = Xs[j]
        # calculate gradient g_j of function f(X)
        g = nd.Gradient(func)(X)
        # append g_j to list in index j+1
        gs.append(g)

        if j == 0:
            # for first iteration, direction d_j is just inverse of gradient for minimising a function
            d = -g

        else:
            beta = beta_FR(gs, j)
            betas.append(beta)
            d = -gs[j] + beta * ds[j - 1]

        ds.append(d)

        # update alpha
        alpha = alpha_GS(func, X, d, eta, i_max)

        # append alpha_j to list
        alphas.append(alpha)

        # update X
        X = X + alpha * d
        # append X to list in index j+1, ready for next iteration
        Xs.append(X)

        print('\n\nAFTER CG ALGORITHM ITERATION', j + 1, ': X =', X, '\n\n')

        j += 1

        delta = func(Xs[j - 1]) - func(Xs[j])

        with open('log.txt', "a") as log_file:
            log_file.write('\n\nAfter Conjugate Gradient algorithm iteration : ' + str(j) + '\nX = ' + str(X))
            log_file.write('\nDelta = ' + str(delta))

        if abs(delta) <= eps:
            with open('log.txt', "a") as log_file:
                log_file.write('\n\nStopping criterion: Delta < epsilon (=' + str(eps) + ')')
                log_file.write('\nDelta = ' + str(delta) + ' < epsilon')
                log_file.write('\n\nTerminate algorithm \n\nX = ' + str(X))
                log_file.write('\n\nAt minimum:\nf(X) = ' + str(func(X)))

    return Xs, gs, ds, alphas, betas, j, delta


Xs, gs, ds, alphas, betas, j, delta = cg(func, X, eps, j_max)

X = Xs[-1]

with open('solution.txt', "w") as solution_file:
    solution_file.write('NON-LINEAR PROGRAMMING LIBRARY\nFelix Newport-Mangell\nsolution file\n\n')
    solution_file.write('Function to optimise: ' + ('f(x) = {}').format(inspect.getsource(func)))
    solution_file.write('\n\nConjugate Gradient descent calculated optimum X = ' + str(X))
    solution_file.write('\n X1 = ' + str(X[0]))
    solution_file.write('\n X2 = ' + str(X[1]))
    if len(X) == 3:
        solution_file.write('\n X3 = ' + str(X[2]))
    else:
        pass
    solution_file.write('\n\nAt minimum:\nf(X) = ' + str(func(X)))
