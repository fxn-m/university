NON-LINEAR PROGRAMMING LIBRARY

Running cg.py
___________________

!!! Before running cg.py, please ensure that you have the dependent libraries installed on your machine. !!!

Running the program will output text to the terminal, but the .txt file 'log.txt' contains all relevant information in a more interpretable form.


Input functions
____________________

The program is initialised to solve the function described in optass2.pdf but the code can be edited to optimise for different functions.

The function selection can be found starting on line 16 of cg.py. Please comment out the function that you wish to minimise, or edit the active function as you wish. 
Functions with two target variables will call a function to plot the function in 3D and a contour plot. These will save themselves as .png files in the program directory. 
The rosenbrock-like function does not like being plotted in 3D for some reason.

The input variables x1, x2, x3 will be initialised at 0.5 as specified by optass2.pdf.


Sub-routines
____________________

This program contains two main sub-routines to calculate the parameters used in conjugate gradient descent

alpha parameter: Golden Line Search. function = alpha_GS()

beta parameter: Fletcher-Reeves form. function = beta_FR. 

the beta parameter can be chosen to be Hestenes-Stiefel form (beta_HS) or Polak-Riviere form (beta_PR), by editing the function call within the function 'cg'
but the Fletcher-Reeves form gives the best performance on functions tested as so is initialised in the function 'cg'.


Parameters and their initial values (line 29 of cg.py)
_____________________________________

This program is initialised with a number of parameters used in the conjugate gradient and golden line search algorithms. Below are their descriptions, and their initial values.

j_max defines maximum number of CG iterations
j_max = 1000

eps defines CG error tolerance
eps = 0.01

i_max defines maximum number of line search iterations
i_max = 100

eta defines line search error tolerance
eta = 0.1


Interpreting log.txt and solution.txt
______________________________________

log.txt returns the updated point coordinates at the end of each cg algorithm step. It also displays the result of the program on termination, namely the optimum values of 
the target variables x1, x2, ..., xn, and the value of the function at this point.

solution.txt summarises the result of the program output in the form requested in optass2.pdf


Research problem
_________________

This research project’s focus is to implement the Conjugate Gradient Descent (CGD) algorithm developed in part A of the assignment in a univariate Linear Regression problem. 
The challenge is to create a model to predict house prices given the square metreage of the property. The mean squared error cost function is optimised and calculates the same 
optimal values of the model weights as a state of the art regression library module.



