LINEAR PROGRAMMING LIBRARY

Running linprog.py
___________________

!!! Before running linprog.py, please ensure that you have the 'numpy' and 'pandas' python libraries installed on your machine. !!!

To run linprog.py, please navigate to the directory that the program is located, and open a terminal in the directory. From there run linprog.py in the method consistent with your OS. 
Running the program will output text to the terminal, but the two .txt files 'log.txt' and 'result.txt' contain all relevant information in a more interpretable form.



Format of input data
____________________

linprog.py takes .csv's as input, and outputs .txt files.

To change the linear programming problem, please update the fields of the .csv in accordance with the formatting described here. 
The .csv is structured with seven parameters: 

- obj (the objective of the problem, takes values 'max' or 'min'. Please enter without apostrophes)

- no. variables (the number of target variables {x1, x2, ..., xn} to optimise for. Please enter an integer)

- obj coeffs (the coefficients of the target variables in the objective function. Please enter floating point numbers separated by commas, e.g.: 60.0,90.0,300.0)

- constraints (the number of constraint equations that the problem is subject to. Please enter an integer)

- constraint coeffs (the coefficients of the variables in the constraint relationships. Please enter collections of floating point numbers separated by semi-colons. 
		    ...Please separate the collections from one another using commas, e.g.: 1;1;1,1;3;0,2;0;1 , for 3 relationships in a problem with 3 target variables {x1, x2, x3}.
	            ...If a target variable {x1, x2, ..., xn} does not appear in a given constraint relationship, insert a 0 in the index where it would be positioned.)

- constraint relationships (the type of relationship that the constraint obeys. Please input entries according to the following scheme without apostrophes: 
		    ...'equals to' as 'eq', 'greater than or equal to' as 'meq', 'less than or equal to' as 'leq')

- constraint bounds (the bounds of the constraint. Please enter floating point numbers separated by commas, e.g.: 600,600,900)


The easiest way to ensure that the input is of the correct format is to follow the structure that the file 'LPP.csv' takes when initialised. 
The structure is clearest to see when using Microsoft Excel to view the .csv, but other text editors work too. 




Interpreting log.txt and result.txt
__________________________________

linprog.py solves linear programming problems using the Simplex method, and outputs the state of the Tableau at various stages throughout the pivoting iterations. 
The log.txt files and result.txt files are automatically generated with annotations to aid interpretability.



Library capabilities
____________________

This library is capable of solving linear programming problems with both maximisation and minimisation objectives; equality, greater than/equal to, less than/equal to relationships; artificial variables,
and is capable of identifying instances of multiple optima, but cannot list the full scope of equivalently optimal solutions.



Research problem
________________

The chosen research problem to apply the library to is an imaginary EV manufacturer that wishes to make a decision on how to allocate resources between the production of 4 vehicle models at 3 factories. 
The main objective is profit, however the strategy of the company is to favour the production of their most recent model. This consideration is implemented by weighting the profit of the model in question, 
and the weight is discounted upon finding a solution to find the real profit of the strategy. 
The constraints include meeting minimum forecast demand of each model, and production constraints that each facility is under.






