DYNAMIC PROGRAMMING LIBRARY

network.py
___________________

This program solves network problems of the form described in chapter 5 of the notes.

The input to the program, inputnetwork.txt is structured as follows.
Each row of the input .txt file represents a node and its parameters: stage, state, UP cost, DOWN cost in the following scheme - 'stage'.'state':['UP cost', 'DOWN cost']
The format for an input problem must be identical to the scheme (described above) that the .txt file is initialised with for the program to work.

Interpreting the results
The outputs of the program, lognetwork.txt and solutionnetwork.txt, contain the following information:
The log file recaps the network to find the optimal route through, and prints out the results of backward recursion at each stage, starting from the penultimate stage.
The solution file summarises the optimal route(s) through the network and their cost, starting from the root node and indicating which node to traverse to next with '>' symbols.


This program is extensible to directional networks with any number of stages where nodes each have only UP and DOWN costs to the next nodes in the network,
and can handle multiple optimum routes through the network.



capbud.py
___________________

Note: There is an error in the capital budgeting problem solution on the notes
Looking at stage 2 with an x of 7, the table suggests that the best plan combination is Stage 2: Plan 1, Return 11. 
The correct solution is instead Stage 2: Plan 2, Return 11. 


This program solves capital budgeting problems of the form described in chapter 5 of the notes.

The input to the program, inputcapbud.txt is structured as follows.
Each row of the input .txt file (other than the final row) represents a subsidiary/plan combination with the format:
SubsidiaryPlan,Cost,Return

using a random example, subsidiary 3's 2nd plan, with a cost of 7 and a return of 9 the input row would be structured as:
32,7,9

The final row is used to input the parent company's overall budget to allocate.
The format for an input problem must be identical to the scheme (described above) that the .txt file is initialised with for the program to work.

Interpreting the results
The outputs of the program, logcapbud.txt and solutioncapbud.txt, contain the following information:
The log file recaps the capital budgeting problem to optimise, and prints out the results of forward recursion at each stage, starting from the first stage.
Each iteration through a stage finds the best plan for a given budget allocation 'x', the 'Recursion Table' at the bottom of logcapbud.txt summarises the results and is structured:
x: [best plan (stage 1), best plan return (stage 1), ... best plan (stage n), best plan return (stage n)] -- identically to Table 6 in chapter 5 of the notes.
The solution file summarises the optimal plan to take at each stage, gives the total return of the combination of plans and calculates the remaining budget.


This program is extensible to capital budgeting problems with any number of stages (subsidiaries) and states (plans). 



initialise_problems.py
________________________

Running this will write the example problems in the notes to the input .txt files 
