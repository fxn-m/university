# Optimisation Theory and Applications in Python 🐍

💡 **Idea**:
A suite of Python libraries designed to provide efficient solutions for various optimization problems, including linear programming (LP), non-linear programming (NLP), and dynamic programming (DP).

🛠️ **Approach**:

1. Linear Programming (LP) Library (linprog.py):
- Uses the Simplex method to solve LP problems.
- Inputs via a structured CSV file format and outputs results to text files.
- Allows optimization for maximization or minimization objectives.
- Handles constraints such as equality, and inequality.

2. Non-Linear Programming (NLP) Library (cg.py):
- Implements the Conjugate Gradient Descent (CGD) algorithm.
- Initialized to solve a specific function, but adaptable for different optimization problems.
- Employs Golden Line Search for the alpha parameter and Fletcher-Reeves form for the beta parameter.
  
3. Dynamic Programming (DP) Library:
- `network.py`: Solves network optimization problems based on the node-stage-cost structure.
- `capbud.py`: Tackles capital budgeting problems, inputting subsidiary plan combinations and allocating the parent company's budget.
- `initialise_problems.py`: Pre-configures example problems from reference notes.

✅ **Result**:
1. **LP Library**: Successfully solved a research problem involving an EV manufacturer's production allocation among different models and factories. Outputs include the state of the tableau and a more interpretable result.txt.
2. **NLP Library**: Addressed a linear regression problem predicting house prices based on property size. Demonstrates equivalent performance to state-of-the-art regression models by optimizing the mean squared error cost function.
3. **DP Library**: Offers optimized solutions for both network and capital budgeting problems. Capable of handling multiple optimum routes in network optimization and provides optimal plan allocations for various budget scenarios in capital budgeting problems.