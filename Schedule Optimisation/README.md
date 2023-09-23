# Schedule Optimisation 🏭

♻ **Idea**:
The core concept is to determine the optimal schedule for a production system. Using mixed integer linear programming, it strategically allocates station operators' tasks in a 250-day working cycle, aiming to maximize final product output while avoiding wasteful overproduction.

🛠️ **Approach**:
Key Performance Indicators to be optimized:
- SU = System Utilisation
- WIP = Work in Process
  
**Model Structure**:
- Each day is divided into time steps, with each time step representing an uninterruptible sub-task.
- The model emphasizes efficiency. If all constraints are met, it prioritizes keeping workers idle over assigning non-value producing tasks.
- In the system, there's an equivalence between an 'operator' and a 'station'.
- Average SU is elevated by aligning fewer workers to the most essential tasks when needed.
- A combination of demand constraint and dependencies between components ensures each component is necessary for the final product.

***Objective Function***

$$
\max_{d,\ a,\ m,\ l} 
5*\sum_{j\in J^a}\sum_{o\in O^a}\sum_{t} a_{j,o,t} - \sum_{j\in J^d}\sum_{o\in O^d}\sum_{t} d_{j,o,t} + \sum_{j\in J^m}\sum_{o\in O^m}\sum_{t} m_{j,o,t} +
\sum_{j\in J^l}\sum_{o\in O^l}\sum_{t} l_{j,o,t}
$$

***Subject to constraints:***

- Operator job limitations.
- Meeting daily product demand.
- Maintaining job hierarchy and precedence.
- Ensuring job downtime after job changes.
- Decision variables are strictly binary.

✅ **Result**:
The outcome of this model is an enhanced production schedule that meets product demand effectively. It reduces wasteful tasks, optimizes station operator's time, and adheres to set constraints.


📄 **Report**: The report can be found [here](https://drive.google.com/file/d/1jc1pwSu0KWL3z2xB4xRNDjy2_dmBP5Kq/view?pli=1). Pages 5-8 cover the formulation and implementation of MILP for scheduling.