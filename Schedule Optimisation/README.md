# schedule-optimisation 🏭
Solves a mixed integer linear programming problem describing the optimal schedule for a production system

See the full report <a href="https://github.com/fxn-m/schedule-optimisation/blob/main/MENGM0056_Product_And_Production_Systems_G109.pdf">here</a>

## Project Summary:
- SU = System Utilisation
- WIP = Work in Process

The system describes a set of station operators whose working cycles are divided into sequentially executed time steps, each day of a 250 working day cycle. Each time step is an uninterruptible sub-task of station operation and also a basic scheduling unit.

An optimal schedule produces as many final products  as possible whilst minimising overproduction of all other parts. With the objective function formulation , if all constraints are met, the optimal solution will tend to assign workers to be idle, instead of assigning workers to jobs that produce no value. This simplifies the calculation of SU and WIP.

In this model, each station is operated throughout the time step by the same operator – such that ‘operator’ and ‘station’ are analogous. 

The average SU can be improved by organising fewer workers to carry out the most valuable tasks at the time they are required – the point of demand. The effect of a demand constraint combined with an encoding of the dependency between components and assemblies in the model creates an artificial, internal demand for each component. This ensures that each activity carried out is necessary to meet the product demand – forcing production of useful components.

Equation 5 adds a constraint that forces a down-time  to be observed after every change between jobs $j$ and $j'$, where $j'$ is any job that can be carried out on the same machine as job $j$, other than $j$.

##### *Objective Function*

$$
\max_{d,\ a,\ m,\ l} 
5*\sum_{j\in J^a}\sum_{o\in O^a}\sum_{t} a_{j,o,t} - \sum_{j\in J^d}\sum_{o\in O^d}\sum_{t} d_{j,o,t} + \sum_{j\in J^m}\sum_{o\in O^m}\sum_{t} m_{j,o,t} +
\sum_{j\in J^l}\sum_{o\in O^l}\sum_{t} l_{j,o,t}
$$

##### *subject to:*

1. For any operator, at most one job can be in progress at a time t.

$$\sum_{j\in J_s} W_{j,o,t}^s≤1, \ \ ∀ s∈S,o∈O^s,t  $$

2. Daily demand must be met by the assemblies in inventory
The sum of all products made must be greater or equal to the sum of the demand for all t

$$
\sum_{t}\sum_{o\in O_a}{a_{j,o,t}\ast P_j}-\sum_{t}\mathrm{\Psi}_t\geq0,\ \ \forall\ t,\ j=J_{\bar{\omega}},\ \bar{\omega}=MA
$$

3. Job hierarchy and precedence

$$ 
\sum_{t}{\sum_{s\in S}\sum_{{o\in O}_a}{W_{j,o,t}^s\ast P_j}\ \ -\ \sum_{t}\sum_{{o\in O}_a}{a_{\bar{\omega},o,t}\ast P_{\bar{\omega}}}\ \ \geq0,\ \ \forall\ \omega\in\ \mathrm{\Omega},\ j\in\omega,\ t}
$$

4. If a job starts in a station at a time interval , no other job can start in the same station until after this job is finished and an additional downtime period 

$$ {\tau_{jj^\prime}\ast\ W_{j^\prime,\ o,\ t+1}^s}_\ \le M\left(1-W_{j,o,t}^s\right),\ \ \forall\ o\in O^s,\ j\in J^s,j^\prime\in J^s,\ t\  $$

5. Decision variables are binary
$$W_(j,o,t)^s∈{0,1},   ∀ j,o,t,s $$
