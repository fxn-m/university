from pulp import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

## Read Data
# Jobs

def load_data():

    '''
    Read in the component database
    '''
    
    # read data in and reformat
    jobs_df = pd.read_csv(os.getcwd() + '/Data/step_1_data.csv')
    for col in jobs_df: 
        if type(jobs_df[col][0]) == str: 
            jobs_df[col] = jobs_df[col].str.lower()

    jobs_df.index = np.arange(1, len(jobs_df) + 1)
    jobs_df.index.rename('job', inplace=True)
    return jobs_df

def find_daily_demand(df, seed):

    '''Estimate Daily Demand'''

    # roughly 21 working days per month
    # already have monthly estimates 

    day = 0
    daily_demand = {}

    np.random.seed(seed)

    while day<250:
        
        month = day//21
        demand = round((df.loc[(month), 'Demand (units)']/21)/100, 1) * 100
        daily_demand[day] = demand*np.random.normal(loc=1, scale = 0.1, size=1)[0]
        day+=1

    return daily_demand

def sum_demands(daily_demand, time_step):

    summed_demands = []

    daily_demand = list(daily_demand.values())

    for i in range(int(len(daily_demand)/time_step)-1):
        summed_demands.append(sum(daily_demand[time_step*i:time_step*(i + 1)]))

    return summed_demands
    
# Implementation

def init_operators(n_dies, n_assemblers, n_mills, n_lathes):

    O = {
        "die": n_dies,
        "assembler": n_assemblers,
        "mill": n_mills,
        "lathe": n_lathes
    }

    return O

def init_jobs(jobs_df, S):
    
    def extract_jobs(jobs_df, station_type):
        return list(jobs_df[jobs_df['station_type']==station_type].index.values)
        
    J = dict([(station, extract_jobs(jobs_df, station)) for station in S])

    return J

def init_forecast(jobs_df, daily_demand, sum_demands, T, dt, production_seed, forecast_type, downtime, plot):
    
    '''
    forecast_type:
        1 = Squishing, fewer time-steps describe the same annual run. Demand is summed over time-steps, production rates are multiplied by the ratio.
        2 = No squishing, just run the optimisation on a shorter period.
    '''

    np.random.seed(production_seed)

    if forecast_type == 1:
        # productive rates
        productive_rate = dict((np.floor(8/jobs_df['hours_pp'] * jobs_df['count']*np.random.normal(loc=1, scale = 0.05, size=1)[0]).astype(int))*dt)
        # demand
        demand_forecast = sum_demands(daily_demand, time_step=dt)

    else:
        # productive rates
        productive_rate = dict(np.floor(8/jobs_df['hours_pp'] * jobs_df['count']*np.random.normal(loc=1, scale = 0.05, size=1)[0]).astype(int))
        # demand
        demand_forecast = list(daily_demand.values())

    for i in range(downtime):
        demand_forecast[i] = 0

    if plot == True:
        fig, ax = plt.subplots(figsize=(10,5))

        plt.plot(demand_forecast[:T])
        ax.set_title('Estimated Daily Demand (Units)', fontsize=18)
        ax.set_xlabel('Working Day', fontsize=14)
        ax.set_ylabel('Forecast Demand (units)', fontsize=14)
        ax.grid(visible='True', alpha=0.4, axis='y')
        plt.grid(visible=True, axis='y')
        plt.show()

    return productive_rate, demand_forecast

def init_assemblies(jobs_df):

    # set of assembly jobs and subcomponent jobs
    def foo(assembly):
        types = []
        for i in assembly:
            types.append(jobs_df.loc[i, 'station_type'])
        a_dict = dict(zip(assembly, types))
        return a_dict

    MA = foo([1, 7, 8, 9, 10, 11, 12, 14, 17, 18, 19, 20, 22, 23, 24])
    SA1 = foo([15, 16])
    SA2 = foo([13])
    SA3 = foo([2])
    # SA4 = foo([3, 4, 5])
    # SA5 = foo([6])

    omega = {
        21:MA,
        22:SA1,
        23:SA2,
        24:SA3,
        # 25:SA4,
        # 26:SA5    
    }

    return omega

# Define decision variables

def init_decision_variables(J, O, T):

    n_dies, n_assemblers, n_mills, n_lathes = O.values()

    d = LpVariable.dicts("d", ([j for j in range(len(J['die']))], [o for o in range(n_dies)], [o for o in range(T)]),0,1,LpInteger)
    a = LpVariable.dicts("a", ([j for j in range(len(J['assembler']))], [o for o in range(n_assemblers)], [o for o in range(T)]),0,1,LpInteger)
    m = LpVariable.dicts("m", ([j for j in range(len(J['mill']))], [o for o in range(n_mills)], [o for o in range(T)]),0,1,LpInteger)
    l = LpVariable.dicts("l", ([j for j in range(len(J['lathe']))], [o for o in range(n_lathes)], [o for o in range(T)]),0,1,LpInteger)

    W = {
        'die':d,
        'assembler':a,
        'mill':m,
        'lathe':l
    }

    return W 

# Objective function

def objective_function(prob, J, O, T, W):

    n_dies, n_assemblers, n_mills, n_lathes = O.values()
    d, a, m, l = list(W.values())

    prob += pulp.lpSum([
        [-1*d[jd][od][td] for jd in range(len(J['die'])) for od in range(n_dies) for td in range(T)],
        [-1*a[ja][oa][ta] for ja in range(len(J['assembler'])) for oa in range(n_assemblers) for ta in range(T)],
        [5*a[0][oa][ta] for oa in range(n_assemblers) for ta in range(T)],
        [-1*m[jm][om][tm] for jm in range(len(J['mill'])) for om in range(n_mills) for tm in range(T)],
        [-1*l[jl][ol][tl] for jl in range(len(J['lathe'])) for ol in range(n_lathes) for tl in range(T)]
    ])
    return prob

# Constraint 1 - Operators can do one job

def constraint_1(prob, J, O, T, S, W):
    for station_type in S:
        for o in range(O[station_type]):
            for t in range(T):
                prob += lpSum([W[station_type][j][o][t] for j in range(len(J[station_type]))]) <= 1

    return prob

# Constraint 2 - Daily demand must be met

def constraint_2(prob, J, O, T, a, productive_rate, demand_forecast):

    station_type = 'assembler'
    j0 = J[station_type][0]
    j1 = J[station_type][4]
    j2 = J[station_type][5]

    for t1 in range(T):
        for j in [j0]:
            prob += lpSum([a[(j%10)-1][o][t2] * productive_rate[j] for t2 in range(t1) for o in range(O[station_type])]) - lpSum([demand_forecast[t2] for t2 in range(t1)]) >= 0

    return prob

# Constraint 3 - Decision variables are binary
# Already managed by LpVariable 😀
    # Constraint 3.2 

def constraint_3(prob, J, O, T, S, W):
    for station_type in S:
        for j in range(len(J[station_type])):
            for t in range(T):
                prob += lpSum([W[station_type][j][o][t] for o in range(O[station_type])]) <= 1

    return prob

# Constraint 4 - Job hierarchy and precedence

def constraint_4(prob, J, O, T, W, a, omega, productive_rate):

    for t1 in range(T):
        for assembly in list(omega):
            assembly_index = (assembly%10)-1
            for subjob in omega[assembly]:
                s = omega[assembly][subjob]
                j = J[s].index(subjob)
                
                prob += lpSum([
                    [W[s][j][o][t2] * productive_rate[subjob] for t2 in range(t1) for o in range(O[s])],
                    [-1 * a[assembly_index][o][t2] * productive_rate[assembly] for t2 in range(t1) for o in range(O['assembler'])]
                ]) >= 0
    return prob

# Constraint 5 - Downtime 

def constraint_5(prob, J, O, T, d):

    # definition of tau
    no_die_jobs = len(J['die'])
    tau = np.ones((no_die_jobs, no_die_jobs))
    for i in range(no_die_jobs):
        tau[i][i] = 0

    M = 10
    for t in range(T-1):
            for j in range(no_die_jobs):
                for j_prime in range(no_die_jobs):
                        for o in range(O['die']):
                            prob += tau[j][j_prime] * d[j_prime][o][t+1] <= M * (1 - d[j][o][t])

    return prob

# Visualisations

def make_schedule_dataframe(prob, J, O, T):

    n_dies, n_assemblers, n_mills, n_lathes = O.values()

    columns = [t for t in range(T)]
    index = [f'd_{d}' for d in range(n_dies)] + [f'a_{a}' for a in range(n_assemblers)] + [f'm_{m}' for m in range(n_mills)] + [f'l_{l}' for l in range(n_lathes)]

    translate = {
        'd':'die',
        'a':'assembler',
        'm':'mill',
        'l':'lathe'
    }

    schedule_df = pd.DataFrame(columns=columns, index=index)

    for e in prob.variables():
        if e.varValue == 1:
            e = str(e).split('_')
            schedule_df.loc[e[0]+'_'+e[2], int(e[-1])] = J[translate[e[0]]][int(e[1])]

    schedule_df.fillna(99, inplace=True)

    return schedule_df

def worker_schedule_visualisation(schedule_df, jobs_df, saveplot=False):

    fig, ax = plt.subplots(figsize=(10,5))
    cmap = plt.cm.get_cmap('viridis')

    for worker in schedule_df.index:
        prev=0
        for day in schedule_df.columns:
            job = schedule_df.loc[worker, day]
            if job == 99:
                colour, alpha = 'red', 0.5
            else:
                colour, alpha = cmap(jobs_df.index[int(job)-1]/len(jobs_df)), 0.8
            plt.barh(worker, 1, left=prev, color=colour, height=0.5, alpha = alpha)
            prev+=1
        
    ax.set_xlabel('Working Day', fontsize=14)
    ax.set_ylabel('Worker', fontsize=14)
    ax.set_title("Annual Schedule", fontsize=18)
    ax.grid(visible=True, which='both', alpha=0.2, axis='x')

    plt.savefig('schedule_vis.png', dpi=500)

    plt.show()

def annual_inventory(schedule_df, jobs_df, productive_rate, T, demand_forecast):
    # jobs_df['daily_rate'] = np.floor(8/jobs_df['hours_pp'] * jobs_df['count']).astype(int)

    columns = [i for i in jobs_df.index]
    inventory_df = pd.DataFrame(columns=columns, index=[t for t in range(T)])
    inventory_df.iloc[0,:] = 2500

    # generate list of assembly job ids
    assembly_job_ids = ['ma'] + [f'sa{i+1}' for i in range(5)]
    # initialise the dictionary that will keep track of how much of each assembly has been made each day
    assembly_quantities = dict.fromkeys(assembly_job_ids)
    # define relationship between assemblies 
    assembly2job = dict([(i, jobs_df[jobs_df['component_name']==i].index.values[0]) for i in assembly_job_ids])

    for day in inventory_df.index:

        if day == 0: continue

        # carry over from previous day
        inventory_df.iloc[day] = inventory_df.iloc[day-1]

        for station in schedule_df.index:
            job = int(schedule_df.loc[station, day])
            if job == 0 : continue
            if job == 99: continue
            inventory_df.loc[day, job] = inventory_df.loc[day, job] + productive_rate[job]

        # the number of each assembly/subassembly made that day (dict)
        for assembly_job_id in assembly_job_ids:
            assembly_quantities[assembly_job_id] = inventory_df.loc[day, assembly2job[assembly_job_id]] - inventory_df.loc[day-1, assembly2job[assembly_job_id]]
            
        # subtract from each of the jobs that is consumed by each assembly (jobs_df['subassembly'])
        for assembly_job_id in assembly_job_ids:
            for consumed_component in jobs_df[jobs_df['subassembly'] == assembly_job_id].index:
                inventory_df.loc[day, consumed_component] = inventory_df.loc[day, consumed_component] - (assembly_quantities[assembly_job_id] * jobs_df.loc[consumed_component, 'count'])

        # take demand from main assembly
        # inventory_df.loc[day, 21] = inventory_df.loc[day, 21] - demand_forecast[day-1]

    return inventory_df

def plot_inventory(inventory_df, jobs_df, fail_type=False, saveplot=False):
    
    fig, ax = plt.subplots(figsize=(10,5))

    cmap = plt.cm.get_cmap('tab20')
    cmap_val_dict = {
        'sa1': 0,
        'sa2': 0.15,
        'sa3': 0.35,
        'sa4': 0.6,
        'sa5': 0.8,
        'ma': 0.9,
        }

    for job in inventory_df:
        
        # assign color 
        if jobs_df.loc[job, 'component_name'] == 'ma': key, alpha = jobs_df.loc[job, 'component_name'], 1
        elif jobs_df.loc[job, 'component_name'] in list(cmap_val_dict): key, alpha = jobs_df.loc[job, 'component_name'], 0.2
        else: key, alpha = jobs_df.loc[job, 'subassembly'], 0.2

        plt.plot(inventory_df[job], label=jobs_df.loc[job, 'component_name'], color=cmap(cmap_val_dict[key]), alpha=alpha)
            
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_title('Inventory', fontsize=18)
    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel('Working Day', fontsize=14)
    ax.set_ylabel('Units', fontsize=14)

    if fail_type == True: 
        plt.savefig('inventory_vis.png', dpi=500)

    plt.show()

def plot_sd(inventory_df, demand_forecast, T, saveplot=False):
    # Supply & Demand
    cumulative_summed_demands = []
    for i in range(T):
        cumulative_summed_demands.append(sum(demand_forecast[:i]))

    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(list(cumulative_summed_demands[0:T]), label='Demand')
    plt.plot(list(inventory_df[21][0:T]), label='Products') # this is cumulative and not scaled 

    ax.set_title('Cumulative Production Count and Demand', fontsize=18)
    ax.grid(visible=True, alpha=0.3)
    ax.set_xlabel('Working Day', fontsize=14)
    ax.set_ylabel('Units', fontsize=14)
    plt.legend()

    plt.savefig('demand_supply_vis.png', dpi=500)

    plt.show()

def calc_system_utilisation(schedule_df):
    schedule_df
    total_stations = schedule_df.shape[0]
    system_utilisation = []

    for col in schedule_df.columns:
        value_counts = schedule_df.loc[:, col].value_counts()
        if 99 not in value_counts.index:
            utilisation = 1
        else: 
            utilisation = (total_stations - value_counts[99])/total_stations
        system_utilisation.append(utilisation)

    return system_utilisation

def calc_work_in_progress(inventory_df):
    # for each day, calculate the sum of all items, other than the main assembly
    WIP = []
    for i in inventory_df.index:
        items = []
        for col in inventory_df.columns:
            if col == 21:
                continue
            items.append(inventory_df.loc[i, col])

        WIP.append(sum(items))
        # print(inventory_df.loc[i])
        for el in WIP:
            if el<0:
                WIP.remove(el)
    return WIP

def setup_and_solve(n_dies, n_assemblers, n_mills, n_lathes, demand_seed, production_seed, i, save=False, saveplot=False):
    
    jobs_df = load_data()

    demand_df = pd.read_csv('./Data/simple_demand_forecast.csv')
    daily_demand = find_daily_demand(demand_df, demand_seed)

    O = init_operators(n_dies, n_assemblers, n_mills, n_lathes)
    S = list(O)
    J = init_jobs(jobs_df, S)
    T = 250
    dt = int(250/T)
    

    productive_rate, demand_forecast = init_forecast(jobs_df, daily_demand, sum_demands, T, dt, production_seed, forecast_type=1, downtime=5, plot=True)
    omega = init_assemblies(jobs_df)

    W = init_decision_variables(J, O, T)
    d, a, m, l = W.values()

    # define the LP problem 
    prob = LpProblem("ScheduleOptimisation", LpMaximize)
    prob = objective_function(prob, J, O, T, W)

    prob = constraint_1(prob, J, O, T, S, W)
    prob = constraint_2(prob, J, O, T, a, productive_rate, demand_forecast)
    prob = constraint_4(prob, J, O, T, W, a, omega, productive_rate)
    # prob = constraint_5(prob, J, O, T, d)

    # Solve
    print(
        "Constraints:", prob.numConstraints(),
        "\nVariables:", prob.numVariables())

    prob.solve(GUROBI_CMD(msg = True, timeLimit=1800))
    # prob.solve(PULP_CBC_CMD(msg = True, timeLimit=1800))
    
    print(str(value(prob.objective)), LpStatus[prob.status])

    print('n_dies =', n_dies, '\nn_assebmlers =', n_assemblers,
      '\nn_mills =', n_mills, '\nn_lathes =', n_lathes)
    
    schedule_df = make_schedule_dataframe(prob, J, O, T)
    inventory_df = annual_inventory(schedule_df, jobs_df, productive_rate, T, demand_forecast)

    if prob.status == 1:
        print('SUCCESS')
        base_path = f'Optimisation_Results_6/{i}/'
        worker_schedule_visualisation(schedule_df, jobs_df, saveplot)
        plot_inventory(inventory_df, jobs_df, saveplot)
        plot_sd(inventory_df, demand_forecast, T, saveplot)
        
    else:
        print('FAIL')
        base_path = f'Optimisation_Results_6/{i}_fail/'

    if prob.status == 1 & save==True: 
        os.mkdir(base_path)
        schedule_df.to_csv(base_path + 'schedule.csv')

    return schedule_df, inventory_df
