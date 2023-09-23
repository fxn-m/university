import math

with open('inputcapbud.txt', 'r') as input_txt:
    lines = input_txt.read().splitlines()


# function to turn strings read in from the text file into lists
def lspl(line):
    return line.strip().split(',')


data = {}
stages = []

for line in lines:
    if line[0].isnumeric():
        company_plan = lspl(line)[0]
        stage = line[0]
        stages.append(stage)
        list_of_str = lspl(line)[1:]
        list_of_int = list(map(int, list_of_str))
        data[company_plan] = list_of_int
        stages.append(stage)
    else:
        budget = int(line.replace(' ', '').split(':')[1])

stages = len(set(stages))

print('company data\nSubsidiary/Plan : [Cost, Return]\n')
for k in data:
    print(k, ':', data[k])

print('\ncapital budget:', budget)


# find the lowest cost of a plan from a given stage
def min_stage_cost(data, stage):
    min_cost = math.inf
    for k in data:
        if int(k[0]) == stage:
            cost = data[k][0]
            if cost < min_cost:
                min_cost = cost
    return min_cost


def stage_data(data, stage):
    stagedata = {}
    for k in data:
        if int(k[0]) == stage:
            stagedata[k] = data[k]
    return stagedata


print(stages)
all_staged_data = {'Stage {}'.format(i): stage_data(data, i) for i in range(1, stages + 1)}
for stage in all_staged_data:
    print(stage, all_staged_data[stage])

# initialise the log file
with open('logcapbud.txt', "w") as log_file:
    log_file.write('CAPBUD.PY\nFelix Newport-Mangell\nlog file\n\n')
    log_file.write("Capital Budgeting problem to optimise: \n\nStage n : {'SubsidiaryPlan': [Cost, Return]}")
    for k in all_staged_data:
        log_file.write('\n ' + k + ' : ' + str(all_staged_data[k]))
    log_file.write('\n____________________________________________\n')
# initialise table to store results of each sub-problem
recursion_table = {}

# create list of minimum costs to call during stage loop - used to initialise budget loop
min_costs = []

# loop through all stages, starting at stage 1
for stage in range(1, stages + 1):
    print('\nIteration through Stage', stage)
    with open('logcapbud.txt', "a") as log_file:
        log_file.write('\n--- ITERATION THROUGH STAGE ' + str(stage) + ' ---')

    if stage == stages:
        min_costs.append(budget)
    elif stage > 1:
        min_costs.append(min_stage_cost(data, stage) + min_costs[-1])
    else:
        min_costs.append(min_stage_cost(data, stage))

    if stage > 1:
        for x in range(min_costs[0], min_costs[stage - 1]):
            recursion_table[x].append(None)
            recursion_table[x].append(None)

    # budget allocation loop
    for x in range(min_costs[stage - 1], budget + 1):
        print("This iteration's budget:", x)
        with open('logcapbud.txt', "a") as log_file:
            log_file.write("\nThis iteration's budget: " + str(x))
        best_return = 0

        # plan loop
        for plan in all_staged_data['Stage {}'.format(stage)]:
            plan_cost = data[plan][0]
            diff = x - plan_cost

            if stage == 1:
                plan_return = data[plan][1]

            if stage > 1:
                # this conditional will skip considering the return of the plan if the cost is not feasible
                # with the remaining budget
                if diff < min(list(recursion_table.keys())) or (recursion_table[diff][1 + 2 * (stage - 2)] is None):
                    print('Stage {}'.format(stage), 'plan', plan, 'plan cost',
                          plan_cost, 'available', x, '...  skipping...plan cost exceeds available funds')
                    continue

                plan_return = data[plan][1] + recursion_table[diff][1 + 2 * (stage - 2)]  # THIS LINE

            # if the plan is feasible and the return is greater than what's already proposed, select that plan
            if (diff >= 0) and (plan_return > best_return):
                best_return = plan_return
                d = plan[1]

        S = best_return
        d = int(d)
        if stage == 1:
            recursion_table[x] = [d, S]
        else:
            recursion_table[x].append(d)
            recursion_table[x].append(S)

        with open('logcapbud.txt', "a") as log_file:
            log_file.write("\n['plan', 'best return'] = " + str([d, S]))
            log_file.write('\n')


print('\n\nRecursion table:\n')
with open('logcapbud.txt', "a") as log_file:
    log_file.write('\n____________________________________________\n')
    log_file.write('\n\nRecursion Table:\n')
    log_file.write('\nx: [best plan (stage 1), best plan return (stage 1), ... best plan (stage n), best plan return (stage n)]')
for row in recursion_table:
    with open('logcapbud.txt', "a") as log_file:
        log_file.write('\n' + str(row) + ':  ' + str(recursion_table[row]))
    print(row, recursion_table[row])

# initialise this variable used in the calculation described above
remaining_budget = budget

with open("solutioncapbud.txt", "w") as solution_file:
    solution_file.write('CAPBUD.PY\nFelix Newport-Mangell\nsolution file\n\n')
    solution_file.write("Capital Budgeting problem to optimise: \n\nStage n : {'SubsidiaryPlan': [Cost, Return]}")
    for k in all_staged_data:
        solution_file.write('\n ' + k + ' : ' + str(all_staged_data[k]))
    solution_file.write('\n____________________________________________\n')

print('\n\nOptimal Solution')
# loop through the stages in reverse
for i, k in enumerate(reversed(all_staged_data)):
    # using the remaining budget as an index to the recursion table,
    row = recursion_table[remaining_budget]
    plan = row[len(row) - ((i + 1) * 2)]
    print('Stage', stages - i, ': plan', plan)

    with open("solutioncapbud.txt", "a") as solution_file:
        solution_file.write('\n\nStage ' + str(stages - i) + ' : plan ' + str(plan))

    index = list(all_staged_data[k].keys())[plan - 1]

    cost = all_staged_data[k][index][0]
    remaining_budget = remaining_budget - cost

print('\nFor a return of', recursion_table[budget][-1], 'with a budget of', remaining_budget, 'remaining')
with open("solutioncapbud.txt", "a") as solution_file:
    solution_file.write('\n\n\nFor a return of ' + str(recursion_table[budget][-1]) + ' with a budget of ' + str(remaining_budget) + ' remaining')
