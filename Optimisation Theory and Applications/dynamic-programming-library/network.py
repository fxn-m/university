import ast

# read in the text file that describes the network
with open('inputnetwork.txt', 'r') as input_txt:
    lines = input_txt.read().splitlines()


# function to help turn the strings read in from the text file into lists
def str2list(_input_):
    output = ast.literal_eval(_input_)
    return output


network = {}
for line in lines:
    network[line.split(':')[0]] = str2list(line.split(':')[1])


def best_next_step(network, node, pos):
    up_node = '{}.{}'.format(str(pos + 1), str(state + 1))
    down_node = '{}.{}'.format(str(pos + 1), str(state))
    min_cost = min(network[node])
    up_cost = network[node][0]
    down_cost = network[node][1]

    if min_cost == up_cost:
        d = 'UP'
        next_node = '{}.{}'.format(str(pos + 1), str(state + 1))
    elif min_cost == down_cost:
        d = 'DOWN'
        next_node = '{}.{}'.format(str(pos + 1), str(state))
    if up_cost == down_cost:
        d = 'UP or DOWN'
        next_node = str(up_node) + ' or ' + str(down_node)

    return min_cost, d, next_node


# find the number of stages in the network, this is used in conditional statements to decide how to calculate the cost
stages = 0
for k in network:
    s = int(k.split('.')[0])
    if s > stages:
        stages = s

max_stage = stages + 1

# initialise the log file
with open('lognetwork.txt', "w") as log_file:
    log_file.write('NETWORK.PY\nFelix Newport-Mangell\nlog file\n\n')
    log_file.write('Network to optimise: \n\nStage.State : [UP cost, DOWN cost]')
    for k in network:
        log_file.write('\n ' + k + ' : ' + str(network[k]))
    log_file.write('\n____________________________________________\n')

# the summary of the best route to take from each stage/state combination
best_routes = {}

# iterate through the stages
for i in range(0, max_stage):
    pos = max_stage - (i + 1)  # integer to keep track of the stage position
    stage = '{}.'.format(
        max_stage - (i + 1))  # create a string that can be used to check against keys in the network dictionary
    print('Stage:', pos)

    with open('lognetwork.txt', "a") as log_file:
        log_file.write('\n\nStage: ' + str(pos))

    # iterate through all nodes in the network
    for node in network:

        # look at only the nodes belonging to the stage that we are iterating through in the outer 'for' loop
        if stage in node:
            state = int(node.split('.')[1])

            # if it's the penultimate stage, only look at the next step costs to determine the best route
            if pos == max_stage - 1:
                min_cost, d, next_node = best_next_step(network, node, pos)
                best_routes[node] = [min_cost, next_node]

            # otherwise, look at the sum of the next steps and the best route from the next step options
            else:
                # find the available next steps
                up_node = '{}.{}'.format(str(pos + 1), str(state + 1))
                down_node = '{}.{}'.format(str(pos + 1), str(state))
                # find the available next steps' optimal route values and sum with the next step cost
                up_cost = network[node][0] + best_routes[up_node][0]
                down_cost = network[node][1] + best_routes[down_node][0]

                min_cost = min(up_cost, down_cost)

                if min_cost == up_cost:
                    d = 'UP'
                    next_node = up_node

                elif min_cost == down_cost:
                    d = 'DOWN'
                    next_node = down_node

                if up_cost == down_cost:
                    d = 'UP or DOWN'
                    next_node = str(up_node) + ' or ' + str(down_node)

                best_routes[node] = [min_cost, next_node]

            print('current node', node, '- min cost:', str(min_cost), '- direction', d, '- next node', next_node)
            with open('lognetwork.txt', "a") as log_file:
                log_file.write('\ncurrent node ' + str(node) + ' - min cost: ' + str(min_cost) + ' - direction ' + str(
                    d) + ' - next node ' + str(next_node))


def find_routes(routes, best_routes):
    """
    Input a data structure representing the best next step at each node (dict: best_routes)
    Output the best route(s) through the network

    1. Create a list of potential routes
    2. Starting at node 0, look at the best routes for that node and take the next step(s)
        If there are multiple routes, add a new string in the list for the split
    3. Loop through the list of routes until the stage of the node being checked is 1 less than the terminal stage
    """

    for i, route in enumerate(routes):
        last_node = route[-3:]

        if last_node in best_routes:
            next_step_options = best_routes[last_node][1].split(' or ')
            if len(next_step_options) > 1:
                routes.append(route + ' > ' + next_step_options[1])
                routes[i] = route + ' > ' + next_step_options[0]
            else:
                routes[i] = route + ' > ' + next_step_options[0]

            routes = find_routes(routes, best_routes)

    return routes


routes = ['0.1']

routes = find_routes(routes, best_routes)

with open('solutionnetwork.txt', "w") as solution_file:
    solution_file.write('NETWORK.PY\nFelix Newport-Mangell\nsolution file\n\n')
    solution_file.write('Network to optimise: \n\nStage.State : [UP cost, DOWN cost]')
    for k in network:
        solution_file.write('\n ' + k + ' : ' + str(network[k]))
    solution_file.write('\n____________________________________________\n')
    solution_file.write('\n\n\nOptimal route(s) through the network:\n')

for i, route in enumerate(routes):
    print(('Optimal route {}: ' + route).format(i))
    with open('solutionnetwork.txt', "a") as solution_file:
        solution_file.write(('\nOptimal route {}: ' + route).format(i))
with open('solutionnetwork.txt', "a") as solution_file:
    solution_file.write('\n\nWith a total cost of ' + str(best_routes['0.1'][0]))
