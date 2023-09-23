# importing libraries used
import os
import numpy as np
import pandas as pd

path = os.getcwd()

df = pd.read_csv(path + '/data/LPP.csv')
# df = pd.read_csv(path + '/data/research_problem.csv')


def create_variables(df):
    obj = df.loc[0, 'obj']
    obj_coeffs_str = df.loc[0, 'obj coeffs'].split(',')
    constraint_coeff_str = df.loc[0, 'constraint coeffs'].split(',')
    constraint_rels = df.loc[0, 'constraint relationships'].split(',')
    constraint_bounds_str = df.loc[0, 'constraint bounds'].split(',')

    return obj, obj_coeffs_str, constraint_coeff_str, constraint_rels, constraint_bounds_str


def list_str_2_float(string):
    temp = []
    for element in string:
        temp.append(float(element))

    return temp


def standardise_variables(constraint_coeff_str, obj_coeffs_str, constraint_bounds_str):
    constraint_coeffs = []
    for coeff_str in constraint_coeff_str:
        temp = []
        for coeff in coeff_str.split(';'):
            temp.append(float(coeff))
        constraint_coeffs.append(temp)

    obj_coeffs = list_str_2_float(obj_coeffs_str)

    constraint_bounds = list_str_2_float(constraint_bounds_str)

    return obj_coeffs, constraint_coeffs, constraint_bounds


obj, obj_coeffs_str, constraint_coeff_str, constraint_rels, constraint_bounds_str = create_variables(df)
obj_coeffs, constraint_coeffs, constraint_bounds = standardise_variables(constraint_coeff_str, obj_coeffs_str,
                                                                         constraint_bounds_str)


def state_problem(obj_coeffs, constraint_rels, A, b):
    print('Objective equation:')

    length = len(obj_coeffs)

    string = ''
    for i in range(length):
        if i < length - 1:
            string += str(obj_coeffs[i]) + '*x{} +'.format(i + 1) + ' '
        else:
            string += str(obj_coeffs[i]) + '*x{}'.format(i + 1)
    print(('{} z =').format(obj), string)

    print('\nSubject to the constraints:')

    with open('log.txt', "w") as log_file:
        log_file.write('LINEAR PROGRAMMING LIBRARY\nlog file\n\n')
        log_file.write('Objective equation: ' + ('{} z = ').format(obj) + string)
        log_file.write('\n\nSubject to the constraints:')
        log_file.close()

    shape = A.shape
    for n in range(shape[0]):
        string = ''
        for p in range(shape[1]):

            if p < shape[1] - 1:
                string += str(A[n][p]) + '*x{} +'.format(p + 1) + ' '
            else:
                string += str(A[n][p]) + '*x{}'.format(p + 1)

        print(string, constraint_rels[n], b[n])

        with open('log.txt', "a") as log_file:
            log_file.write('\n' + string + ' ' + constraint_rels[n] + ' ' + str(b[n]))
            log_file.close()


A = np.array(constraint_coeffs)
b = constraint_bounds
state_problem(obj_coeffs, constraint_rels, A, b)

# create slack or surplus variables as a square identity matrix with dimensions mxm
n = A.shape[0]
slack_surplus = np.identity(n)

# if no slack or surplus is necessary because the constraint is already an equality, remove the column
for i, rel in enumerate(constraint_rels):
    if rel == 'eq':
        slack_surplus = np.delete(slack_surplus, i, 1)
    if rel == 'meq':
        slack_surplus[i] = slack_surplus[i] * -1

T = np.hstack((A, slack_surplus))

inv_obj_coeffs = [-x for x in obj_coeffs]
for i in range(slack_surplus.shape[1]):
    inv_obj_coeffs.append(0)

T = np.vstack((T, inv_obj_coeffs))

sols = b + [0]

# ensure that the solutions column is all positive, multiply the Tableau row by -1 if not
for i, sol in enumerate(sols):
    if sol < 0:
        sols[i] = sols[i] * -1
        T[i] = T[i] * -1

sols = [[i] for i in sols]
sols = np.array(sols)

T_with_ss = np.append(T, sols, axis=1)

variables = []
slacks = []
for i in range(1, A.shape[1] + 1):
    variables.append('x{}'.format(str(i)))
for i in range(1, slack_surplus.shape[1] + 1):
    slacks.append('s{}'.format(str(i)))

columns = variables + slacks + ['Solutions']
result = pd.DataFrame(T_with_ss, columns=columns)

# create an array of artificial variables for the two conditions, (non-neg constraint violation, equality constraint) and
# insert the array in the Tableau

non_neg_constrain_abiding_rows = []
sols_signs = np.sign(sols)

for i, ss in enumerate(T[:-1, A.shape[1]:]):
    ss_signs = np.sign(ss)
    for val in ss_signs:
        if val != 0 and val != sols_signs[i][0]:
            non_neg_constrain_abiding_rows.append(i)

nnc_avs = np.zeros((T.shape[0], len(non_neg_constrain_abiding_rows)))

if obj == 'min':
    nnc_avs[-1, :] -= 10
if obj == 'max':
    nnc_avs[-1, :] += 10

for i, row_index in enumerate(non_neg_constrain_abiding_rows):
    nnc_avs[row_index, i] = 1

eq_rows = [i for i in range(len(constraint_rels)) if constraint_rels[i] == 'eq']

eq_avs = np.zeros((T.shape[0], len(eq_rows)))
if obj == 'min':
    eq_avs[-1, :] -= 10
if obj == 'max':
    eq_avs[-1, :] += 10
for i, row_index in enumerate(eq_rows):
    eq_avs[row_index, i] = 1

avs = np.append(nnc_avs, eq_avs, axis=1)
T = np.append(T, avs, axis=1)
T = np.append(T, sols, axis=1)

if obj == 'min':
    T[-1, :] = T[-1, :] + 10 * np.sum(T[non_neg_constrain_abiding_rows], axis=0)
    T[-1, :] = T[-1, :] + 10 * np.sum(T[eq_rows], axis=0)
if obj == 'max':
    T[-1, :] = T[-1, :] - 10 * np.sum(T[non_neg_constrain_abiding_rows], axis=0)
    T[-1, :] = T[-1, :] - 10 * np.sum(T[eq_rows], axis=0)

variables = []
slacks = []
artificials = []
for i in range(1, A.shape[1] + 1):
    variables.append('x{}'.format(str(i)))
for i in range(1, slack_surplus.shape[1] + 1):
    slacks.append('s{}'.format(str(i)))
for i in range(1, avs.shape[1] + 1):
    artificials.append('a{}'.format(str(i)))

print(variables, slacks, artificials)

init_columns = variables + slacks + artificials + ['Solutions']

result = pd.DataFrame(T, columns=init_columns)

# create dictionaries for target variables
all_variables = result.columns[0:-1].values
variables_dict = {init_columns.index(v): v for v in all_variables}

if artificials != []:
    artificial_variables = result.columns[-len(artificials) - 1:-1].values

save = T.copy()


def find_pivot_column(T, count):
    # the last row in the Tableau is the objective row
    objective_row = T[-1]

    if obj == 'max':
        # for the initial pivot, look at the objective row up to the last variable, not including the solutions column
        if count == 0:
            pivot_column = np.argmin(objective_row[0:-1])
            print('min of obj row:', min(objective_row[0:-1]))
        # from then on, look at the objective row up to the last variable, not including the solutions column OR
        # ratios column
        else:
            pivot_column = np.argmin(objective_row[0:-2])
            print('min of obj row:', min(objective_row[0:-2]))

    if obj == 'min':
        # for the initial pivot, look at the objective row up to the last variable, not including the solutions column
        if count == 0:
            pivot_column = np.argmax(objective_row[0:-1])
            print('max of obj row:', max(objective_row[0:-1]))
        # from then on, look at the objective row up to the last variable, not including the solutions column OR
        # ratios column
        else:
            pivot_column = np.argmax(objective_row[0:-2])
            print('max of obj row:', max(objective_row[0:-2]))

    return pivot_column


def find_pivot_row(T, count, pivot_column):
    # for each row, create a list of ratios
    ratios = []
    if count == 0:
        for row in T:
            ratios.append((row[-1]) / row[pivot_column])
    else:
        for row in T:
            ratios.append((row[-2]) / row[pivot_column])

    if count == 0:
        T = np.vstack((T.T, ratios)).T  # append to Tableau
    else:
        T[:, -1] = ratios  # update ratios column in Tableau

    # find the smallest NON-NEGATIVE ratio, ensuring to not include the objective equation row in the calculation
    ratios = T[:, -1]

    usable_ratios_column = np.array(ratios[:-1])
    mask = usable_ratios_column >= 0
    _min = usable_ratios_column[mask][usable_ratios_column[mask].argmin()]
    pivot_row = np.where(usable_ratios_column == _min)

    return T, pivot_row[0][0], _min


def row_manipulator(T, count, pivot_row, pivot_column):
    T[pivot_row] = T[pivot_row] / T[pivot_row, pivot_column]
    for row in range(T.shape[0]):
        if row != pivot_row:
            T[row] = T[row] - T[pivot_row] * T[row, pivot_column]
    return T


def solve_simplex_1(T, basis, columns, count):
    print(" --- Iteration ", count, "---")

    pivot_column = find_pivot_column(T, count)
    print('Column to pivot on:', pivot_column)
    print('Dividing solution column by pivot column...')

    ##
    with open('log.txt', "a") as log_file:
        log_file.write('\n\n\n --- Iteration ' + str(count) + ' ---')
        log_file.write('\nColumn to pivot on: ' + str(pivot_column))
        log_file.write(', ' + variables_dict[pivot_column] + ' is entering the basis')
        log_file.write('\nDividing solution column by pivot column...\n\n')

        log_file.close()
    ##

    T, pivot_row, _min = find_pivot_row(T, count, pivot_column)
    basis.append([pivot_row, pivot_column])

    print(pd.DataFrame(T, columns=columns))
    print('Smallest non-negative ratio:', _min)
    print('Row to pivot on:', pivot_row)
    print('Pivoting on:', pivot_row, pivot_column)

    ##
    T_str = (pd.DataFrame(T, columns=columns)).to_string()
    with open('log.txt', "a") as log_file:
        log_file.write(T_str)
        log_file.write('\n\nSmallest non-negative ratio: ' + str(round(_min, 2)))
        log_file.write('\nRow to pivot on: ' + str(pivot_row))
        log_file.write('\nPivoting on: ' + str(pivot_row) + ', ' + str(pivot_column) + '\n')
        log_file.close()
    ##

    print('\nManipulating rows...')
    T = row_manipulator(T, count, pivot_row, pivot_column)
    print('Pivot complete, result:')
    print(pd.DataFrame(T, columns=columns))

    ##
    T_str = (pd.DataFrame(T, columns=columns)).to_string()
    with open('log.txt', "a") as log_file:
        log_file.write('\nManipulating rows...\n')
        log_file.write('Pivot complete, result:\n\n')
        log_file.write(T_str)

        log_file.close()
    ##

    print('\n')
    count += 1
    return T, basis, count


def solve_simplex(T, basis, columns):
    count = 0

    if obj == 'max':

        while (count == 0 and any(x < 0 for x in T[-1, 0:-1])) or (count < 50 and any(x < 0 for x in T[-1, 0:-2])):
            T, basis, count = solve_simplex_1(T, basis, columns, count)

    elif obj == 'min':

        while (count == 0 and any(x > 0 for x in T[-1, 0:-1])) or (count < 50 and any(x > 0 for x in T[-1, 0:-2])):
            T, basis, count = solve_simplex_1(T, basis, columns, count)

    return T, basis


T = save.copy()
initial_tableau = pd.DataFrame(T, columns=init_columns)
initial_tableau_str = initial_tableau.to_string()

with open('log.txt', "a") as log_file:
    log_file.write('\n\nInitial Tableau: \n')
    log_file.write(initial_tableau_str)
    log_file.close()

basis = []
columns = variables + slacks + artificials + ['Solutions'] + ['Ratios']
T, basis = solve_simplex(T, basis, columns)

print('\n\nFinal Result:')
result = pd.DataFrame(T, columns=columns)
print(result)

# basic_variables = [('{}').format(variables_dict[basis[i][1]]) for i in range(len(basis))]
basic_variables = ['{}'.format(variables_dict[i[1]]) for i in basis if result[variables_dict[i[1]]].iloc[-1] == 0]
basic_variables = sorted(basic_variables)
strings = []
values = {}
for i, variable in enumerate(basic_variables):
    if 'x' in variable:
        col = result[variable]
        row = np.where(col == 1)
        val = round(result.loc[row, 'Solutions'].values[0], 2)
        string = ['{}'.format(variable), ' = ', str(val)]
        string = ''.join(string)
        values[variable] = val
        strings.append(string)

print(values)
obj_coeffs_dict: dict = {'x{}'.format(i + 1): obj_coeffs[i] for i in range(len(obj_coeffs))}

print('Optimal solution:', ', '.join(strings), '... z =', round(result['Solutions'].iloc[-1], 2))

result = pd.DataFrame(T, columns=columns)
result_str = result.to_string()
z = sum(obj_coeffs_dict[i] * values[i] for i in basic_variables if 'x' in i)

string = ''
for i in range(len(obj_coeffs)):
    if i < len(obj_coeffs) - 1:
        string += str(obj_coeffs[i]) + '*x{} +'.format(i + 1) + ' '
    else:
        string += str(obj_coeffs[i]) + '*x{}'.format(i + 1)

with open('result.txt', "w") as results_file:
    results_file.write('LINEAR PROGRAMMING LIBRARY\nresults file\n\n')
    results_file.write('\n' + 'Final Tableau \n' + result_str + '\n')
    results_file.write(
        ''.join(['\nOptimal solution: ', ', '.join(strings), '\nObjective equation: ', '{} z = '.format(obj), string,
                 '\n\n ... z (from Tableau) = ', str(round(result['Solutions'].iloc[-1], 2)),
                 '\n ... z (from calculation) = ', str(z)]))

    results_file.close()


def find_multi_opt(T, basis):
    # find the column indexes of the columns who's objective function row = 0
    col_idx = [list(result.columns).index(i) for i in basic_variables if 'x' in i]
    col_idx.sort()

    # compare these column indexes to the indexes of the variables in the basis as defined by 'basis'
    basis_variables = [i[1] for i in basis if i[1] < A.shape[1]]
    basis_variables.sort()

    with open('result.txt', "a") as results_file:

        if basis_variables != col_idx:
            results_file.write('\n\nMultiple Optima? Multiple optima exist\n')
        else:
            results_file.write('\n\nMultiple Optima? Only one optimal solution')
        results_file.close()


find_multi_opt(T, basis)
