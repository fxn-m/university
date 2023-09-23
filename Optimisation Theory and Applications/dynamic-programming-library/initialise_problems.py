# create the text files that describes the initial network and capital budgeting problems
# to be read back into the corresponding python scripts

with open('inputnetwork.txt', 'w') as input_txt:
    input_txt.write('0.1:[2, 1]\n')
    input_txt.write('1.2:[2, 3]\n')
    input_txt.write('1.1:[3, 2]\n')
    input_txt.write('2.3:[2, 3]\n')
    input_txt.write('2.2:[6, 2]\n')
    input_txt.write('2.1:[4, 5]\n')
    input_txt.write('3.4:[3, 4]\n')
    input_txt.write('3.3:[5, 1]\n')
    input_txt.write('3.2:[2, 3]\n')
    input_txt.write('3.1:[3, 4]\n')


with open('inputcapbud.txt', 'w') as input_txt:
    input_txt.write('11,2,3\n')
    input_txt.write('12,4,6\n')
    input_txt.write('13,7,10\n')
    input_txt.write('21,1,2\n')
    input_txt.write('22,3,5\n')
    input_txt.write('31,3,5\n')
    input_txt.write('32,5,7\n')
    input_txt.write('33,8,13\n')
    input_txt.write('CB:14')
