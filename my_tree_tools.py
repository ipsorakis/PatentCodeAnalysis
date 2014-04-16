__author__ = 'yannis'


import numpy

def convert_dot_tree_to_tree_string(filename):
    tree_list = convert_dot_tree_to_tree_list(filename)
    tree_string = convert_list_to_string_representation(tree_list)

    return tree_string

def convert_dot_tree_to_tree_list(filename):
    X,node_labels = read_dot_tree_file_contents(filename)
    return parse_dot_tree_column(X,node_labels)

def read_dot_tree_file_contents(filename):
    node_labels = []
    max_depth = 0
    with open(filename,'r') as fp:
        Y = []
        line_count = 0
        for aline in fp:
            line_count +=1

            if line_count == 1:continue

            aline = aline.strip()

            fields = aline.split(' ')

            node_name = fields[-1]
            node_labels.append(node_name[1:-1])

            elems = fields[0].split(':')

            y = map(int,elems)
            if len(y) > max_depth:
                max_depth = len(y)
            Y.append(y)

        for y in Y:
            lendiff = max_depth - len(y)
            if lendiff >0:
                for i in range(0,lendiff):
                    y.append(1)


    X = numpy.array(Y)
    return X,node_labels

def parse_dot_tree_column(X,node_labels):
    current_level = []

    comm_indices = X[:,0]
    max_index = comm_indices[-1]

    pivot2 = 0
    for i in range(1,max_index+1):
        pivot1 = pivot2
        curr_elem = comm_indices[pivot2]
        while curr_elem == i and pivot2<len(comm_indices):
            pivot2 +=1
            if pivot2<len(comm_indices): curr_elem = comm_indices[pivot2]

        if pivot2 - pivot1 > 1:
            current_level.append(parse_dot_tree_column(X[pivot1:pivot2,1:],node_labels[pivot1:pivot2]))
        else:
            current_level.append(node_labels[i-1])

    if len(current_level) ==1:current_level = current_level[0]

    return current_level

def convert_list_to_string_representation(tree_list):
    tree_string = str(tree_list)
    tree_string = tree_string.replace('[','(')
    tree_string = tree_string.replace(']',')')
    tree_string += ';'

    return tree_string