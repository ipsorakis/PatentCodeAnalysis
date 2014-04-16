__author__ = 'yannis'

import numpy
import copy
import math
import graph_tool as gt

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

def read_top_level_community_structure_from_dot_tree_to_community_list(filename, G = None, allowed_nodes = None):
    node_labels = []
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

            Y.append(int(elems[0]))

    community_indices = set(Y)
    C = len(community_indices)
    g = [ [] for i in range(0,C) ]

    for i in range(0,len(Y)):
        c = Y[i]-1

        #if G is not None and filter is not None and len(node_labels):
        #    filtered_node_labels = [node_label for node_label in node_labels[i] if allowed_nodes[G.vertex(G.graph_properties['index_of'][node_label])]]
        #    if len(filtered_node_labels)>0:
        #        g[c].append(f)
        #else:
        g[c].append(node_labels[i])

    if G is not None and filter is not None:
        for c in range(0,len(g)):
            g_base = [elem for elem in g[c]]
            g[c] = [node_label for node_label in g_base if allowed_nodes[G.vertex(G.graph_properties['index_of'][node_label])]]

        g_filtered = [g[c] for c in range(0,len(g)) if len(g[c])!=0]
        return g_filtered
    else:
        return g

def read_top_level_community_structure_from_dot_tree_to_gt_property_map(G,filename):
    pmap = G.new_vertex_property('int')
    with open(filename,'r') as fp:
        line_count = 0
        for aline in fp:
            line_count +=1
            if line_count == 1:continue
            aline = aline.strip()

            fields = aline.split(' ')
            node_name = fields[-1]
            node_name = node_name[1:-1]

            node_index = G.graph_properties['index_of'][node_name]
            node = G.vertex(node_index)

            elems = fields[0].split(':')
            com_cur = int(elems[0])
            pmap[node] = com_cur

    return pmap

def read_bottom_level_community_structure_from_dot_tree_to_community_list(filename,G = None,allowed_nodes = None):
    node_labels = []
    COMM = []
    with open(filename,'r') as fp:

        line_count = 0
        for aline in fp:
            line_count +=1
            if line_count == 1:
                com_prev = -1
                no_elems_prev = -1
                g = []
                continue
            aline = aline.strip()

            fields = aline.split(' ')
            node_name = fields[-1]
            node_labels.append(node_name[1:-1])

            elems = fields[0].split(':')
            no_elems_cur = len(elems)
            if no_elems_cur>1:
                com_cur = int(elems[-2])
            else:
                com_cur = int(elems[0])

            if ((no_elems_prev == no_elems_cur and com_cur != com_prev) or no_elems_cur != no_elems_prev) and com_prev != -1:
                COMM.append(g)
                g = []
            g.append(node_name[1:-1])
            com_prev = com_cur
            no_elems_prev = no_elems_cur

    COMM.append(g)

    if G is not None and filter is not None:
        for c in range(0,len(COMM)):
            COMM_base = [elem for elem in COMM[c]]
            COMM[c] = [node_label for node_label in COMM_base if allowed_nodes[G.vertex(G.graph_properties['index_of'][node_label])]]

        COMM_filtered = [COMM[c] for c in range(0,len(COMM)) if len(COMM[c])!=0]
        return COMM_filtered
    else:
        return COMM

def read_bottom_level_community_structure_from_dot_tree_to_gt_property_map(G,filename):
    pmap = G.new_vertex_property('int')
    with open(filename,'r') as fp:
        line_count = 0
        for aline in fp:
            line_count +=1
            if line_count == 1:continue
            aline = aline.strip()

            fields = aline.split(' ')
            node_name = fields[-1]
            node_name = node_name[1:-1]
            node_index = G.graph_properties['index_of'][node_name]
            node = G.vertex(node_index)

            elems = fields[0].split(':')
            if len(elems)>1:
                com_cur = int(elems[-2])
            else:
                com_cur = int(elems[0])

            pmap[node] = com_cur

    return pmap


def read_community_structure_from_dot_map_to_community_list(filename):
    node_labels = []
    Y = []
    with open(filename,'r') as fp:

        in_com = False
        line_count = 0
        for aline in fp:
            line_count +=1
            aline = aline.strip()
            fields = aline.split(' ')

            if fields[0] == '*Nodes':
                in_com = True
                continue
            elif fields[0] == '*Links':
                break

            if in_com:
                elems = fields[0].split(':')
                Y.append(int(elems[0]))
                node_name = fields[1]
                node_labels.append(node_name[1:-1])

    community_indices = set(Y)
    C = len(community_indices)
    g = [ [] for i in range(0,C) ]

    for i in range(0,len(Y)):
        c = Y[i]-1
        g[c].append(node_labels[i])

    return g

def get_normalised_mutual_information(g1,g2):
    # get confusion matrix
    N = numpy.zeros((len(g1),len(g2)))

    for i in range(0,len(g1)):
        for j in range(0,len(g2)):
            if not isinstance(g1[i],list):
                g1i = [g1[i]]
            else:
                g1i = g1[i]

            if not isinstance(g2[j],list):
                g2j = [g2[j]]
            else:
                g2j = g2[j]
            N[i,j] = len([val for val in g1i if val in g2j])


    a = N.shape[0]
    b = N.shape[1]

    Na = numpy.sum(N,1)
    Nb = numpy.sum(N,0)
    n = N.sum()

    enumerator = 0
    for i in range(0,a):
        for j in range(0,b):

            joint = N[i,j]*n
            prod_marginals = Na[i]*Nb[j]

            if prod_marginals !=0:
                ratio = joint/prod_marginals
            else:
                ratio = 1

            if ratio !=0:
                aux = numpy.log(ratio)
            else:
                aux = 0

            enumerator += N[i,j]*aux
    enumerator *= -2

    denominator = 0
    for i in range(0,a):
        summand = Na[i]*numpy.log(Na[i]/n)
        if numpy.isnan(summand):
            summand = 0
        denominator += summand

    for i in range(0,b):
        summand = Nb[i]*numpy.log(Nb[i]/n)
        if numpy.isnan(summand):
            summand = 0

        denominator += summand

    NMI = enumerator/denominator

    if math.isnan(NMI):
        NMI = 0
    return NMI


def get_community_substructure_from_node_intersect(g1_base,g2_base,node_set):
    g1 = copy.deepcopy(g1_base)
    g2 = copy.deepcopy(g2_base)

    C1 = len(g1)
    to_remove = []
    for c in range(0,C1):
        if not isinstance(g1[c],int):
            g1[c] = [v for v in g1[c] if v in node_set]
            if len(g1[c]) == 0:to_remove.append(c)
        else:
            if g1[c] not in node_set:to_remove.append(c)

    g1 = [g1[c] for c in range(0,len(g1)) if c not in to_remove]

    C2 = len(g2)
    to_remove = []
    for c in range(0,C2):
        if not isinstance(g2[c],int):
            g2[c] = [v for v in g2[c] if v in node_set]
            if len(g2[c]) == 0:to_remove.append(c)
        else:
            if g2[c] not in node_set:to_remove.append(c)

    g2 = [g2[c] for c in range(0,len(g2)) if c not in to_remove]

    return g1,g2

def get_node_set_from_community_list(g):
    return set([v for subg in g for v in subg])

def get_normalised_mutual_information_between_code_communities_and_USPTO_hierarchy(g):
    code_set = set([code for subg in g for code in subg])
    classes = dict()
    for code in code_set:
        elems = code.split('/')
        class_label = elems[0]
        if not classes.has_key(class_label):
            classes[class_label] = []
        classes[class_label].append(code)

    class_hierarchy = [classes[class_label] for class_label in classes.keys()]

    return get_normalised_mutual_information(g,class_hierarchy),class_hierarchy

def convert_property_map_to_community_list(G,pmap):
    comm_labels_lookup = dict()
    for v in G.vertices():
        if not comm_labels_lookup.has_key(pmap[v]):
            comm_labels_lookup[pmap[v]] = [G.vertex_properties['label'][v]]
        else:
            aux = comm_labels_lookup[pmap[v]]
            aux.append(G.vertex_properties['label'][v])
            comm_labels_lookup[pmap[v]] = aux

    return comm_labels_lookup.values()