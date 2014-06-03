__author__ = 'yannis'


import numpy

class tree_node:
    def __init__(self,name, parent = None):
        self.name = name
        self.parent = parent


class tree:
    def __init__(self):
        self.node_lookup = dict()
        self.root = tree_node('ROOT')
        self.node_lookup[self.root.name] = self.root

    def get_node_by_name(self,name):
        return self.node_lookup[name]

    def add_node(self,name,parent = None):
        if parent is None:
            parent = self.root
        elif isinstance(parent,str):
            parent = self.node_lookup[parent]

        n = tree_node(name,parent)
        self.node_lookup[n.name] = n

        return n

    def get_parent(self,node_object):
        if isinstance(node_object,tree_node):
            return node_object.parent
        elif isinstance(node_object,str):
            focal_node = self.node_lookup[node_object]
            return focal_node.parent

    def get_parent_name(self,node_object):
        parent = self.get_parent(node_object)
        return parent.name

    def get_children(self,node_object):
        if isinstance(node_object,str):
            focal_node = self.node_lookup[node_object]
        elif isinstance(node_object,tree_node):
            focal_node = node_object

        children = [node_object for node_object in self.node_lookup.values() if node_object.parent is focal_node]
        return children

    def get_children_names(self,node_object):
        children = self.get_children(node_object)
        return [child.name for child in children]

    def get_path_to_root(self,node_object):
        if isinstance(node_object,str):
            u = self.node_lookup[node_object]
        elif isinstance(node_object,tree_node):
            u = node_object

        path = []
        current_parent = u.parent
        while current_parent is not None:
            path.append(current_parent)
            current_parent = current_parent.parent
        return path

    def get_path_to_root_as_node_names(self,u):
        path_to_root = self.get_path_to_root(u)
        return [check_point.name for check_point in path_to_root]

    def get_tree_distance(self,node_object_u,node_object_v):
        if isinstance(node_object_u,str):
            u = self.node_lookup[node_object_u]
        elif isinstance(node_object_u,tree_node):
            u = node_object_u

        if isinstance(node_object_v,str):
            v = self.node_lookup[node_object_v]
        elif isinstance(node_object_v,tree_node):
            v = node_object_v

        path_u = self.get_path_to_root_as_node_names(u)
        path_v = self.get_path_to_root_as_node_names(v)

        i=0
        for y in path_u:
            if y in path_v:
                lu = i+1
                lv = path_v.index(y)+1
                break
        return lu+lv


######## AD HOC FUNCTIONS
def convert_infomap_hierarchy_dot_tree_to_tree_string(filename):
    tree_list = convert_infomap_hierarchy_dot_tree_to_tree_list(filename)
    tree_string = convert_list_to_string_representation(tree_list)

    return tree_string

def convert_infomap_hierarchy_dot_tree_to_tree_list(filename):
    X,node_labels = read_infomap_hierarchy_dot_tree_file_contents(filename)
    return parse_infomap_hierarchy_dot_tree_column(X,node_labels)

def read_infomap_hierarchy_dot_tree_file_contents(filename):
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

def parse_infomap_hierarchy_dot_tree_column(X,node_labels):
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
            current_level.append(parse_infomap_hierarchy_dot_tree_column(X[pivot1:pivot2,1:],node_labels[pivot1:pivot2]))
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