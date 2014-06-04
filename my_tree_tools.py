__author__ = 'yannis'


import numpy

class tree_node:
    def __init__(self,name, parent = None):
        self.name = name
        self.parent = parent
        self.children = set()
        if isinstance(self.parent,tree_node):
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

class tree:
    def __init__(self,root_object = None):
        self.node_lookup = dict()
        if root_object is None:
            self.root = tree_node('ROOT')
        elif isinstance(root_object,str):
            self.root = tree_node(root_object)
        elif isinstance(root_object,tree_node):
            self.root = root_object
        self.node_lookup[self.root.name] = self.root
        self.max_depth = 0
    
    def format_input_node_object_to_tree_node(self,node_object,None_option = None):
        if isinstance(node_object,str):
            return self.node_lookup[node_object]
        elif isinstance(node_object,tree_node):
            return node_object
        elif node_object is None:
            return None_option

    def get_total_number_of_nodes(self):
        return len(self.node_lookup)

    def has_node_with_name(self,name):
        return self.node_lookup.has_key(name)

    def get_node_by_name(self,name):
        return self.node_lookup[name]
    
    def get_depth_of_node(self,node_object):
        u = self.format_input_node_object_to_tree_node(node_object)
        return u.depth

    def add_node(self,name,parent_object = None):
        parent = self.format_input_node_object_to_tree_node(parent_object,self.root)

        new_node = tree_node(name,parent)
        parent.children.add(new_node)

        self.node_lookup[new_node.name] = new_node
        if self.max_depth<new_node.depth:
            self.max_depth = new_node.depth

        return new_node

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
        focal_node = self.format_input_node_object_to_tree_node(node_object)
        return focal_node.children

    def get_children_names(self,node_object):
        children = self.get_children(node_object)
        return [child.name for child in children]

    def get_path_to_root(self,node_object):
        u = self.format_input_node_object_to_tree_node(node_object)

        path = list()
        # by default includes self
        path.append(u)
        current_parent = u.parent
        while current_parent is not None:
            path.append(current_parent)
            current_parent = current_parent.parent
        return path

    def get_path_to_root_as_node_names(self,u):
        path_to_root = self.get_path_to_root(u)
        return [check_point.name for check_point in path_to_root]

    def get_path_from_to(self,node_object_u,node_object_v):
        u = self.format_input_node_object_to_tree_node(node_object_u)
        v = self.format_input_node_object_to_tree_node(node_object_v)

        path_u = self.get_path_to_root(u)
        path_v = self.get_path_to_root(v)

        path_u_names = [n.name for n in path_u]
        path_v_names = [n.name for n in path_v]

        i=0
        for y in path_u_names:
            if y in path_v_names:
                lu = i+1
                lv = path_v_names.index(y)+1

                path_to_common_ancestor_u = path_u[0:lu]
                path_to_common_ancestor_v = path_v[0:lv]

                return path_to_common_ancestor_u[0:-1] + path_to_common_ancestor_v[::-1]
            i+=1

    def get_path_from_to_as_node_names(self,node_object_u,node_object_v):
        u = self.format_input_node_object_to_tree_node(node_object_u)
        v = self.format_input_node_object_to_tree_node(node_object_v)

        path_u = self.get_path_to_root_as_node_names(u)
        path_v = self.get_path_to_root_as_node_names(v)

        i=0
        for y in path_u:
            if y in path_v:
                lu = i+1
                lv = path_v.index(y)+1

                path_to_common_ancestor_u = path_u[0:lu]
                path_to_common_ancestor_v = path_v[0:lv]

                return path_to_common_ancestor_u[0:-1] + path_to_common_ancestor_v[::-1]
            i+=1

    def get_leaf_to_leaf_distance(self,node_object_u,node_object_v):
        u = self.format_input_node_object_to_tree_node(node_object_u)
        v = self.format_input_node_object_to_tree_node(node_object_v)

        common_ancestor_depth = self.get_first_common_ancestor_depth(u,v)
        return u.depth + v.depth - 2*common_ancestor_depth

    def get_leaf_to_leaf_distance_exp_weighted(self,node_object_u,node_object_v,alpha=1,beta=1):
        u = self.format_input_node_object_to_tree_node(node_object_u)
        v = self.format_input_node_object_to_tree_node(node_object_v)

        #common_ancestor_depth = self.get_first_common_ancestor_depth(u,v)
        #u_depths = range(u.depth,common_ancestor_depth-1,-1)
        #v_depths = range(v.depth,common_ancestor_depth-1,-1)
        #

        path_u_v = self.get_path_from_to(u,v)
        depth_path = [node.depth for node in path_u_v]
        weighted_distances = [numpy.exp(-(alpha/beta) *d) for d in depth_path[1:-1]]

        return sum(weighted_distances)

    def get_first_common_ancestor(self,node_object_u,node_object_v):
        u = self.format_input_node_object_to_tree_node(node_object_u)
        v = self.format_input_node_object_to_tree_node(node_object_v)

        path_u = self.get_path_to_root_as_node_names(u)
        path_v = self.get_path_to_root_as_node_names(v)

        for ancestor_name in path_u:
            if ancestor_name in path_v:
                ancestor_node = self.node_lookup[ancestor_name]
                return ancestor_node

    def get_first_common_ancestor_name(self,node_object_u,node_object_v):
        ancestor_node = self.get_first_common_ancestor(node_object_u,node_object_v)
        return ancestor_node.name

    def get_first_common_ancestor_depth(self,node_object_u,node_object_v):
        ancestor_node = self.get_first_common_ancestor(node_object_u,node_object_v)
        return ancestor_node.depth

    def get_leaves(self):
        return [anode for anode in self.node_lookup.values() if len(anode.children)==0]

    def get_leaf_names(self):
        leaves = self.get_leaves()
        return [leaf.name for leaf in leaves]
    
    def get_number_of_leaves(self):
        leaves = self.get_leaves()
        return len(leaves)

    def get_number_of_children(self,node_object):
        focal_node = self.format_input_node_object_to_tree_node(node_object)
        return focal_node.children

    def get_siblings(self,node_object):
        focal_node = self.format_input_node_object_to_tree_node(node_object)

        if focal_node.parent is self.root:
            return []
        else:
            siblings = [child for child in focal_node.parent.children if child is not focal_node]
            return siblings

    def get_sibling_names(self,node_object):
        siblings = self.get_siblings(node_object)
        return [sibling.name for sibling in siblings]

# TO IMPLEMENT
# =============
# - CHANGE ROOT LEVEL TO 0 from -1 - DONE
# - DROPPED DUMMY ROOT - DONE
# - RE-WRITE TREE DISTANCE BASED ON DEPTH - DONE
# - FUNCTION: FIND PATH FROM U TO V - DONE
# - PATHS INCLUDE SELF - DONE
# - add get depth given name - DONE
# - added function to format input node_object to tree_node - DONE
# - get_distance changed to get_leaf_to_leaf_distance - DONE

# - export to nested list
# - flatten tree given root node
# - read tree from a nested list
# - automatic node name generator


######## AD HOC FUNCTIONS
#--- Daniel's tree output
def parse_Daniel_semicolon_based_tree_format_to_my_tree_class(filename):
    t = tree()

    with open(filename,'r') as fp:
        for aline in fp:
            aline = aline.strip()
            node_names = aline.split(':')
            n=0
            for node_name in node_names:
                if n==0:
                    class_name = node_name
                    if not t.has_node_with_name(node_name):
                        current_node = t.add_node(node_name)
                    else:
                        current_node = t.get_node_by_name(node_name)
                else:
                    code_name = class_name + '/' + node_name
                    if not t.has_node_with_name(code_name):
                        current_node = t.add_node(code_name,prev_node)
                    else:
                        current_node = t.get_node_by_name(code_name)
                prev_node = current_node
                n+=1
    return t
#---- Infomap
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

######## AUX

def convert_list_to_string_representation(tree_list):
    tree_string = str(tree_list)
    tree_string = tree_string.replace('[','(')
    tree_string = tree_string.replace(']',')')
    tree_string += ';'

    return tree_string