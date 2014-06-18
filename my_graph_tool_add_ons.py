__author__ = 'yannis'

import graph_tool as gt
import graph_tool.community as gtcom
import math
import numpy
import scipy.sparse
import my_community_tools as mycomms
import itertools
import my_stat_tools as mystats

def is_vertex_connected(v):
    return v.out_degree()!=0
def is_vertex_singleton(v):
    return not is_vertex_connected(v)

# IN LINK-LIST AND PAJEK, NODE INDICES START FROM 1 !!!

def write_graph_to_link_list(G,filename,weight_name = None):
    with open(filename,'w') as fid:
        for an_edge in G.edges():
            i = int(an_edge.source()) + 1
            j = int(an_edge.target()) + 1

            output_line1 = str(i)+' '+str(j)
            #output_line2 = str(j)+' '+str(i)


            if not weight_name is None:
                w = G.edge_properties[weight_name][an_edge]
                if isinstance(w,int):
                    output_line1 = output_line1 + ' ' + str(w)
                    #output_line2 = output_line2 + ' ' + str(w)
                else:
                    output_line1 = output_line1 + ' ' + '{0:.2f}'.format(w)
                    #output_line2 = output_line2 + ' ' + '{0:.2f}'.format(w)

            fid.write(output_line1 + '\n')
            #fid.write(output_line2 + '\n')

def write_graph_to_pajek(G,filename,node_name = 'label',weight_name = 'co_oc'):
    vertex_lookup = dict()
    i=0
    with open(filename,'w') as fid:
        fid.write('*Vertices ' + str(G.num_vertices()) + '\n')
        for v in G.vertices():

            i +=1

            label = '"' +  G.vertex_properties[node_name][v] + '"'
            vertex_lookup[G.vertex_properties[node_name][v]] = i

            #if not node_name is None:
            #    label = '"' +  G.vertex_properties[node_name][v] + '"'
            #else:
            #    label = '"' + str(i) + '"'

            output_line = str(i) + ' ' + label

            fid.write(output_line + '\n')

        fid.write('*Edges ' + str(G.num_edges()) + '\n')
        for an_edge in G.edges():

            label_i = G.vertex_properties['label'][an_edge.source()]
            label_j = G.vertex_properties['label'][an_edge.target()]

            i = vertex_lookup[label_i]
            j = vertex_lookup[label_j]

            output_line1 = str(i)+' '+str(j)
            #output_line2 = str(j)+' '+str(i)


            if not weight_name is None:
                w = G.edge_properties[weight_name][an_edge]
                if isinstance(w,int):
                    output_line1 = output_line1 + ' ' + str(w)
                    #output_line2 = output_line2 + ' ' + str(w)
                else:
                    output_line1 = output_line1 + ' ' + '{0:.2f}'.format(w)
                    #output_line2 = output_line2 + ' ' + '{0:.2f}'.format(w)


            fid.write(output_line1 + '\n')
            #fid.write(output_line2 + '\n')

def load_graph_from_pajek(filename,is_directed=False,vertex_label = 'label',weight_label = None,weight_type = None):
    G = gt.Graph(directed = is_directed)
    #if vertex_label is not None:
    G.vertex_properties[vertex_label] = G.new_vertex_property('string')
    if weight_label is not None:
        G.edge_properties[weight_label] = G.new_edge_property(weight_type)

    G.graph_properties['index_of'] = G.new_graph_property('object')
    G.graph_properties['index_of'] = dict()

    with open(filename,'r') as fp:
        line_count=0

        for aline in fp:
            line_count+=1
            aline = aline.strip()

            if aline[0:3]=='*Ve':
                in_vertices = True
                continue

            if aline[0:3] == '*Ed':
                in_vertices = False
                in_edges = True
                continue

            if in_vertices:
                elems = aline.split(' ')
                #vertex_index = int(elems[0]-1)
                elem_label = elems[1]
                elem_label = elem_label[1:-1]

                v = G.add_vertex()
                G.vertex_properties[vertex_label][v] = elem_label
                G.graph_properties['index_of'][elem_label] = int(v)

            elif in_edges:
                elems = aline.split(' ')

                vertex1_index = int(elems[0])-1
                vertex2_index = int(elems[1])-1
                e = G.add_edge(G.vertex(vertex1_index),G.vertex(vertex2_index))

                if weight_label is not None:
                    if weight_type is 'int':
                        G.edge_properties[weight_label][e] = int(elems[2])
                    elif weight_type is 'float':
                        G.edge_properties[weight_label][e] = float(elems[2])

    return G

def are_indices_consistent_with_labels(G):
    ic = G.graph_properties['index_of'][G.vertex_properties['label'][G.vertex(0)]] == 0
    if not ic:
        print 'Problem in Node: 0 with Label: ' + G.vertex_properties['label'][G.vertex(0)]

    for i in range(1,G.num_vertices()):
        ic_current = G.graph_properties['index_of'][G.vertex_properties['label'][G.vertex(i)]] == i
        if not ic_current:
            print 'Problem in Node: ' + str(i) + ' with Label: ' + G.vertex_properties['label'][G.vertex(0)]
        ic = ic and ic_current
    return ic

def export_vertex_map_to_python_list(G,property_name = 'label'):
    node_labels = [None] * G.num_vertices()
    for v in G.vertices():
        i = int(v)
        node_labels[i] = G.vertex_properties[property_name][v]
    return node_labels

def export_vertex_map_to_python_set(G,property_name = 'label'):
    return set(export_vertex_map_to_python_list(G,property_name))

####

def get_edge_weight(G,label1,label2,weight_label = 'co_oc'):
    v1_index = G.graph_properties['index_of'][label1]
    v2_index = G.graph_properties['index_of'][label2]

    v1 = G.vertex(v1_index)
    v2 = G.vertex(v2_index)

    e = G.edge(v1,v2)

    if not e is None:
        w = G.edge_properties[weight_label][e]
    else:
        w = 0

    return w

def get_edge_info(G,e,node_label = 'label', weight_label = 'co_oc'):
    node1 = e.source()
    node2 = e.target()

    node1_label = G.vertex_properties[node_label][node1]
    node2_label = G.vertex_properties[node_label][node2]

    weight = G.edge_properties[weight_label][e]

    return node1_label + ' -> ' + node2_label + ', ' + weight_label + ': ' + str(weight)

###

def get_vertex_strength(G,vertex,weight_label='co_oc'):
    strength = 0
    for e in vertex.out_edges():
        strength += G.edge_properties[weight_label][e]

    return strength

def get_strength_lookup_table(G,vertex_label='label',weight_label='co_oc'):
    strengths = dict()
    for v in G.vertices():
        strengths[G.vertex_properties[vertex_label][v]] = get_vertex_strength(G,v,weight_label)
    return strengths

def get_strength_sequence(G,weight_label='co_oc'):
    strengths=[]
    for v in G.vertices():
        strengths.append(get_vertex_strength(G,v,weight_label))
    return strengths
####

def get_degree_lookup_table(G,label = 'label'):
    degrees = dict()
    for v in G.vertices():
        degrees[G.vertex_properties[label][v]] = v.out_degree()
    return degrees

def get_degree_sequence(G):
    degrees = []
    for v in G.vertices():
        degrees.append(v.out_degree())
    return degrees

def get_degree_distribution(G):
    N = G.num_vertices()
    degree = numpy.zeros((N,1),dtype=numpy.int)
    for i in range(0,N):
        v = G.vertex(i)
        degree[i] = v.out_degree()

    min_degree = 0
    max_degree = max(degree)
    degree_value_range = range(min_degree,max_degree+1)

    P_degree = numpy.zeros(len(degree_value_range),dtype=numpy.int)

    for n in range(0,N):
        P_degree[degree[n]]+=1

    return degree_value_range,P_degree

####

def calculate_SR(G, co_oc_label = 'co_oc', occurrence_label = 'No_of_occurrences'):
    G.edge_properties['SR'] = G.new_edge_property('float')
    for e in G.edges():
        node1 = e.source()
        node2 = e.target()

        co_oc = float(G.edge_properties[co_oc_label][e])
        X1 = G.vertex_properties[occurrence_label][node1]
        X2 = G.vertex_properties[occurrence_label][node2]
        X12 = X1 + X2 - co_oc

        if X12 != 0:
            G.edge_properties['SR'][e] = co_oc / X12
        else:
            print 'Warning!!! co_oc: {0}, X1: {1}, X2: {2}'.format(co_oc,X1,X2)
    return G

def calculate_odds_ratio(G, co_oc_label = 'co_oc', occurrence_label = 'No_of_occurrences'):
    G.edge_properties['LOGODDS'] = G.new_edge_property('float')
    for e in G.edges():
        node1 = e.source()
        node2 = e.target()

        # number of patents both 1 and 2 appeared
        co_oc = float(G.edge_properties[co_oc_label][e])
        # number of patents 1 appeared
        X1 = G.vertex_properties[occurrence_label][node1]
        # number of patents 2 appeared
        X2 = G.vertex_properties[occurrence_label][node2]

        G.edge_properties['LOGODDS'][e] = numpy.log(co_oc / (X1 * X2),10)

    return G

####

def merge_cooccurrence_networks(Gbase,Gnew = None,renormalise=True,check_consistency=False):
    Gmerged = gt.Graph(Gbase)

    if Gnew is None:
        return Gmerged

    #if Gmerged.graph_properties.has_key('total_cooc') and Gnew.graph_properties.has_key('total_cooc'):
    #    Gmerged.graph_properties['total_cooc'] += Gnew.graph_properties['total_cooc']

    if Gmerged.graph_properties.has_key('total_patents') and Gnew.graph_properties.has_key('total_patents'):
        Gmerged.graph_properties['total_patents'] += Gnew.graph_properties['total_patents']

    nodes_added = 0
    for v_of_new in Gnew.vertices():
        vID = Gnew.vertex_properties['label'][v_of_new]
        if not Gmerged.graph_properties['index_of'].has_key(vID):
            v_of_merged = Gmerged.add_vertex()
            Gmerged.graph_properties['index_of'][vID] = int(v_of_merged)
            Gmerged.vertex_properties['label'][v_of_merged] = vID
            if Gmerged.vertex_properties.has_key('No_of_occurrences') and Gnew.vertex_properties.has_key('No_of_occurrences'):
                Gmerged.vertex_properties['No_of_occurrences'][v_of_merged] = Gnew.vertex_properties['No_of_occurrences'][v_of_new]
            if Gmerged.vertex_properties.has_key('No_of_singleton_occurrences') and Gnew.vertex_properties.has_key('No_of_singleton_occurrences'):
                Gmerged.vertex_properties['No_of_singleton_occurrences'][v_of_merged] = Gnew.vertex_properties['No_of_singleton_occurrences'][v_of_new]

            nodes_added +=1
        else:
            v_of_merged_index = Gmerged.graph_properties['index_of'][vID]
            v_of_merged = Gmerged.vertex(v_of_merged_index)

            if Gmerged.vertex_properties.has_key('No_of_occurrences') and Gnew.vertex_properties.has_key('No_of_occurrences'):
                Gmerged.vertex_properties['No_of_occurrences'][v_of_merged] += Gnew.vertex_properties['No_of_occurrences'][v_of_new]
            if Gmerged.vertex_properties.has_key('No_of_singleton_occurrences') and Gnew.vertex_properties.has_key('No_of_singleton_occurrences'):
                Gmerged.vertex_properties['No_of_singleton_occurrences'][v_of_merged] += Gnew.vertex_properties['No_of_singleton_occurrences'][v_of_new]
    print 'Nodes added: ' + str(nodes_added)

    edges_added = 0
    for e_of_new in Gnew.edges():
        source_new = e_of_new.source()
        target_new = e_of_new.target()

        source_ID = Gnew.vertex_properties['label'][source_new]
        target_ID = Gnew.vertex_properties['label'][target_new]

        source_index_base = Gmerged.graph_properties['index_of'][source_ID]
        target_index_base = Gmerged.graph_properties['index_of'][target_ID]

        source_base = Gmerged.vertex(source_index_base)
        target_base = Gmerged.vertex(target_index_base)

        if Gmerged.edge(source_base,target_base) is None:
            e_merged = Gmerged.add_edge(source_base,target_base)
            Gmerged.edge_properties['co_oc'][e_merged] = Gnew.edge_properties['co_oc'][e_of_new]
            edges_added +=1
        else:
            Gmerged.edge_properties['co_oc'][Gmerged.edge(source_base,target_base)] += Gnew.edge_properties['co_oc'][e_of_new]

    print 'Edges added: ' + str(edges_added)

    if renormalise:calculate_SR(Gmerged)

    if check_consistency:are_cooccurrences_consistent_in_merged_graph(Gmerged,Gbase,Gnew)

    return Gmerged

def are_cooccurrences_consistent_in_merged_graph(Gmerged,Gbase,Gnew):
    ic = True

    for e in Gmerged.edges():

        co_oc_merged = Gmerged.edge_properties['co_oc'][e]

        v1 = e.source()
        v2 = e.target()

        v1_ID = Gmerged.vertex_properties['label'][v1]
        v2_ID = Gmerged.vertex_properties['label'][v2]

        node_pair_exists_in_Gbase = Gbase.graph_properties['index_of'].has_key(v1_ID) and Gbase.graph_properties['index_of'].has_key(v2_ID)
        node_pair_exists_in_Gnew = Gnew.graph_properties['index_of'].has_key(v1_ID) and Gnew.graph_properties['index_of'].has_key(v2_ID)

        if node_pair_exists_in_Gbase:
            try:
                n1_base_index = Gbase.graph_properties['index_of'][v1_ID]
                n2_base_index = Gbase.graph_properties['index_of'][v2_ID]
                n1_base = Gbase.vertex(n1_base_index)
                n2_base = Gbase.vertex(n2_base_index)
            except ValueError:
                print 'to err is human'
            if Gbase.edge(n1_base,n2_base) is None:
                co_oc_base = 0
            else:
                co_oc_base = Gbase.edge_properties['co_oc'][Gbase.edge(n1_base,n2_base)]
        else:
            co_oc_base = 0

        if node_pair_exists_in_Gnew:
            n1_new_index = Gnew.graph_properties['index_of'][v1_ID]
            n2_new_index = Gnew.graph_properties['index_of'][v2_ID]
            n1_new = Gnew.vertex(n1_new_index)
            n2_new = Gnew.vertex(n2_new_index)
            if Gnew.edge(n1_new,n2_new) is None:
                co_oc_new = 0
            else:
                co_oc_new = Gnew.edge_properties['co_oc'][Gnew.edge(n1_new,n2_new)]
        else:
            co_oc_new = 0

        ic_current = co_oc_merged == co_oc_new + co_oc_base
        if not ic_current:
            print '*** Problem with edge (' + v1_ID + ',' + v2_ID + '). Weight in merged: ' + str(co_oc_merged) + ', weight in base: ' + str(co_oc_base) + ', weight in new:' + str(co_oc_new)
        ic = ic and ic_current
        return ic

def get_common_vertices(G1,G2):
    vertex_set_1 = export_vertex_map_to_python_list(G1)
    vertex_set_2 = export_vertex_map_to_python_list(G2)

    return set([v for v in vertex_set_1 if v in vertex_set_2])

####

def is_vertex_connected_property_map(G):
    cmap = G.new_vertex_property('bool')
    for v in G.vertices():
        cmap[v] = is_vertex_connected(v)
    return cmap

def is_vertex_singleton_property_map(G):
    cmap = G.new_vertex_property('bool')
    for v in G.vertices():
        cmap[v] = is_vertex_singleton(v)

    return cmap


def add_number_of_singletons_graph_property(G):
    n=0
    for v in G.vertices():
       if is_vertex_singleton(v):n+=1

    G.graph_properties['number_of_singletons'] = G.new_graph_property('int')
    G.graph_properties['number_of_singletons'] = n


def get_modularity_from_dot_tree_via_gt(G,filename,weight = 'co_oc',tree_level = 'top'):
    if tree_level == 'top':
        cmap = mycomms.read_top_level_community_structure_from_dot_tree_to_gt_property_map(G,filename)
    else:
        cmap = mycomms.read_bottom_level_community_structure_from_dot_tree_to_gt_property_map(G,filename)
    Q = gtcom.modularity(G,cmap,G.edge_properties[weight])

    return Q


#### ASSORTATIVITY STUFF

def get_fraction_of_interclass_links(G,class_map = None):
    M = G.num_edges()

    if class_map is None:
        class_map = get_class_property_map_from_code_network(G)

    if M==0:
        return numpy.nan

    Mc = 0
    for e in G.edges():
        vertex_i = e.source()
        vertex_j = e.target()

        class_i = class_map[vertex_i]
        class_j = class_map[vertex_j]
        #label_i = G.vertex_properties['label'][vertex_i]
        #label_j = G.vertex_properties['label'][vertex_j]
        #
        #class_i = label_i.split('/')[0]
        #class_j = label_j.split('/')[0]

        Mc += class_i!=class_j

    return (1.*Mc)/M

def get_fraction_of_interclass_weights(G,weight = 'co_oc',class_map=None):
    M = 0

    if class_map is None:
        class_map = get_class_property_map_from_code_network(G)

    Mc = 0
    for e in G.edges():
        strength_ij =G.edge_properties[weight][e]
        M += strength_ij

        vertex_i = e.source()
        vertex_j = e.target()

        class_i = class_map[vertex_i]
        class_j = class_map[vertex_j]
        #
        #label_i = G.vertex_properties['label'][vertex_i]
        #label_j = G.vertex_properties['label'][vertex_j]
        #
        #class_i = label_i.split('/')[0]
        #class_j = label_j.split('/')[0]

        Mc += (class_i!=class_j)*strength_ij

    if M!=0:
        return (1.*Mc)/M
    else:
        return numpy.nan


def get_binary_assortativity_given_fraction_of_positives(eii):

    return (1 - 2*(eii*(1-eii))) / (1- eii*(1-eii))

###

def get_class_property_map_from_code_network(G):
    pmap = G.new_vertex_property('string')
    for v in G.vertices():
        vlabel = G.vertex_properties['label'][v]
        vclass = vlabel.split('/')[0]
        pmap[v] = vclass

    return pmap

def get_class_based_modularity_of_code_graph(G,pmap = None,weight = None):
    if pmap is None:
        pmap = get_class_property_map_from_code_network(G)

    if weight is None:
        return gtcom.modularity(G,pmap)
    else:
        return gtcom.modularity(G,pmap,G.edge_properties[weight])

######

def merge_vertex_filters(G,map1,map2):
    vmap = G.new_vertex_property('bool')
    for v in G.vertices():
        vmap[v] = map1[v] and map2[v]
    return vmap

def invert_edge_weight(G,emap):
    new_emap = G.new_edge_property('float')
    for e in G.edges():
        new_emap[e] = 1./emap[e]
    return new_emap

#####

def is_edge_interclass_property_map(G,class_map = None):
    pmap = G.new_edge_property('bool')

    if class_map is None:
        class_map = get_class_property_map_from_code_network(G)

    for e in G.edges():
        vertex_i = e.source()
        vertex_j = e.target()

        class_i = class_map[vertex_i]
        class_j = class_map[vertex_j]


        pmap[e] = class_i!=class_j
    return pmap

def is_edge_intraclass_property_map(G,class_map = None):
    return flip_edge_graph_filter(G,is_edge_interclass_property_map(G,class_map))

def flip_vertex_graph_filter(G,pmap):
    npmap = G.new_vertex_property('bool')
    for v in G.vertices():
        npmap[v] = not pmap[v]
    return npmap

def flip_edge_graph_filter(G,pmap):
    npmap = G.new_edge_property('bool')
    for e in G.edges():
        npmap[e] = not pmap[e]
    return npmap

def get_interclass_weights(G,class_map = None,ic_map = None,weight = 'co_oc'):
    if ic_map is None:
        ic_map = is_edge_interclass_property_map(G,class_map)

    return numpy.array([G.edge_properties[weight][e] for e in G.edges() if ic_map[e]])

def get_intraclass_weights(G,class_map = None,ic_map = None,weight = 'co_oc'):
    if ic_map is None:
        ic_map = flip_edge_graph_filter(G,is_edge_interclass_property_map(G,class_map))

    return numpy.array([G.edge_properties[weight][e] for e in G.edges() if ic_map[e]])

def get_inter_and_intraclass_weights(G,class_map = None,in_class_map = None,weight = 'co_oc'):
    if in_class_map is None:
        in_class_map = is_edge_intraclass_property_map(G,class_map)

    inter_class_weights = []
    intra_class_weights = []
    for e in G.edges():
        weight_e =G.edge_properties[weight][e]
        if in_class_map[e]:
            intra_class_weights.append(weight_e)
        else:
            inter_class_weights.append(weight_e)

    return numpy.array(inter_class_weights),numpy.array(intra_class_weights)

def is_member_of_largest_component_property_map(G):
    vmap = G.new_vertex_property('bool')
    for v in G.vertices():
        vmap[v] = G.vertex_properties['component_index'][v] == G.graph_properties['largest_component_index']

    return vmap

def get_fraction_of_interclass_strength_per_node(G,cmap = None, weight = 'co_oc'):
    if cmap is None:
        cmap = get_class_property_map_from_code_network(G)

    #if inter_class_edge_map is None:
    #    inter_class_edge_map = is_edge_interclass_property_map(G,cmap)

    N = G.num_vertices()
    IWF = numpy.zeros(N)
    i=-1
    for v in G.vertices():
        i+=1
        strenght_v = 0
        inter_class_strength_v = 0
        for u in v.out_neighbours():
            class_v = cmap[v]
            class_u = cmap[u]

            edge_v_u = G.edge(v,u)
            weight_v_u = G.edge_properties[weight][edge_v_u]
            strenght_v += weight_v_u

            if class_v == class_u:
                inter_class_strength_v += weight_v_u

            IWF[i] = (1.*inter_class_strength_v)/strenght_v
    return IWF

def get_number_of_codes_per_class_in_network(G,cmap=None):
    if cmap is None:
        cmap = get_class_property_map_from_code_network(G)

    class_hist = dict()
    for v in G.vertices():
        if class_hist.has_key(cmap[v]):
            class_hist[cmap[v]]+=1
        else:
            class_hist[cmap[v]]=1
    return class_hist

def get_assortative_mixing_matrix(G,vmap):
    unique_attribute_list = set([vmap[v] for v in G.vertices()])
    C = len(unique_attribute_list)

    unique_attribute_map = dict()
    for c in range(0,C):
        unique_attribute_map[unique_attribute_list[c]] = c

    E = numpy.zeros((C,C))
    for e in G.edges():

        att_of_i = vmap[e.source()]
        att_of_j = vmap[e.target()]

        att_index_i = unique_attribute_map[att_of_i]
        att_index_j = unique_attribute_map[att_of_j]

        E[att_index_i,att_index_j] +=1
        E[att_index_j,att_index_i] +=1*(att_of_i!=att_of_j)

    return E/(1.*G.num_edges())

def get_null_assortative_mixing_matrix(E):
    NULLmat = numpy.zeros(E.shape)
    a = E.sum(0)
    C = E.shape[0]

    for i in range(0,C):
        for j in range(i,C):
            NULLmat[i,j] = a[i]*a[j]
            NULLmat[j,i] = NULLmat[i,j] * (i!=j)

    return NULLmat

def get_vertex_attribute_mask(G,vmap,att_value):
    att_mask = G.new_vertex_property('bool')
    for v in G.vertices():
        att_mask[v] = vmap[v] == att_value
    return att_mask

def get_categorical_attribute_based_assortativity(G,pmap,weight = None):
    if weight is None:
        return gtcom.modularity(G,pmap)
    else:
        return gtcom.modularity(G,pmap,G.edge_properties[weight])


def get_comm_size_versus_assortative_mixing(G,base_filter,com_map,att_map,weight,min_comm_size=0):
    G.set_vertex_filter(base_filter)
    com_indices_list = [com_map[v] for v in G.vertices()]
    C = max(com_indices_list) # NUMBER OF COMMUNITIES
    histcom = numpy.zeros(C)
    for i in range(0,len(com_indices_list)):
        histcom[com_indices_list[i]-1] +=1

    assortativity = numpy.zeros(C)

    for c in range(1,max(com_indices_list)+1):
        G.set_vertex_filter(base_filter)
        com_mask = get_vertex_attribute_mask(G,com_map,c)
        vmask = merge_vertex_filters(G,base_filter,com_mask)
        G.set_vertex_filter(vmask)
        if G.num_vertices()<=min_comm_size:
            continue

        assortativity[c-1] = get_class_based_modularity_of_code_graph(G,att_map,weight)

    return histcom,assortativity

def get_comm_size_versus_fraction_of_homophilic_links(G,base_filter,com_map,att_map,min_comm_size=0):
    G.set_vertex_filter(base_filter)
    com_indices_list = [com_map[v] for v in G.vertices()]
    C = max(com_indices_list) # NUMBER OF COMMUNITIES
    histcom = numpy.zeros(C)
    for i in range(0,len(com_indices_list)):
        histcom[com_indices_list[i]-1] +=1

    frac_hom = numpy.zeros(C)

    for c in range(1,max(com_indices_list)+1):
        G.set_vertex_filter(base_filter)
        com_mask = get_vertex_attribute_mask(G,com_map,c)
        vmask = merge_vertex_filters(G,base_filter,com_mask)
        G.set_vertex_filter(vmask)
        if G.num_vertices()<=min_comm_size:
            continue

        frac_hom[c-1] = 1 - get_fraction_of_interclass_links(G,att_map)

    return histcom,frac_hom

def get_fraction_of_interclass_edges_versus_binned_weight(G,interclass_map,weight,bin_size):
    # 0: all
    # 1: interclass
    num_in_bin = dict()

    if weight == 'SR':
        bin_size = int(100*bin_size)
        multiplier = 100
    else:
        multiplier = 1
        bin_size = int(bin_size)

    for e in G.edges():
        w = G.edge_properties[weight][e]

        binned_w = int(w*multiplier)/bin_size

        if num_in_bin.has_key(binned_w):
            num_in_bin[binned_w][0]+=1
            num_in_bin[binned_w][1] += int(interclass_map[e])
        else:
            num_in_bin[binned_w]= [0,0]
            num_in_bin[binned_w][0]=1
            num_in_bin[binned_w][1] = int(interclass_map[e])

    return num_in_bin

def get_egonet_map_from_vertex_label_set(G,vlabel_set):
    ego_vmap = G.new_vertex_property('bool')
    ego_emap = G.new_edge_property('bool')
    for v in G.vertices():
        if G.vertex_properties['label'][v] in vlabel_set:
            ego_vmap[v] = True
            for u in v.out_neighbours():
                ego_vmap[u] = True
                ego_emap[G.edge(v,u)] = True

    return ego_vmap,ego_emap

def create_gt_graph_from_sparse_adjacency_matrix(A, is_directed = False, weight_type = None, list_of_vertex_labels = None):
    N = A.shape[0]
    A = A.tocoo()
    G = gt.Graph(directed=is_directed)

    if list_of_vertex_labels is not None:
        G.graph_properties['index_of'] = G.new_graph_property('python::object')
        G.graph_properties['index_of'] = dict()
        G.vertex_properties['label'] = G.new_vertex_property('string')
        for n in range(0,N):
            v = G.add_vertex()
            G.vertex_properties['label'][v] = list_of_vertex_labels[n]
            G.graph_properties['index_of'][list_of_vertex_labels[n]] = n
    else:
        G.add_vertex(N)

    if weight_type is not None:
        G.edge_properties['weight'] = G.new_edge_property(weight_type)

    for i,j,v in itertools.izip(A.row, A.col, A.data):
        if (not is_directed and i>j) or is_directed:
            e = G.add_edge(i,j)
            if weight_type is not None:
                G.edge_properties['weight'][e] = v

    return G

def has_vertex(G,v_index):
    try:
        aux = G.vertex(v_index)
        return True
    except ValueError:
        return False

def has_edge(G,v,u):
    try:
        aux = G.edge(v,u)
        return aux is not None
    except ValueError:
        return False

#######

def get_strength_entropy_property_map(G,weight = 'co_oc'):
    vmap = G.new_vertex_property('float')
    for v in G.vertices():
        if v.out_degree()==0:
            vmap[v] = numpy.nan
        else:
            strengths = []
            for e in v.out_neighbours():
                strengths.append(G.edge_properties[weight][e])
            strengths = numpy.array(strengths)
            strengths /= 1.*strengths.sum()
            vmap[v] = mystats.get_normalised_entropy(strengths)
    return vmap

def check_index_of_consistency(G):
    is_OK = True

    for vlabel in G.graph_properties['index_of'].keys():
        is_OK = is_OK and has_vertex(G,G.graph_properties['index_of'][vlabel])

    return is_OK

def check_label_consistency(G):
    is_OK = True

    for v in G.vertices():
        vlabel = G.vertex_properties['label'][v]
        is_OK = is_OK and int(v) == G.graph_properties['index_of'][vlabel]

    return is_OK

def are_nodes_same_class(G,v,u):
    label_v = G.vertex_properties['label'][v]
    label_u = G.vertex_properties['label'][u]

    class_v = label_v.split('/')[0]
    class_u = label_u.split('/')[0]

    return class_v == class_u

def get_strength_distribution_by_neighbour_type(G,v,weight_type='co_oc'):
    self_strength = G.vertex_properties['No_of_singleton_occurrences'][v]

    same_class_strength = 0
    diff_class_strength = 0

    for u in v.out_neighbours():
        same_class = are_nodes_same_class(G,v,u)
        e = G.edge(v,u)
        connection_strength = G.edge_properties[weight_type][e]

        if same_class:
            same_class_strength += connection_strength
        else:
            diff_class_strength += connection_strength

    return {'self_strength':self_strength,'same_class_strength':same_class_strength,'diff_class_strength':diff_class_strength}

def get_adjacency_matrix_from_gt_graph(G,weight = 'co_oc'):
    xs = []
    ys = []
    vals = []

    label_lookup = dict()

    for e in G.edges():
        x = int(e.source())
        y = int(e.target())
        v = G.edge_properties[weight][e]
        labelx = G.vertex_properties['label'][e.source()]
        labely = G.vertex_properties['label'][e.target()]

        if not label_lookup.has_key(labelx):
            label_lookup[labelx] = x
        if not label_lookup.has_key(labely):
            label_lookup[labely] = y

        xs.append(x)
        ys.append(y)
        vals.append(v)


    A = scipy.sparse.coo_matrix((vals,(xs,ys)),shape=(G.num_vertices(),G.num_vertices()))
    A = A.tocsc()

    return A,label_lookup

def get_filter_given_node_name_list(G,vertex_list):
    vmap = G.new_vertex_property('bool')
    for v_name in vertex_list:
        try:
            v_index = G.graph_properties['index_of'][v_name]
            v = G.vertex(v_index)
            vmap[v] = True
        except KeyError:
            continue
    return vmap