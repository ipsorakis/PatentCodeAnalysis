__author__ = 'yannis'

import Parsers_Decade as PD
import graph_tool as gt
import my_graph_tool_add_ons as mygt
import pandas
import my_tree_tools as mytree
import my_community_tools as mycomms
import my_containers
import cPickle
import Bayesian_Non_Parametrics as BNP
import numpy
import my_stat_tools as mystat
import my_containers as mycons
import BOMP
import matplotlib.pylab as pylab




#AdjList = PD.get_patent_code_incidence_matrix_from_multiple_files('Patent_files/Patents_v2',range(1790,1810,10),True)
#node_label_list = [None]*len(lookup.keys())
#for elem in lookup.keys():
#    node_label_list[lookup[elem]] = elem
#B = mycons.convert_list_to_sparse_matrix(AdjList)

#test = PD.get_adjacency_frames_CP_class_groups(range(1790,1820,10))

#G = gt.load_graph('Network_files/Gclasses_1790.xml.gz')
#remove_set = ['423','082']
#mygt.safe_delete_vertices_based_on_label(G,remove_set)

#t = mytree.tree('c')
#t.add_node('d','c')
#t.add_node('g','d')
#t.add_node('h','d')
#t.add_node('b','c')
#t.add_node('i','b')
#t.add_node('a','b')
#t.add_node('e','a')
#t.add_node('f','a')
#
##x = t.depth_first_search('c')
##print [n.name for n in x]
#
#print t.are_ancestor_descendant_pair('i','b')

#t = mytree.parse_Daniel_semicolon_based_tree_format_to_my_tree_class('tree_data/daniel_sample.txt')

#G = gt.load_graph('Gclasses_1830.xml.gz')
#A = mygt.get_adjacency_matrix_from_gt_graph(G)

#Gclasses_merged = gt.load_graph('Gclasses_1790.xml.gz')
#Gcodes_merged = gt.load_graph('Gcodes_1790.xml.gz')
#for d in range(1800,2020,10):
#    print '*** Processing decade {0}...'.format(d)
#    print '*Creating current graphs...'
#    Gclasses_current,Gcodes_current = PD.load_coocurrence_networks_from_patent_code_file_to_graph_tool('Patents_v2_{0}.csv'.format(d))
#
#    print 'Saving current graphs...'
#    Gclasses_current.save('Gclasses_' + str(d) + '.xml.gz')
#    Gcodes_current.save('Gcodes_' + str(d) + '.xml.gz')
#
#    print '*Merging decade {0} with previous...'.format(d)
#    print 'Classes:'
#    Gclasses_merged = mygt.merge_cooccurrence_networks(Gclasses_merged,Gclasses_current,True,True)
#    print 'Codes:'
#    Gcodes_merged = mygt.merge_cooccurrence_networks(Gcodes_merged,Gcodes_current,True,True)
#
#    print 'Saving merged graphs...'
#    Gclasses_merged.save('Gclasses_{0}to{1}.xml.gz'.format(1790,d))
#    Gcodes_merged.save('Gcodes_{0}to{1}.xml.gz'.format(1790,d))

#B = PD.get_patent_code_incidence_matrix_from_file('Patents_v2_1790.csv')
#B.get_correlation_column_elem_age_vs_degree()

#INNOV = PD.get_innovation_coordinates('Patents_v2_1790.csv',use_codes=True)

#
# aux = mystat.time_series_flatness([numpy.nan,numpy.nan,5,3,4])

#AdjList,lookup = PD.get_patent_code_incidence_matrix_from_file('Patents_v2_1790.csv',True)
#B = mycons.convert_list_to_sparse_matrix(AdjList)

#PD.get_patent_code_incidence_matrix_from_file('Patents_v2_1790.csv',True)

#PD.filter_patents_decade_given_code_set(True,{'D11/079000'},'CD11',decade_range = range(1900,1910,10),primary_only=True)

#Photoelectric_codes = cPickle.load(open('Photoelectric_codes.cpickle','rb'))
#PD.filter_patents_decade_given_code_set(True,Photoelectric_codes,'PV',decade_range = range(1900,2020,10),primary_only=True)


#node_data = pandas.read_csv('PV_CODES_1790to2010.csv')
#PV_nodes = node_data[node_data.Class=='136']
#
#G_PV_COOC = dict()
#G_PV_SR = dict()
#for d in range(1900,2020,10):
#    print 'Loading decade {0}...'.format(d)
#    G_PV_COOC[d] = mygt.load_graph_from_pajek('G_PE_{0}_COOC.net'.format(d),weight_label='co_oc',weight_type='int')
#    G_PV_SR[d] = mygt.load_graph_from_pajek('G_PE_{0}_SR.net'.format(d),weight_label='SR',weight_type='float')
#
#G_PV_merged_COOC = dict()
#
#G_PV_merged_COOC[1900] = G_PV_COOC[1900]
#
#for d in range(1910,2020,10):
#    print 'processing decade {0}...'.format(d)
#    G_PV_merged_COOC[d] = mygt.merge_cooccurrence_networks(G_PV_merged_COOC[d-10],G_PV_COOC[d],False,True)
#    Gcodes = gt.load_graph('Gcodes_1790to{0}.xml.gz'.format(d))
#    G = G_PV_merged_COOC[d]
#    G.vertex_properties['No_of_occurrences'] = G.new_vertex_property('int')
#    for pv_label in PV_nodes.Label:
#        if G.graph_properties['index_of'].has_key(pv_label):
#            no_of_occurrences = Gcodes.vertex_properties['No_of_occurrences'][Gcodes.vertex(G.graph_properties['index_of'][pv_label])]
#            G.vertex_properties['No_of_occurrences'][G.vertex(G.graph_properties['index_of'][pv_label])] = no_of_occurrences
#


#mygt.load_graph_from_pajek('G_PE_{0}_COOC.net'.format(1900),False,weight_label='co_oc',weight_type='int')

#PV_codes = PD.read_PV_codes_per_decade('PV_patents.csv',range(1970,1980,10))

#PV = PD.read_PV_patents('PV_patents.csv')

#comms_full = mycomms.read_bottom_level_community_structure_from_dot_tree_to_community_list('Gcodes_1800_COOC.tree')

#t1 = numpy.array([1,2,3])
#t2 = numpy.array([0,1,numpy.nan])
#
#c = mystat.normalised_cross_correlation(t1,t2)

#Z = BNP.iBT_random_sample(20,2)

#PD.load_coocurrence_networks_from_patent_code_file_to_graph_tool('Patents_v2_1840.csv')


#PD.split_Patent_Codes_to_decades('Patents_v2.csv')

#Z = BNP.iBT_random_sample(20,2,2)

#Gcl,Gco = PD.load_coocurrence_networks_from_file_to_graph_tool('code_pairs_1830.csv')
#G2 = mygt.get_graph_without_singletons(Gco)

#PD = reload(PD)
#for d in range(1790,2020,10):
#    print '*** Processing activity of the ' + str(d) + 's...'
#    Rclasses = PD.get_patent_to_technology_matrix_from_file('PatentCodes_{0}.csv'.format(d),use_codes=False)
#    cPickle.dump(Rclasses['B'],open('Rclasses_B_' + str(d) + '.cpickle','wb'))
#    cPickle.dump(Rclasses['Patents'],open('Rclasses_Patents_' + str(d) + '.cpickle','wb'))
#    cPickle.dump(Rclasses['Techs'],open('Rclasses_Techs_' + str(d) + '.cpickle','wb'))
#
#    Rcodes = PD.get_patent_to_technology_matrix_from_file('PatentCodes_{0}.csv'.format(d),use_codes = True)
#    cPickle.dump(Rcodes['B'],open('Rcodes_B_' + str(d) + '.cpickle','wb'))
#    cPickle.dump(Rcodes['Patents'],open('Rcodes_Patents_' + str(d) + '.cpickle','wb'))
#    cPickle.dump(Rcodes['Techs'],open('Rcodes_Techs_' + str(d) + '.cpickle','wb'))

#Gtest = mygt.merge_cooccurrence_networks(gt.load_graph('Gclasses_1790.xml.gz'),gt.load_graph('Gclasses_1800.xml.gz'),True,True)

#g = mycomm.read_community_structure_from_dot_map_to_community_list('Gclasses_2010_COOC.map')

##mycomm.read_bottom_level_community_structure_from_dot_tree('Gclasses_1810_COOC.tree')
#decade_range = range(1790,2020,10)
#print '*** opening files...'
#COMMs_SR = PD.read_bottom_level_decade_community_structures_from_dot_tree_files('Gcodes_','_SR',decade_range)
#COMMs_COOC = PD.read_bottom_level_decade_community_structures_from_dot_tree_files('Gcodes_','_COOC',decade_range)
#
#NMI_SR = []
#NMI_COOC = []
#for d in decade_range:
#    print '*** processing decade ' + str(d) + '...'
#    g_sr = COMMs_SR[d]
#    g_cooc = COMMs_COOC[d]
#
#    nmi_sr,class_hierarchy = mycomm.get_normalised_mutual_information_between_code_communities_and_USPTO_hierarchy(g_sr)
#    nmi_cooc,class_hierarchy = mycomm.get_normalised_mutual_information_between_code_communities_and_USPTO_hierarchy(g_cooc)
#
#    NMI_COOC.append(nmi_cooc)
#    NMI_SR.append(nmi_sr)
#
#    print 'NMI_COOC: {0:.2f}, NMI_SR: {1:.2f}'.format(nmi_cooc,nmi_sr)

#print NMI

#
#g1 = [[1,2],3]
#g2 = [[1,2,3],[4,5]]
#
#mycomm.get_community_substructure_from_node_intersect(g1,g2,set(range(1,4)))
#
#print str(mygt.get_normalised_mutual_information(g1,g2))


#G = PD.merge_decade_graphs(1790,2010)


#decade_range = range(1790,2020,10)
#
#for decade in decade_range:
#    print '*** Processing the ' + str(decade) + 's...'
#    Gclasses,Gcodes = PD.load_coocurrence_networks_from_file_to_graph_tool('code_pairs_'+str(decade)+'.csv')
#    print 'saving graphs...'
#    Gclasses.save('Gclasses_'+str(decade)+'.xml.gz')
#    Gcodes.save('Gcodes_'+str(decade)+'.xml.gz')
#
#    print('normalising edge weights...')
#    Gclasses = mygt.calculate_SR(Gclasses)
#    Gcodes = mygt.calculate_SR(Gcodes)
#    print 're-saving graphs...'
#    Gclasses.save('Gclasses_'+str(decade)+'.xml.gz')
#    Gcodes.save('Gcodes_'+str(decade)+'.xml.gz')
#PD.split_Patent_Codes_to_decades('patents_sample.csv')


#decades = range(1830,2000,10)
#
#for d in decades:
#    print '*** processing decade ' + str(d) + '...'
#    Gclasses = gt.load_graph('Gclasses_' + str(d) + '.xml.gz')
#    Gcodes = gt.load_graph('Gcodes_' + str(d) + '.xml.gz')
#
#    for e in Gclasses.edges():
#        Gclasses.edge_properties['co_oc'][e] /= 2
#    Gclasses.save('Gclasses_'+str(d) + '.xml.gz')
#
#    for e in Gcodes.edges():
#        Gcodes.edge_properties['co_oc'][e] /=2
#    Gcodes.save('Gcodes_'+str(d) + '.xml.gz')
#
#
#
#G1830 = gt.load_graph('Gclasses_1830.xml.gz')
#
#tree_string = convert_dot_tree_to_tree_string('Gclasses_1830.tree')

#Decades = range(1830,2020,10)
#
#for decade in Decades:
#    filename = 'code_pairs_'+str(decade)+'.csv'
#    print '******* Reading file ' + filename + '...'
#    Gclasses,Gcodes = PD.load_coocurrence_networks_from_file_to_graph_tool(filename)
#
#    print 'saving to disk...'
#    Gclasses.save('Gclasses_'+str(decade)+'.xml.gz')
#    Gcodes.save('Gcodes_'+str(decade)+'.xml.gz')