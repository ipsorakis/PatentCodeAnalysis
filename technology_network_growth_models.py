__author__ = 'yannis'

import graph_tool as gt
import my_graph_tool_add_ons as mygt
import math
import numpy
import scipy
import my_community_tools as mycomms

def synapse_reinforcement(Gbase, No_of_incoming_patents, avg_no_codes_per_patent, jump_rate):
    G = gt.Graph(Gbase)

    for pat_iter in range(0,No_of_incoming_patents):

        # draw a number of codes per patent
        nt = numpy.random.poisson(avg_no_codes_per_patent)
        if nt<2:continue

        # draw a random node(tech) from the graph - uniformly for now
        i = numpy.random.randint(0,G.num_vertices())

        for visit_iter in range(0,nt):
            # flip a coin and jump with probability jump_rate
            do_jump = numpy.random.binomial(1,jump_rate,1)
            if do_jump:
                j = numpy.random.randint(0,G.num_vertices())
            # if not, pick a neighbour (uniformly at random for now)
            else:
                neighbours_of_i = [int(v) for v in G.vertex(i).out_neighbours()]
                aux = numpy.random.randint(0,len(neighbours_of_i))
                j = neighbours_of_i[aux]
            # increment cooccurrence
            if G.edge(G.vertex(i),G.vertex(j)) is None:
                G.add_edge(G.vertex(i),G.vertex(j))
            G.edge_properties['co_oc'][G.edge(G.vertex(i),G.vertex(j))] +=1

            if not do_jump:i = j
    return G

#def class_reinforcement(Gbase, No_of_incoming_patents, avg_no_codes_per_patent, jump_rate):
#    return None

def small_world_arrivals(Bbase, No_of_incoming_patents, avg_no_codes_per_patent, jump_rage):
    return None