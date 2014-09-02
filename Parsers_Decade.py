__author__ = 'yannis'

import graph_tool as gt
import my_graph_tool_add_ons as mygt
import my_community_tools as mycomms
import numpy
import scipy
import cPickle as cpickle
import my_containers as mycons
import scipy.sparse as sparse
import my_stat_tools as mystat
import pandas
import re
import BOMP

def split_Patent_Codes_to_decades(filename = 'PatentCodes.csv', decade_range = range(1790,2020,10), Patents = None, last_read_line = 1):
    print 'opening Patents lookup table...'

    #with open('Patents.cpickle','rb') as fid:
    #    Patents = cpickle.load(fid)
    #print 'done.'

    file_parts = filename.split('.')

    Decade_filenames = dict(zip(decade_range,[file_parts[0] + '_' + str(i) + '.csv' for i in decade_range]))

    for d in Decade_filenames:
        with open(Decade_filenames[d],'w') as fp:
            fp.write('Pat_Type,Patent,Primary,Class,Subclass,Type,GDate,AppDate,Appyear\n')


    print 'opening ' + filename + '...'
    with open(filename) as fid:
        print 'done. Iterating through contents...'
        num_lines = 0
        for aline in fid:
            num_lines +=1
            if num_lines <= last_read_line:
                continue

            elems = aline.split(',')
            #   Pat_Type,Patent,Primary,Class,Subclass,Type,GDate,AppDate,Appyear
            patentID = elems[0]+elems[1]

            if Patents is not None:
                if not Patents.has_key(patentID):continue
                year = Patents[patentID]
            else:
                year = None
                if elems[-1] !='':
                    year = int(elems[-1])

                if elems[-2] !='':
                    AppDate = elems[-2].split('/')
                    year = int(AppDate[-1])

                if elems[-3] !='':
                    GDate = elems[-3].split('/')
                    year = int(GDate[-1])

            if year is None:continue

            patent_year = str(year)
            patent_decade = int(patent_year[0:-1] + '0')

            if patent_decade in decade_range:
                with open(Decade_filenames[patent_decade],'a') as fid:
                    fid.write(aline)

                #if num_lines % 1000000 == 0:
                #    print 'line: {0}, No. of patents: {1}'.format(num_lines,len(Patents.keys()))

    print 'Successfully read ' + str(num_lines) + ' lines.'

def split_All_Code_Pairs_to_decades(filename = 'All_Code_Pairs.csv',decade_range = range(1790,2020,10),last_read_line = 1):
    print 'opening Patents lookup table...'
    
    with open('Patents.cpickle','rb') as fid:
        Patents = cpickle.load(fid)
    print 'done.'
    
    Decade_filenames = dict(zip(decade_range,['code_pairs_' + str(decade) + '.csv' for decade in decade_range]))
    
    if last_read_line == 1:    
        print 'initialising decade files...'
        for decade in Decade_filenames.keys():
            with open(Decade_filenames[decade],'w') as fid:
                fid.write('Pat_Type,Patent,Class,Subclass,N1,Class2,subclass2,N2\n')
        print 'done.'
    
    print 'opening master file...'
    with open(filename) as fid:
        print 'done. Iterating through contents...'
        num_lines = 0
        for aline in fid:
            num_lines +=1
            if num_lines <= last_read_line:
                continue
    
            elems = aline.split(',')
            #   0 'Pat_Type', 1 'Patent', 2 'Class', 3 'Subclass', 4 'N1', 5 'Class2', 6 'subclass2', 7 'N2'
            patentID = elems[0]+elems[1]
    
            if not Patents.has_key(patentID):
                continue
    
            patent_year = str(Patents[patentID])
            patent_decade = int(patent_year[0:-1] + '0')
    
            if patent_decade in decade_range:
                with open(Decade_filenames[patent_decade],'a') as fid:
                    fid.write(aline)
                if num_lines % 1000000 == 0:print str(num_lines)

    print 'Successfully read ' + str(num_lines) + ' lines.'

def top_up_split_All_Code_Pairs_to_decades(file_name):
    print 'opening Patents lookup table...'

    with open('Patents.cpickle','rb') as fid:
        Patents = cpickle.load(fid)
    print 'done.'

    Decade_filenames = dict(zip(range(1830,2020,10),['' for i in range(1830,2020,10)]))

    for decade in Decade_filenames.keys():
        Decade_filenames[decade] = 'code_pairs_' + str(decade) + '.csv'

    print 'opening ' + file_name + '...'
    with open(file_name) as fid:
        print 'Iterating through contents...'
        num_lines = 0
        for aline in fid:
            num_lines +=1

            elems = aline.split(',')
            #   0 'Pat_Type', 1 'Patent', 2 'Class', 3 'Subclass', 4 'N1', 5 'Class2', 6 'subclass2', 7 'N2'
            patentID = elems[0]+elems[1]

            if not Patents.has_key(patentID):
                continue

            patent_year = str(Patents[patentID])
            patent_decade = int(patent_year[0:-1] + '0')

            if Decade_filenames.has_key(patent_decade):
                with open(Decade_filenames[patent_decade],'a') as fid:
                    fid.write(aline)
            else:
                print 'skipping unknown decade: ' + str(decade)

    print 'Successfully read ' + str(num_lines) + ' lines.'

def load_patent_codes_from_csv_to_dict(datafile):
    """
    0.Pat_Type,1.Patent,2.Primary,3.Class,4.Subclass,5.Type,6.GDate,7.AppDate,8.Appyear

    cd Documents/PatentCodes.zip\ Folder
    """
    Patents = dict()

    with open(datafile,'r') as fid:
        line_count  =0
        for aline in fid:
            # skip the column headers
            line_count +=1
            if line_count ==1 : continue
            # start with second line

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]

            if not Patents.has_key(patentID):
                year = None

                if entry[-1] !='':
                    year = int(entry[-1])

                if year is None and entry[-2] !='':
                    AppDate = entry[-2].split('/')
                    year = int(AppDate[-1])

                if year is None and entry[-3] !='':
                    GDate = entry[-3].split('/')
                    year = int(GDate[-1])

                if year is not None:
                    Patents[patentID] = year

            if line_count % 500000 == 0:
                print 'line: {0}, No. of patents: {1}'.format(line_count,len(Patents.keys()))

    return Patents

def find_newly_introduced_technologies_per_decade(decade):
    if decade !=1790:
        Gcodes = gt.load_graph('Gcodes_1790to{0}.xml.gz'.format(decade))
        Gclasses = gt.load_graph('Gclasses_1790to{0}.xml.gz'.format(decade))
    else:
        Gcodes = gt.load_graph('Gcodes_1790.xml.gz')
        Gclasses = gt.load_graph('Gclasses_1790.xml.gz')

    with open('code_pairs_{0}.csv'.format(decade)) as fid:
        num_lines = 0
        for aline in fid:
            num_lines+=1
            if num_lines==1:continue
            aline = aline.strip()
            entries = aline.split(',')
            N1 = int(entries[4])
            N2 = int(entries[7])

    #    TO BE FINISHED

def load_coocurrence_networks_from_patent_code_file_to_graph_tool(datafile,Gclasses = None,Gcodes = None, normalise_weights = True):

    def close_patent_batch():
            # *** NODE PROCESSING ***
            # iterate through each code:
            for aclass_label in current_classes:
                # check if node of class1 exists:
                if not Gclasses.graph_properties['index_of'].has_key(aclass_label):
                    aclass_vertex = Gclasses.add_vertex()
                    aclass_index = int(aclass_vertex)
                    Gclasses.vertex_properties['label'][aclass_vertex] = aclass_label
                    Gclasses.graph_properties['index_of'][aclass_label] = aclass_index
                # else just find its index/object pointer
                else:
                    aclass_index = Gclasses.graph_properties['index_of'][aclass_label]
                    aclass_vertex = Gclasses.vertex(aclass_index)
                # using the vertex object increment its number of patents:
                Gclasses.vertex_properties['No_of_occurrences'][aclass_vertex] +=1

            for acode_label in current_codes:
                if not Gcodes.graph_properties['index_of'].has_key(acode_label):
                    acode_vertex = Gcodes.add_vertex()
                    acode_index = int(acode_vertex)
                    Gcodes.vertex_properties['label'][acode_vertex] = acode_label
                    Gcodes.graph_properties['index_of'][acode_label] = acode_index
                else:
                    acode_index = Gcodes.graph_properties['index_of'][acode_label]
                    acode_vertex = Gcodes.vertex(acode_index)

                # using the vertex object increment its number of patents:
                Gcodes.vertex_properties['No_of_occurrences'][acode_vertex] +=1

            # =================== EDGE PROCESSING ===================
            # first check if there is only one class:
            if len(current_classes)>1:
                class_list = [aclass for aclass in current_classes]
                N = len(class_list)
                # then for each unique pair of classes,
                for i in range(0,N-1):
                    for j in range(i+1,N):
                        class_i_label = class_list[i]
                        class_i_vertex = Gclasses.graph_properties['index_of'][class_i_label]
                        class_j_label = class_list[j]
                        class_j_vertex = Gclasses.graph_properties['index_of'][class_j_label]

                        # check if edge exists:
                        if not Gclasses.edge(class_i_vertex,class_j_vertex):
                            # if not, create it
                            edge_i_j = Gclasses.add_edge(class_i_vertex,class_j_vertex)
                        else:
                            edge_i_j = Gclasses.edge(class_i_vertex,class_j_vertex)

                        # increment the edge weight
                        Gclasses.edge_properties['co_oc'][edge_i_j] +=1

            # count self-appearances
            elif len(current_classes)==1:
                class_label = current_classes.pop()
                class_index = Gclasses.graph_properties['index_of'][class_label]
                class_vertex = Gclasses.vertex(class_index)
                Gclasses.vertex_properties['No_of_singleton_occurrences'][class_vertex]+=1

            # repeat the process for codes
            # first check if there is only one code:
            if len(current_codes)>1:
                code_list = [acode for acode in current_codes]
                N = len(code_list)
                # then for each unique pair of codes,
                for i in range(0,N-1):
                    for j in range(i+1,N):
                        code_i_label = code_list[i]
                        code_j_label = code_list[j]

                        code_i_vertex = Gcodes.graph_properties['index_of'][code_i_label]
                        code_j_vertex = Gcodes.graph_properties['index_of'][code_j_label]

                        # check if edge exists:
                        if not Gcodes.edge(code_i_vertex,code_j_vertex):
                            # if not, create it
                            edge_i_j = Gcodes.add_edge(code_i_vertex,code_j_vertex)
                        else:
                            edge_i_j = Gcodes.edge(code_i_vertex,code_j_vertex)

                        # increment the edge weight
                        Gcodes.edge_properties['co_oc'][edge_i_j] +=1

            # count self-appearances
            elif len(current_codes)==1:
                code_label = current_codes.pop()
                code_index = Gcodes.graph_properties['index_of'][code_label]
                code_vertex = Gcodes.vertex(code_index)
                Gcodes.vertex_properties['No_of_singleton_occurrences'][code_vertex]+=1

    if Gclasses is None:
        Gclasses = gt.Graph(directed=False)
        Gclasses.graph_properties['total_cooc'] = Gclasses.new_graph_property('int',0)
        Gclasses.graph_properties['total_patents'] = Gclasses.new_graph_property('int',0)

        Gclasses.graph_properties['index_of'] = Gclasses.new_graph_property('object')
        Gclasses.graph_properties['index_of'] = dict()

        Gclasses.vertex_properties['label'] = Gclasses.new_vertex_property('string')
        Gclasses.vertex_properties['No_of_occurrences'] = Gclasses.new_vertex_property('int')
        Gclasses.vertex_properties['No_of_singleton_occurrences'] = Gclasses.new_vertex_property('int')

        Gclasses.edge_properties['co_oc'] = Gclasses.new_edge_property('int')

    if Gcodes is None:
        Gcodes = gt.Graph(directed=False)
        Gcodes.graph_properties['total_cooc'] = Gcodes.new_graph_property('int',0)
        Gcodes.graph_properties['total_patents'] = Gcodes.new_graph_property('int',0)

        Gcodes.graph_properties['index_of'] = Gcodes.new_graph_property('object')
        Gcodes.graph_properties['index_of'] = dict()

        Gcodes.vertex_properties['label'] = Gcodes.new_vertex_property('string')
        Gcodes.vertex_properties['No_of_occurrences'] = Gcodes.new_vertex_property('int')
        Gcodes.vertex_properties['No_of_singleton_occurrences'] = Gcodes.new_vertex_property('int')

        Gcodes.edge_properties['co_oc'] = Gcodes.new_edge_property('int')

    print 'opening ' + datafile + '...'
    with open(datafile,'r') as fid:
        total_patents = 0

        previous_patent = 'N/A'
        current_classes = set()
        current_codes = set()

        print 'reading contents...'
        curr_line = 0
        for aline in fid:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]
            current_class_label = entry[3]
            current_code_label = entry[3] + '/' + entry[4]

            # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
            if patentID == previous_patent:
                # =================== IN A PATENT BATCH ===================
                current_classes.add(current_class_label)
                current_codes.add(current_code_label)
            else:
                # =================== CLOSING A PATENT BATCH ===================
                if len(current_classes)!=0 and len(current_codes)!=0:
                    # A) INCREMENT PATENTS
                    total_patents+=1
                    # B) GATHER NODES AND LINKS
                    close_patent_batch()
                    # C) CLOSE CLASS / NODE SETS
                current_classes = set()
                current_classes.add(current_class_label)
                current_codes = set()
                current_codes.add(current_code_label)
                previous_patent = patentID

            if curr_line % 1000000 == 0:
                print 'current line: {0}, Class nodes/edges: {1}/{2}, Code nodes/edges {3}/{4}'.format(curr_line,Gclasses.num_vertices(),Gclasses.num_edges(),Gcodes.num_vertices(),Gcodes.num_edges())

        # close the patent patch for the remaining entries
        if len(current_classes)>0 or len(current_codes)>0:
            close_patent_batch()

    if normalise_weights:
        print 'normalising weights for classes...'
        Gclasses = mygt.calculate_SR(Gclasses)
        print 'normalising weights for codes...'
        Gcodes = mygt.calculate_SR(Gcodes)

    Gclasses.graph_properties['total_patents'] = total_patents
    Gcodes.graph_properties['total_patents'] = total_patents

    mygt.add_number_of_singletons_graph_property(Gclasses)
    mygt.add_number_of_singletons_graph_property(Gcodes)
    return Gclasses,Gcodes

def load_coocurrence_networks_from_code_cooc_file_to_graph_tool(datafile,Gclasses = None,Gcodes = None, allow_singletons = True, normalise_weights = True):
    """
    0 'Pat_Type', 1 'Patent', 2 'Class', 3 'Subclass', 4 'N1', 5 'Class2', 6 'subclass2', 7 'N2'
    """
    if Gclasses is None:
        Gclasses = gt.Graph(directed=False)
        Gclasses.graph_properties['total_cooc'] = Gclasses.new_graph_property('int',0)
        Gclasses.graph_properties['total_patents'] = Gclasses.new_graph_property('int',0)

        Gclasses.graph_properties['index_of'] = Gclasses.new_graph_property('object')
        Gclasses.graph_properties['index_of'] = dict()

        Gclasses.vertex_properties['label'] = Gclasses.new_vertex_property('string')
        Gclasses.vertex_properties['No_of_occurrences'] = Gclasses.new_vertex_property('int')

        Gclasses.edge_properties['co_oc'] = Gclasses.new_edge_property('int')

    if Gcodes is None:
        Gcodes = gt.Graph(directed=False)
        Gcodes.graph_properties['total_cooc'] = Gcodes.new_graph_property('int',0)
        Gcodes.graph_properties['total_patents'] = Gcodes.new_graph_property('int',0)

        Gcodes.graph_properties['index_of'] = Gcodes.new_graph_property('object')
        Gcodes.graph_properties['index_of'] = dict()

        Gcodes.vertex_properties['label'] = Gcodes.new_vertex_property('string')
        Gcodes.vertex_properties['No_of_occurrences'] = Gcodes.new_vertex_property('int')

        Gcodes.edge_properties['co_oc'] = Gcodes.new_edge_property('int')

    print 'opening ' + datafile + '...'
    with open(datafile,'r') as fid:
        curr_line = 0
        print 'reading contents...'
        for aline in fid:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            N1 = int(entry[4])
            N2 = int(entry[7])
            if N1>N2:
                continue
            #patentID = entry[1]
            # This file contains only the co-occurrences of patents that exist in Patents.cpickle
            ############################################################################

            class1_label = entry[2]
            class2_label = entry[5]

            if class1_label != class2_label:

                # check if node of class1 exists:
                if not Gclasses.graph_properties['index_of'].has_key(class1_label):
                    class1_vertex = Gclasses.add_vertex()
                    class1_index = int(class1_vertex)
                    Gclasses.vertex_properties['label'][class1_vertex] = class1_label
                    Gclasses.graph_properties['index_of'][class1_label] = class1_index
                else:
                    class1_index = Gclasses.graph_properties['index_of'][class1_label]
                    class1_vertex = Gclasses.vertex(class1_index)

                # check if node of class2 exists:
                if not Gclasses.graph_properties['index_of'].has_key(class2_label):
                    class2_vertex = Gclasses.add_vertex()
                    class2_index = int(class2_vertex)
                    Gclasses.vertex_properties['label'][class2_vertex] = class2_label
                    Gclasses.graph_properties['index_of'][class2_label] = class2_index
                else:
                    class2_index = Gclasses.graph_properties['index_of'][class2_label]
                    class2_vertex = Gclasses.vertex(class2_index)

                #---------------------------------------------------------------------------
                # check if edge between class1 and class2 exists:
                if not Gclasses.edge(class1_vertex,class2_vertex):
                    edge_class1_class2 =  Gclasses.add_edge(class1_vertex,class2_vertex)
                else:
                    edge_class1_class2 = Gclasses.edge(class1_vertex,class2_vertex)

                #
                Gclasses.edge_properties['co_oc'][edge_class1_class2] +=1

                Gclasses.vertex_properties['No_of_occurrences'][class1_vertex] +=1
                Gclasses.vertex_properties['No_of_occurrences'][class2_vertex] +=1

                Gclasses.graph_properties['total_cooc'] +=1
                #Gclasses.graph['total_patents'] +=1

                #Gclasses.node[class1]['No_of_Patents'] +=1
                #Gclasses.node[class2]['No_of_Patents'] +=1
            elif class1_label == class2_label and allow_singletons:
                # check if node of class1 exists and if not, add it:
                if not Gclasses.graph_properties['index_of'].has_key(class1_label):
                    class1_vertex = Gclasses.add_vertex()
                    class1_index = int(class1_vertex)
                    Gclasses.vertex_properties['label'][class1_vertex] = class1_label
                    Gclasses.graph_properties['index_of'][class1_label] = class1_index
                else:
                    class1_index = Gclasses.graph_properties['index_of'][class1_label]
                    class1_vertex = Gclasses.vertex(class1_index)
                Gclasses.vertex_properties['No_of_occurrences'][class1_vertex] +=1
            ############################################################################

            code1_label = entry[2] + '/' + entry[3]
            code2_label = entry[5] + '/' + entry[6]

            if code1_label != code2_label:
                # check if node with code1 exists:
                if not Gcodes.graph_properties['index_of'].has_key(code1_label):
                    code1_vertex = Gcodes.add_vertex()
                    code1_index = int(code1_vertex)
                    Gcodes.vertex_properties['label'][code1_vertex] = code1_label
                    Gcodes.graph_properties['index_of'][code1_label] = code1_index
                else:
                    code1_index = Gcodes.graph_properties['index_of'][code1_label]
                    code1_vertex = Gcodes.vertex(code1_index)

                # check if node with code2 exists:
                if not Gcodes.graph_properties['index_of'].has_key(code2_label):
                    code2_vertex = Gcodes.add_vertex()
                    code2_index = int(code2_vertex)
                    Gcodes.vertex_properties['label'][code2_vertex] = code2_label
                    Gcodes.graph_properties['index_of'][code2_label] = code2_index
                else:
                    code2_index = Gcodes.graph_properties['index_of'][code2_label]
                    code2_vertex = Gcodes.vertex(code2_index)

                #---------------------------------------------------------------------------
                # check if edge between code1 and code2 exists
                if not Gcodes.edge(code1_vertex,code2_vertex):
                    edge_code1_code2 = Gcodes.add_edge(code1_vertex,code2_vertex)
                else:
                    edge_code1_code2 = Gcodes.edge(code1_vertex,code2_vertex)

                #
                Gcodes.edge_properties['co_oc'][edge_code1_code2] +=1

                Gcodes.vertex_properties['No_of_occurrences'][code1_vertex] +=1
                Gcodes.vertex_properties['No_of_occurrences'][code2_vertex] +=1

                Gcodes.graph_properties['total_cooc'] +=1
                #Gcodes.graph_properties['total_patents'] +=1
            elif code1_label == code2_label and allow_singletons:
                # check if node with code1 exists:
                if not Gcodes.graph_properties['index_of'].has_key(code1_label):
                    code1_vertex = Gcodes.add_vertex()
                    code1_index = int(code1_vertex)
                    Gcodes.vertex_properties['label'][code1_vertex] = code1_label
                    Gcodes.graph_properties['index_of'][code1_label] = code1_index
                else:
                    code1_index = Gcodes.graph_properties['index_of'][code1_label]
                    code1_vertex = Gcodes.vertex(code1_index)
                Gcodes.vertex_properties['No_of_occurrences'][code1_vertex] +=1
            #---------------------------------------------------------------------------
            if curr_line % 1000000 == 0:
                print 'current line: {0}, Class nodes/edges: {1}/{2}, Code nodes/edges {3}/{4}'.format(curr_line,Gclasses.num_vertices(),Gclasses.num_edges(),Gcodes.num_vertices(),Gcodes.num_edges())

    if normalise_weights:
        print 'normalising weights for classes...'
        Gclasses = mygt.calculate_SR(Gclasses)
        print 'normalising weights for codes...'
        Gcodes = mygt.calculate_SR(Gcodes)

    return Gclasses,Gcodes

def load_number_of_patents_per_node_to_graph_tool_for_each_decade_network(decades = range(1790,2020,10)):
    for d in decades:
        print '*** Processing activity of the ' + str(d) + 's...'
        Gclasses = gt.load_graph('Gclasses_' + str(d) + '.xml.gz')
        Gcodes = gt.load_graph('Gcodes_' + str(d) + '.xml.gz')
        Gclasses,Gcodes = load_number_of_patents_per_node_to_graph_tool_decade_network(Gclasses,Gcodes,d)
        print 'File read. Saving graphs...'
        Gclasses.save('Gclasses_' + str(d) + '.xml.gz')
        Gcodes.save('Gcodes_' + str(d) + '.xml.gz')

def load_number_of_patents_per_node_to_graph_tool_decade_network(Gclasses,Gcodes,d):
    Gclasses.vertex_properties['No_of_patents'] = Gclasses.new_vertex_property('int')
    Gcodes.vertex_properties['No_of_patents'] = Gcodes.new_vertex_property('int')

    # add vertex property No_of_patents

    # 0.Pat_Type,1.Patent,2.Primary,3.Class,4.Subclass,   5.GDate,6.AppDate,   7.Appyear
    print 'Reading patents file...'
    with open('PatentCodes_' + str(d) + '.csv','r') as fp:
        num_lines = 0
        for aline in fp:
            num_lines +=1
            if num_lines == 1:continue

            aline = aline.strip()
            entries = aline.split(',')

            pat_class = entries[3]
            pat_code = entries[3] + '/' + entries[4]

            # find pat_class in Gclasses
            if Gclasses.graph_properties['index_of'].has_key(pat_class):
                pat_class_index = Gclasses.graph_properties['index_of'][pat_class]
                pat_class_vertex = Gclasses.vertex(pat_class_index)
                Gclasses.vertex_properties['No_of_patents'][pat_class_vertex] +=1

            # find pat_code in Gclodes
            if Gcodes.graph_properties['index_of'].has_key(pat_code):
                pat_code_index = Gcodes.graph_properties['index_of'][pat_code]
                pat_code_vertex = Gcodes.vertex(pat_code_index)
                Gcodes.vertex_properties['No_of_patents'][pat_code_vertex] +=1

        if num_lines % 1000000:
            print 'read {0} lines'.format(num_lines)

    return Gclasses,Gcodes

def load_number_of_occurrences_per_node_to_graph_tool_for_each_decade_network(decades = range(1790,2020,10)):
    for d in decades:
        print '*** Processing activity of the ' + str(d) + 's...'
        Gclasses,Gcodes = load_number_of_occurrences_per_node_to_graph_tool_decade_network(d)
        print 'File read. Saving graphs...'
        Gclasses.save('Gclasses_' + str(d) + '.xml.gz')
        Gcodes.save('Gcodes_' + str(d) + '.xml.gz')

def load_number_of_occurrences_per_node_to_graph_tool_decade_network(d):
    Gclasses = gt.load_graph('Gclasses_' + str(d) + '.xml.gz')
    Gclasses.vertex_properties['No_of_occurrences'] = Gclasses.new_vertex_property('int')

    Gcodes = gt.load_graph('Gcodes_' + str(d) + '.xml.gz')
    Gcodes.vertex_properties['No_of_occurrences'] = Gclasses.new_vertex_property('int')

    print 'Reading code pairs file...'
    with open('code_pairs_' + str(d) + '.csv','r') as fp:
        curr_line = 0
        for aline in fp:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            N1 = int(entry[4])
            N2 = int(entry[7])
            if N1>=N2:
                continue
            #patentID = entry[1]
            # This file containes only the co-occurrences of patents that exist in Patents.cpickle
            ############################################################################

            class1_label = entry[2]
            class2_label = entry[5]

            if class1_label != class2_label:
                if Gclasses.graph_properties['index_of'].has_key(class1_label):
                    class1_index = Gclasses.graph_properties['index_of'][class1_label]
                    class1 = Gclasses.vertex(class1_index)
                    Gclasses.vertex_properties['No_of_occurrences'][class1] +=1

                if Gclasses.graph_properties['index_of'].has_key(class2_label):
                    class2_index = Gclasses.graph_properties['index_of'][class2_label]
                    class2 = Gclasses.vertex(class2_index)
                    Gclasses.vertex_properties['No_of_occurrences'][class2] +=1

            code1_label = entry[2] + '/' + entry[3]
            code2_label = entry[5] + '/' + entry[6]

            if code1_label != code2_label:
                if Gcodes.graph_properties['index_of'].has_key(class1_label):
                    code1_index = Gcodes.graph_properties['index_of'][class1_label]
                    code1 = Gcodes.vertex(code1_index)
                    Gcodes.vertex_properties['No_of_occurrences'][code1] +=1

                if Gcodes.graph_properties['index_of'].has_key(code2_label):
                    code2_index = Gcodes.graph_properties['index_of'][code2_label]
                    code2 = Gcodes.vertex(code2_index)
                    Gcodes.vertex_properties['No_of_occurrences'][code2] +=1

    print 'done.'
    return Gclasses,Gcodes

def merge_decade_graphs(start_decade,end_decade,graph_type = 'classes'):
    decade_range = range(start_decade+10,end_decade+10,10)

    print '*** Processing decade ' + str(start_decade) + '...'
    Gmerged = gt.load_graph('G' + graph_type + '_' + str(start_decade) + '.xml.gz')
    print 'done. Start network has {0} nodes and {1} edges'.format(Gmerged.num_vertices(),Gmerged.num_edges())
    for d in decade_range:
        print '*** Processing decade ' + str(d) + '...'
        Gnew = gt.load_graph('G' + graph_type + '_' + str(d) + '.xml.gz')
        Gmerged = mygt.merge_cooccurrence_networks(Gmerged,Gnew)


    print 'Re-calculating SRs...'
    Gmerged = mygt.calculate_SR(Gmerged)
    print 'done.'

    print '*** All Done. '
    print 'Merged graph has {0} nodes and {1} edges'.format(Gmerged.num_vertices(),Gmerged.num_edges())
    return Gmerged

def read_top_level_decade_community_structures_from_dot_tree_files(file_head,file_tail,decade_range = range(1790,2020,10)):
    COMMs = dict()
    for d in decade_range:
        print 'Processing decade {0}...'.format(d)
        filename = file_head + str(d) + file_tail
        COMMs[d] = mycomms.read_top_level_community_structure_from_dot_tree_to_community_list(filename + '.tree')

    return COMMs

def read_bottom_level_decade_community_structures_from_dot_tree_files(file_head,file_tail,decade_range = range(1790,2020,10)):
    COMMs = dict()
    for d in decade_range:
        print 'Processing decade {0}...'.format(d)
        filename = file_head + str(d) + file_tail
        COMMs[d] = mycomms.read_bottom_level_community_structure_from_dot_tree_to_community_list(filename + '.tree')

    return COMMs

def read_decade_community_structures_from_dot_map_files(file_head,file_tail,decade_range = range(1790,2020,10)):
    COMMs = dict()
    for d in decade_range:
        print 'Processing decade {0}...'.format(d)
        filename = file_head + str(d) + file_tail
        COMMs[d] = mycomms.read_bottom_level_community_structure_from_dot_tree_to_community_list(filename + '.tree')

    return COMMs

def get_community_structure_similarity_via_NMI_for_each_decade_pair(COMMs,decade_range = None):
    if decade_range is None:
        decade_range = sorted(COMMs.keys())

    no_decades = len(decade_range)

    NMI = numpy.zeros([no_decades,no_decades])

    decade_node_sets = dict()

    for i in range(0,no_decades-1):
        for j in range(i+1,no_decades):

            d1 = decade_range[i]
            d2 = decade_range[j]

            print 'Processing couple ({0},{1})...'.format(d1,d2)

            if not decade_node_sets.has_key(d1):
                decade_node_sets[d1] = mycomms.get_node_set_from_community_list(COMMs[d1])
            node_set1 = decade_node_sets[d1]

            if not decade_node_sets.has_key(d2):
                decade_node_sets[d2] = mycomms.get_node_set_from_community_list(COMMs[d2])
            node_set2 = decade_node_sets[d2]

            common_nodes = node_set1.intersection(node_set2)

            if len(common_nodes) <2:
                NMI[i,j] = 0
            else:
                g1,g2 = mycomms.get_community_substructure_from_node_intersect(COMMs[d1],COMMs[d2],common_nodes)
                NMI[i,j] = mycomms.get_normalised_mutual_information(g1,g2)

            if NMI[i,j] is numpy.nan:
                print 'Warning'

            print 'NMI ({0},{1}) = {2:.2f}'.format(i,j,NMI[i,j])
    NMI = NMI + NMI.transpose()
    for i in range(0,no_decades):NMI[i,i] = 1

    return NMI

def get_community_structure_similarity_via_NMI_across_COOC_SR_per_decade(node_type = 'classes',decade_range = range(1790,2020,10)):
    NMIs = []
    for d in decade_range:
        # read bottom-level comm structure for SR
        gSR = mycomms.read_bottom_level_community_structure_from_dot_tree_to_community_list('G'+node_type+'_'+str(d)+'_SR.tree')
        # read bottom-level comm structure for COOC
        gCOOC = mycomms.read_bottom_level_community_structure_from_dot_tree_to_community_list('G'+node_type+'_'+str(d)+'_COOC.tree')
        # compare groups via NMI
        NMI = mycomms.get_normalised_mutual_information(gSR,gCOOC)
        # store NMI to list
        NMIs.append(NMI)

    return NMIs

def get_modularity_across_decades_from_dot_tree_community_structure(node_type = 'classes_',weight = 'co_oc',decade_range = range(1790,2020,10),tree_level = 'top'):
    Qs = []
    for d in decade_range:
        print '***Processing decade {0}...'.format(d)
        print 'loading graph...'
        G = gt.load_graph('G'+node_type+str(d)+'.xml.gz')
        print 'calculating modularity...'
        Q = mygt.get_modularity_from_dot_tree_via_gt(G,'G'+node_type+str(d)+'_SR.tree',weight,tree_level)
        print 'Q({0}) = {1:.2f}'.format(d,Q)
        Qs.append(Q)

    return Qs

def load_patent_to_technology_matrix_from_file(datafile,use_codes = False):

    #0.Pat_Type,1.Patent,2.Primary,3.Class,4.Subclass,   5.GDate,6.AppDate,   7.Appyear
    patent_count = 0
    tech_count = 0
    AdjList = []
    Patents = my_containers.TwoWayDict()
    Techs = my_containers.TwoWayDict()

    with open(datafile,'r') as fid:
        line_count  =0
        for aline in fid:
            # skip the column headers
            line_count +=1
            if line_count ==1 : continue
            # start with second line

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]
            if not Patents.has_key(patentID):
                patent_count +=1
                patent_index = patent_count -1
                Patents[patentID] = patent_index
            else:
                patent_index = Patents[patentID]

            techID = entry[3]
            if use_codes:techID += '/'+entry[4]
            if not Techs.has_key(techID):
                tech_count +=1
                tech_index = tech_count -1
                Techs[techID] = tech_index
            else:
                tech_index = Techs[techID]

            AdjList.append([patent_index,tech_index])

    #B = convert_list_to_sparse_matrix(AdjList,patent_count,tech_count)

    Results = {'Patents':Patents,'Techs':Techs,'AdjList':AdjList}
    return Results


def get_number_of_patents_per_tech_per_decade(tech_list,tech_type='codes',decade_range = range(1790,2020,10), use_merged_networks=False):
    if use_merged_networks:
        merged_mask = '1790to'
    else:
        merged_mask = ''

    number_of_patents_per_tech_per_decade = dict()
    for k in tech_list:number_of_patents_per_tech_per_decade[k]=dict()

    for d in decade_range:
        if use_merged_networks and d == 1790:
            G = gt.load_graph('G'+tech_type+'_1790.xml.gz'.format(d))
        else:
            G = gt.load_graph('G'+tech_type+'_'+merged_mask+'{0}.xml.gz'.format(d))
        for tech_label in tech_list:
            if G.graph_properties['index_of'].has_key(tech_label):
                tech_index = G.graph_properties['index_of'][tech_label]
                tech_vertex = G.vertex(tech_index)
                number_of_patents_per_tech_per_decade[tech_label][d] = G.vertex_properties['No_of_occurrences'][tech_vertex]
            else:
                number_of_patents_per_tech_per_decade[tech_label][d] = numpy.nan

    return number_of_patents_per_tech_per_decade


def get_average_number_of_neighbour_patents_per_tech_per_decade(tech_list,tech_type='codes',decade_range = range(1790,2020,10), remove_common_patents = True,use_merged_networks=False):
    if use_merged_networks:
        merged_mask = '1790to'
    else:
        merged_mask = ''

    number_of_neighbour_patents_per_tech_per_decade = dict()
    for k in tech_list:number_of_neighbour_patents_per_tech_per_decade[k]=dict()

    for d in decade_range:
        if use_merged_networks and d == 1790:
            G = gt.load_graph('G'+tech_type+'_1790.xml.gz'.format(d))
        else:
            G = gt.load_graph('G'+tech_type+'_'+merged_mask+'{0}.xml.gz'.format(d))
        for tech_label in tech_list:
            if G.graph_properties['index_of'].has_key(tech_label):
                tech_index = G.graph_properties['index_of'][tech_label]
                tech_vertex = G.vertex(tech_index)

                total_neighbours = 0
                total_neighbour_patents = 0
                for neighbour_vertex in tech_vertex.out_neighbours():
                    total_neighbours +=1
                    total_neighbour_patents += G.vertex_properties['No_of_occurrences'][neighbour_vertex]
                    if remove_common_patents:
                        total_neighbour_patents -= G.edge_properties['co_oc'][G.edge(tech_vertex,neighbour_vertex)]

                    number_of_neighbour_patents_per_tech_per_decade[tech_label][d] = (1.0*total_neighbour_patents)/total_neighbours
            else:
                number_of_neighbour_patents_per_tech_per_decade[tech_label][d] = numpy.nan

    return number_of_neighbour_patents_per_tech_per_decade

def get_first_appearance_year_per_tech(tech_list_baseline,tech_type='codes',decade_range = range(1790,2020,10),use_merged_networks=False):
    if use_merged_networks:
        merged_mask = '1790to'
    else:
        merged_mask = ''

    first_appearance_year_per_tech = dict()
    tech_list = [elem for elem in tech_list_baseline]

    for d in decade_range:
        if use_merged_networks and d == 1790:
            G = gt.load_graph('G'+tech_type+'_1790.xml.gz'.format(d))
        else:
            G = gt.load_graph('G'+tech_type+'_'+merged_mask+'{0}.xml.gz'.format(d))

        not_scanned_yet = [elem for elem in tech_list]

        for tech_label in not_scanned_yet:
            if G.graph_properties['index_of'].has_key(tech_label):
                first_appearance_year_per_tech[tech_label] = d
                tech_list.remove(tech_label)

    for tech_label in tech_list:
        first_appearance_year_per_tech[tech_label] = None

    return first_appearance_year_per_tech

def get_first_combination_year_per_tech(tech_list_baseline,tech_type='codes',decade_range = range(1790,2020,10),use_merged_networks=False):
    if use_merged_networks:
        merged_mask = '1790to'
    else:
        merged_mask = ''

    first_combination_year_per_tech = dict()
    tech_list = [elem for elem in tech_list_baseline]

    for d in decade_range:
        if use_merged_networks and d == 1790:
            G = gt.load_graph('G'+tech_type+'_1790.xml.gz'.format(d))
        else:
            G = gt.load_graph('G'+tech_type+'_'+merged_mask+'{0}.xml.gz'.format(d))

        not_scanned_yet = [elem for elem in tech_list]

        for tech_label in not_scanned_yet:
            if G.graph_properties['index_of'].has_key(tech_label):
                tech_index = G.graph_properties['index_of'][tech_label]
                tech_vertex = G.vertex(tech_index)
                if mygt.is_vertex_connected(tech_vertex):
                    first_combination_year_per_tech[tech_label] = d
                    tech_list.remove(tech_label)

    for tech_label in tech_list:
        first_combination_year_per_tech[tech_label] = None

    return first_combination_year_per_tech

def get_number_of_patents_until_combination(tech_list_baseline,tech_type='codes',decade_range = range(1790,2020,10),first_combination_year_per_tech = None,use_merged_networks=False):
    if use_merged_networks:
        merged_mask = '1790to'
    else:
        merged_mask = ''

    if first_combination_year_per_tech is None:
        first_combination_year_per_tech = get_first_combination_year_per_tech(tech_list_baseline,tech_type,decade_range,use_merged_networks)

    number_of_patents_until_combination_per_tech = dict()
    for tech_label in tech_list_baseline:
        number_of_patents_until_combination_per_tech[tech_label]=0

    tech_list = [elem for elem in tech_list_baseline]

    for d in decade_range:
        if use_merged_networks and d == 1790:
            G = gt.load_graph('G'+tech_type+'_1790.xml.gz'.format(d))
        else:
            G = gt.load_graph('G'+tech_type+'_'+merged_mask+'{0}.xml.gz'.format(d))

        not_scanned_yet = [elem for elem in tech_list]

        for tech_label in not_scanned_yet:
            if d<first_combination_year_per_tech[tech_label] and G.graph_properties['index_of'].has_key(tech_label):
                tech_index = G.graph_properties['index_of'][tech_label]
                tech_vertex = G.vertex(tech_index)

                number_of_patents_until_combination_per_tech[tech_label] += G.vertex_properties['No_of_occurrences'][tech_vertex]
            elif d>=first_combination_year_per_tech[tech_label]:
                tech_list.remove(tech_label)

    return number_of_patents_until_combination_per_tech

def get_number_of_decades_between_first_appearance_and_combination(tech_list,first_appearance_year_per_class,first_combination_year_per_tech,tech_type='codes',decade_range = range(1790,2020,10)):
    if first_appearance_year_per_class is None:
        first_appearance_year_per_class = get_first_appearance_year_per_tech(tech_list,tech_type,decade_range)
    if first_combination_year_per_tech is None:
        first_combination_year_per_tech = get_first_combination_year_per_tech(tech_list,tech_type,decade_range)


    decade_diff = dict()
    for tech in tech_list:
        if isinstance(first_combination_year_per_tech[tech],int):
            decade_diff[tech] = first_combination_year_per_tech[tech] - first_appearance_year_per_class[tech]
        else:
            decade_diff[tech] = len(decade_range)*10

    return decade_diff

def get_timeseries_cross_correlation_matrix(At,techs_focus,techs_lookup,time_lag=0):
    N = len(techs_focus)

    C = numpy.zeros((N,N))

    for i in range(0,N-1):
        for j in range(i+1,N):

            tech_i_index = techs_lookup[techs_focus[i]]
            tech_j_index = techs_lookup[techs_focus[j]]

            ait = At[tech_i_index,:]
            ajt = At[tech_j_index,:]

            C[i,j] = mystat.cross_correlation(ait,ajt,time_lag)

    return C

def get_timeseries_normalised_cross_correlation_matrix(At,techs_focus,techs_lookup,time_lag=0):
    N = len(techs_focus)

    C = numpy.zeros((N,N))

    for i in range(0,N-1):
        for j in range(i+1,N):

            tech_i_index = techs_lookup[techs_focus[i]]
            tech_j_index = techs_lookup[techs_focus[j]]

            ait = At[tech_i_index,:]
            ajt = At[tech_j_index,:]

            C[i,j] = mystat.normalised_cross_correlation(ait,ajt,time_lag)

    return C

def read_PV_patents(filename):
    PV_patents = set([])
    with open(filename,'r') as fp:
        line_count=0
        for aline in fp:
            line_count+=1
            if line_count==1:continue

            aline = aline.strip()
            elems = aline.split(',')
            pat_no = elems[0]+elems[1]

            PV_patents.add(pat_no)
    return PV_patents

def read_PV_codes_per_decade(PV_patents=None,decade_range = range(1790,2020,10)):
    if PV_patents is None:
        PV_patents = read_PV_patents('PV_patents.csv')

    PV_codes_per_decade = dict()
    for d in decade_range:
        PV_codes_per_decade[d] = set([])

        with open('Patents_v2_{0}.csv'.format(d),'r') as fp:
            line_count = 0
            pat_no_prev = None
            is_PV_patent = False
            for aline in fp:
                line_count+=1
                if line_count==1:continue

                aline = aline.strip()
                elems = aline.split(',')
                pat_no_cur = elems[0]+elems[1]

                tech_code = elems[3] + '/' + elems[4]

                if pat_no_cur!=pat_no_prev:
                    is_PV_patent =  pat_no_cur in PV_patents
                    if is_PV_patent:
                        PV_codes_per_decade[d].add(tech_code)
                elif pat_no_cur==pat_no_prev and is_PV_patent:
                    PV_codes_per_decade[d].add(tech_code)

    return PV_codes_per_decade

def get_number_of_patents_per_decade_per_CP_categorisation(classes_per_era,decade_range=range(1790,2010,10)):

    def close_patent_batch():
        for era in range(1,6):
            classes_of_era = classes_per_era[era]
            found_flag = False
            for patent_class in current_classes:
                for era_class in classes_of_era:
                    if patent_class==era_class:
                        found_flag = True
                        break
                if found_flag:break

            DATAFRAME[era][d] += found_flag
            DATAFRAME_PRIMARY_ONLY[era][d] += primary_class in classes_of_era


    DATAFRAME = pandas.DataFrame(numpy.zeros((len(decade_range),5)),index=decade_range,columns=range(1,6))
    DATAFRAME_PRIMARY_ONLY = pandas.DataFrame(numpy.zeros((len(decade_range),5)),index=decade_range,columns=range(1,6))

    for d in decade_range:
        print 'Processing decade {0}...'.format(d)
        with open('Patents_v2_{0}.csv'.format(d),'r') as fp:
            total_patents = 0
            previous_patent = 'N/A'
            current_classes = set()
            #current_codes = set()
            print 'reading contents...'
            curr_line = 0
            for aline in fp:
                curr_line +=1
                if curr_line == 1:continue

                aline = aline.strip()
                entry = aline.split(',')

                patentID = entry[0] + entry[1]
                current_class_label = entry[3]
                #current_code_label = entry[3] + '/' + entry[4]
                current_class_is_primary = int(entry[2])
                if current_class_is_primary:
                    primary_class = current_class_label

                # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
                if patentID == previous_patent:
                    # =================== IN A PATENT BATCH ===================
                    current_classes.add(current_class_label)

                    #current_codes.add(current_code_label)
                else:
                    # =================== CLOSING A PATENT BATCH ===================
                    if len(current_classes)!=0:
                        # A) INCREMENT PATENTS
                        total_patents+=1
                        # B) GATHER NODES AND LINKS
                        close_patent_batch()
                    # C) CLOSE CLASS / NODE SETS
                    current_classes = set()
                    current_classes.add(current_class_label)
                    #primary_class = 'N/A'
                    previous_patent = patentID

                if curr_line % 1000000 == 0:
                    print 'current line: {0}, patents so far: {1}'.format(curr_line,total_patents)

            # close the patent patch for the remaining entries
            if len(current_classes)>0:
                close_patent_batch()

    return DATAFRAME,DATAFRAME_PRIMARY_ONLY

def filter_patents_decade_given_code_set(filter_by_code,tech_set, prefix, base_filename = 'Patents_v2', decade_range = range(1790,2020,10),primary_only = False):

    All_Decade_filenames = dict(zip(decade_range,[base_filename+str(d)+'.csv' for d in decade_range]))
    Code_Decade_filenames = dict(zip(decade_range,[prefix + '_' + base_filename+ primary_only*'_Primary_based_'+ str(d) +'.csv' for d in decade_range]))

    total_patents = 0
    for d in decade_range:
        print '*** Opening ' + All_Decade_filenames[d] + '...'
        with open(All_Decade_filenames[d],'r') as read_fp, open(Code_Decade_filenames[d],'w') as write_fp:
            print 'done. Iterating through contents...'
            previous_patent = 'N/A'
            raw_patent_lines = ''
            IN_SET = False

            num_lines = 0
            written_lines = 0
            for aline in read_fp:
                num_lines +=1
                if num_lines ==1:
                    write_fp.write('Pat_Type,Patent,Primary,Class,Subclass,Type,GDate,AppDate,Appyear\n')
                    continue


                entry = aline.strip().split(',')
                #   Pat_Type,Patent,Primary,Class,Subclass,Type,GDate,AppDate,Appyear

                patentID = entry[0] + entry[1]
                current_class_label = entry[3]
                current_code_label = entry[3] + '/' + entry[4]

                is_primary = entry[2]=='1'

                # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
                if patentID == previous_patent:
                    # =================== IN A PATENT BATCH ===================
                    raw_patent_lines += aline
                else:
                    # =================== CLOSING A PATENT BATCH ===================
                    if IN_SET:
                        write_fp.write(raw_patent_lines)
                        written_lines += len(raw_patent_lines.split('\n'))
                        total_patents+=1

                    previous_patent = patentID
                    raw_patent_lines = aline
                    IN_SET = False

                if not IN_SET:
                    if not primary_only:
                        IN_SET = (filter_by_code and current_code_label in tech_set) or (not filter_by_code and current_class_label in tech_set)
                    else:
                        IN_SET = (filter_by_code and current_code_label in tech_set) or (not filter_by_code and current_class_label in tech_set)
                        IN_SET = IN_SET and is_primary

        print 'Successfully read {0} lines and added {1} lines, with {2} total {3}-related patents.'.format(num_lines,written_lines,total_patents,prefix)

def get_patent_code_incidence_matrix_from_file(filename,use_codes=True):
    # use_coo: COOrdinate format
    # use_lil: linked-list format
    def update_AdjList():
        if use_codes:
            current_patent_tech_indices = [code_lookup[acode] for acode in current_codes]
        else:
            current_patent_tech_indices = [class_lookup[aclass] for aclass in current_classes]

        if len(current_patent_tech_indices)!=0:
            for t in current_patent_tech_indices:
                AdjList.append([total_patents,t])
            #AdjList.append(current_patent_tech_indices)
            return True
        else:
            return False

    code_lookup = dict()
    class_lookup = dict()
    AdjList = []

    print 'opening ' + filename + '...'
    with open(filename,'r') as fid:
        total_patents = 0
        total_codes = 0
        total_classes = 0

        previous_patent = 'N/A'
        current_classes = set()
        current_codes = set()

        print 'reading contents...'
        curr_line = 0
        for aline in fid:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]

            current_class_label = entry[3]
            if not class_lookup.has_key(current_class_label):
                class_lookup[current_class_label] = total_classes
                total_classes+=1

            current_code_label = entry[3] + '/' + entry[4]
            if not code_lookup.has_key(current_code_label):
                code_lookup[current_code_label] = total_codes
                total_codes+=1

            # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
            if patentID == previous_patent:
                # =================== IN A PATENT BATCH ===================
                current_classes.add(current_class_label)
                current_codes.add(current_code_label)
            else:
                # =================== CLOSING A PATENT BATCH ===================
                # A) UPDATE ADJLIST
                new_entries_added = update_AdjList()
                # B) INCREMENT PATENTS
                if new_entries_added:total_patents+=1
                # C) CLOSE CLASS / NODE SETS
                current_classes = set()
                current_classes.add(current_class_label)
                current_codes = set()
                current_codes.add(current_code_label)
                previous_patent = patentID

        # close the patent patch for the remaining entries
        if len(current_classes)>0 or len(current_codes)>0:
            new_entries_added = update_AdjList()
            if new_entries_added:total_patents+=1

        if use_codes:
            tech_lookup = code_lookup
        else:
            tech_lookup = class_lookup


    COOarray = numpy.array(AdjList)

    K = use_codes*total_codes + (not use_codes)*total_classes
    #sparse.coo_matrix((numpy.ones(COOarray.shape[0]),(COOarray[:,0],COOarray[:,1])),(total_patents,K))
    B = scipy.sparse.coo_matrix(COOarray[:,0],COOarray[:,1],total_patents,K)
    INCIDENCE_MATRIX = mycons.TechnologyMatrix(B,col_elem_labels=tech_lookup)

    return INCIDENCE_MATRIX
    #return AdjList,tech_lookup

def get_patent_code_incidence_matrix_from_multiple_files(base_filename,decade_range,use_codes = True):

    def update_AdjList():
        if use_codes:
            current_patent_tech_indices = [code_lookup[acode] for acode in current_codes]
        else:
            current_patent_tech_indices = [class_lookup[aclass] for aclass in current_classes]

        if len(current_patent_tech_indices)!=0:
            for t in current_patent_tech_indices:
                AdjList.append([total_patents,t])
            #AdjList.append(current_patent_tech_indices)
            return True
        else:
            return False

    code_lookup = dict()
    class_lookup = dict()
    AdjList = []

    filenames = [base_filename + '_' + str(d) + '.csv' for d in decade_range]

    total_patents = 0
    total_codes = 0
    total_classes = 0

    for filename in filenames:
        print 'opening ' + filename + '...'
        with open(filename,'r') as fid:
            previous_patent = 'N/A'
            current_classes = set()
            current_codes = set()

            print 'reading contents...'
            curr_line = 0
            for aline in fid:

                curr_line +=1
                if curr_line == 1:
                    continue

                aline = aline.strip()
                entry = aline.split(',')

                patentID = entry[0] + entry[1]

                current_class_label = entry[3]
                if not class_lookup.has_key(current_class_label):
                    class_lookup[current_class_label] = total_classes
                    total_classes+=1

                current_code_label = entry[3] + '/' + entry[4]
                if not code_lookup.has_key(current_code_label):
                    code_lookup[current_code_label] = total_codes
                    total_codes+=1

                # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
                if patentID == previous_patent:
                    # =================== IN A PATENT BATCH ===================
                    current_classes.add(current_class_label)
                    current_codes.add(current_code_label)
                else:
                    # =================== CLOSING A PATENT BATCH ===================
                    # A) UPDATE ADJLIST
                    new_entries_added = update_AdjList()
                    # B) INCREMENT PATENTS
                    if new_entries_added:total_patents+=1
                    # C) CLOSE CLASS / NODE SETS
                    current_classes = set()
                    current_classes.add(current_class_label)
                    current_codes = set()
                    current_codes.add(current_code_label)
                    previous_patent = patentID

            # close the patent patch for the remaining entries
            if len(current_classes)>0 or len(current_codes)>0:
                new_entries_added = update_AdjList()
                if new_entries_added:total_patents+=1


    if use_codes:
        tech_lookup = code_lookup
    else:
        tech_lookup = class_lookup

    COOarray = numpy.array(AdjList)

    K = use_codes*total_codes + (not use_codes)*total_classes
    #sparse.coo_matrix((numpy.ones(COOarray.shape[0]),(COOarray[:,0],COOarray[:,1])),(total_patents,K))
    B = scipy.sparse.coo_matrix(COOarray[:,0],COOarray[:,1],total_patents,K)
    INCIDENCE_MATRIX = mycons.TechnologyMatrix(B,col_elem_labels=tech_lookup)

    return INCIDENCE_MATRIX
    #return AdjList,tech_lookup


def get_innovation_coordinates(filename,use_codes = False):

    INNOV_COORDINATES = []
    SPECIALIZATION = 0
    INNOVATION_SOFT = 1
    INNOVATION_HARD = 2
    total_patents = 0
    total_codes = 0
    total_classes = 0
    G = gt.Graph(directed=False)


    def update_coordinates():
        if use_codes:
            current_patent_tech_indices = sorted([code_lookup[acode] for acode in current_codes])
        else:
            current_patent_tech_indices = sorted([class_lookup[aclass] for aclass in current_classes])

        number_of_techs_in_patent_batch = len(current_patent_tech_indices)

        if number_of_techs_in_patent_batch>0:
            is_new_flags = [False]*number_of_techs_in_patent_batch
            for i in range(0,number_of_techs_in_patent_batch):
                v_index = current_patent_tech_indices[i]
                is_new_flags[i] = not mygt.has_vertex(G,v_index)
                if is_new_flags[i]:
                    G.add_vertex()

            innov_coord = [0,0,0]

            try:
                if number_of_techs_in_patent_batch>1:
                    for i in range(0,number_of_techs_in_patent_batch-1):
                        v_i_index = current_patent_tech_indices[i]
                        for j in range(i+1,number_of_techs_in_patent_batch):
                            v_j_index = current_patent_tech_indices[j]

                            if is_new_flags[i] or is_new_flags[j]:
                                innov_coord[INNOVATION_HARD]+=1
                                G.add_edge(v_i_index,v_j_index)
                            elif (not is_new_flags[i]) and (not is_new_flags[j]):
                                if not mygt.has_edge(G,v_i_index,v_j_index):
                                    innov_coord[INNOVATION_SOFT]+=1
                                    G.add_edge(v_i_index,v_j_index)
                                else:
                                    innov_coord[SPECIALIZATION]+=1
            except ValueError:
                print 'to err is human'

            INNOV_COORDINATES.append(innov_coord)

    code_lookup = dict()
    class_lookup = dict()

    print 'opening ' + filename + '...'
    with open(filename,'r') as fid:
        previous_patent = 'N/A'
        current_classes = set()
        current_codes = set()

        print 'reading contents...'
        curr_line = 0
        for aline in fid:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]

            current_class_label = entry[3]
            if not class_lookup.has_key(current_class_label):
                class_lookup[current_class_label] = total_classes
                total_classes+=1

            current_code_label = entry[3] + '/' + entry[4]
            if not code_lookup.has_key(current_code_label):
                code_lookup[current_code_label] = total_codes
                total_codes+=1

            # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
            if patentID == previous_patent:
                # =================== IN A PATENT BATCH ===================
                current_classes.add(current_class_label)
                current_codes.add(current_code_label)
            else:
                # =================== CLOSING A PATENT BATCH ===================
                # A) INCREMENT PATENTS
                total_patents+=1
                # B) UPDATE COORDINATES
                update_coordinates()
                # C) CLOSE CLASS / NODE SETS
                current_classes = set()
                current_classes.add(current_class_label)
                current_codes = set()
                current_codes.add(current_code_label)
                previous_patent = patentID

        # close the patent patch for the remaining entries
        if len(current_classes)>0 or len(current_codes)>0:
            update_coordinates()

        if use_codes:
            tech_lookup = code_lookup
        else:
            tech_lookup = class_lookup

    return INNOV_COORDINATES

def get_class_combination_ratios(class_list,method='No_of_patents',decade_range=range(1790,2020,10)):
    # methods = {'No_of_patents','Strength'}
    CRs = dict()
    for c in class_list:
        CRs[c] = []

    for d in decade_range:
        G = gt.load_graph('Gclasses_{0}.xml.gz'.format(d))

        for c in class_list:
            if G.graph_properties['index_of'].has_key(c):
                v_index = G.graph_properties['index_of'][c]
                v = G.vertex(v_index)

                self_combination_intensity = G.vertex_properties['No_of_singleton_occurrences'][v]

                if method == 'No_of_patents':
                    external_combination_intensity = G.vertex_properties['No_of_occurrences'][v] - G.vertex_properties['No_of_singleton_occurrences'][v]
                else:
                    external_combination_intensity = mygt.get_vertex_strength(G,v)

                if external_combination_intensity !=0:
                    cr = (1.* self_combination_intensity) / external_combination_intensity
                else:
                    cr = numpy.inf
                CRs[c].append(cr)
            else:
                CRs[c].append(numpy.nan)

    max_timeline = 0
    for c in class_list:
        if max_timeline<len(CRs[c]):
            max_timeline = len(CRs[c])

    Vals = [[] for i in range(0,max_timeline)]
    for i in range(0,max_timeline):
        for c in class_list:
            if len(CRs[c])>i:
                Vals[i].append(CRs[c][i])

    return Vals

def convert_code_from_Debbie_to_USPTO_daniel(given_code):
    abnormal_map = {'340/310110': '340/FOR465',
                    '340/310120': '340/FOR466',
                    '340/310130': '340/FOR467',
                    '340/310140': '340/FOR468',
                    '340/310150': '340/FOR469',
                    '340/310160': '340/FOR470',
                    '340/310170': '340/FOR471',
                    '340/310180': '340/FOR472'}

    tkns = given_code.split('/')
    cls = tkns[0]
    subcls = tkns[1]

    m = re.match(r'\d{3}\d[A-z]0', subcls)
    if m is not None:
        subcls = '%s%s0%s' % (m.group(0)[:3], m.group(0)[3], m.group(0)[4])
    else:
        m = re.match(r'\d{3}[A-z]00', subcls)
        if m is not None:
            subcls = '%s00%s' % (m.group(0)[:3], m.group(0)[3])
        else:
            m = re.match(r'\d{3}[A-z]{2}0', subcls)
            if m is not None:
                subcls = '%s0%s'  % (m.group(0)[:3], m.group(0)[3:5])
            else:
                m = re.match(r'0[A-z]\d\w{3}', subcls)
                if m is not None:
                    subcls = '%s0%s' % (m.group(0)[1], m.group(0)[2:])

    code   = cls + '/' + subcls
    if code in abnormal_map:
        code = abnormal_map[code]

    return code

def count_patents_where_a_two_class_pools_cooccur(class_list_A,class_list_B,datafile):
    def close_patent_batch():
            exists_in_A = False
            exists_in_B = False

            # iterate classes of patent
            for aclass_label in current_classes:
                # check if class is in class_list_A
                exists_in_A += aclass_label in class_list_A
                # check if class is in class_list_B
                exists_in_B += aclass_label in class_list_B
                # stop if both found
                if exists_in_A and exists_in_B:break

            return exists_in_A and exists_in_B

    NUMBER_OF_RELEVANT_PATENTS = 0
    total_patents = 0
    #datafile = file_header + str(decade) + file_tail
    print 'opening ' + datafile + '...'
    with open(datafile,'r') as fid:

        previous_patent = 'N/A'
        current_classes = set()
        current_codes = set()

        #print 'reading contents...'
        curr_line = 0
        for aline in fid:

            curr_line +=1
            if curr_line == 1:
                continue

            aline = aline.strip()
            entry = aline.split(',')

            patentID = entry[0] + entry[1]
            current_class_label = entry[3]
            #current_code_label = entry[3] + '/' + entry[4]

            # CHECK IF WE ARE WITHIN, OR OUTSIDE A NEW PATENT BATCH:
            if patentID == previous_patent:
                # =================== IN A PATENT BATCH ===================
                current_classes.add(current_class_label)
                #current_codes.add(current_code_label)
            else:
                NUMBER_OF_RELEVANT_PATENTS += close_patent_batch()

                # re-initialised
                current_classes = set()
                current_classes.add(current_class_label)
                #current_codes = set()
                #current_codes.add(current_code_label)
                previous_patent = patentID
                total_patents+=1

            if curr_line % 1000000 == 0:
                print 'current line: {0}'.format(curr_line)

        # CLOSE PATENT BATCH FOR THE REMAINING ENTRIES
        if len(current_classes)>0 or len(current_codes)>0:
            NUMBER_OF_RELEVANT_PATENTS += close_patent_batch()
            total_patents+=1

    return NUMBER_OF_RELEVANT_PATENTS,total_patents

def get_network_density_from_two_class_pools(class_list_A,class_list_B,network_file):
    G = gt.load_graph(network_file)
    is_bridge_edge = G.new_edge_property('bool')

    for e in G.edges():
        u = e.source()
        v = e.target()

        uname = G.vertex_properties['label'][u]
        vname = G.vertex_properties['label'][v]

        is_bridge_edge[e] =  (uname in class_list_A and vname in class_list_B) or (uname in class_list_B and vname in class_list_A)

    G.set_edge_filter(is_bridge_edge)

    return (2.*G.num_edges())/(G.num_vertices()**2 - G.num_vertices())

def build_temporal_link_from_coocurrence_history_of_class_pair(classA,classB,decade_range):
    w = []
    xi = []
    xj = []

    for d in decade_range:
        G = gt.load_graph('Network_files/Gclasses_{0}.xml.gz'.format(d))

        vi = mygt.get_vertex_by_label(G,classA)
        if vi is not None:
            xi.append(G.vertex_properties['No_of_occurrences'][vi])
        else:
            xi.append(numpy.nan)

        vj = mygt.get_vertex_by_label(G,classB)
        if vj is not None:
            xj.append(G.vertex_properties['No_of_occurrences'][vj])
        else:
            xj.append(numpy.nan)

        if (vi is None) or (vj is None):
            w.append(numpy.nan)
        else:
            w.append(mygt.get_edge_weight(G,classA,classB))

    return BOMP.TemporalLink(w,xi,xj,decade_range)

def get_adjacency_frames_CP_class_groups(decade_range = range(1790,2020,10)):
    classes_of_era = cpickle.load(open('classes_of_era.cpickle','rb'))
    N = 5
    A_FRAMES = dict()
    for d in decade_range:
        print('Processing decade {0}...'.format(d))
        A_FRAMES[d] = numpy.zeros((N,N),dtype=numpy.int)

        for i in range(0,N-1):
            for j in range(i+1,N):
                aux = count_patents_where_a_two_class_pools_cooccur(classes_of_era[i],classes_of_era[j],'Patent_files/Patents_v2_{0}.csv'.format(d))
                A_FRAMES[d][i][j] = aux[0]

    return A_FRAMES