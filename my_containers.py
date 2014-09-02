__author__ = 'yannis'

import numpy
import scipy
import scipy.sparse as sparse
import matplotlib.pylab as pylab
import my_graph_tool_add_ons as mygt
import graph_tool as gt

# TASKS:
# -change constructor to take only incidence matrix - DONE
# -if coordinates provided, sparse matrix is built by a separate, static method - DONE
# -add co-occurrences i,j for both directions
# -add SR (Jaccard Index) for i,j for both directions
# -get SR-based strength of i for both directions
# -get


class IncidenceMatrix:
    def __init__(self,B,row_elem_labels = None,col_elem_labels = None):
        self.B = B
        self.row_elem_labels = row_elem_labels
        self.col_elem_labels = col_elem_labels

    def configure_B_for_row_operations(self):
        curr_format = self.B.getformat()
        if curr_format != 'csr':
            self.B = self.B.tocsr()
    def configure_B_for_column_operations(self):
        curr_format = self.B.getformat()
        if curr_format != 'csc':
            self.B = self.B.tocsc()

    def degrees_of_row_elems(self,i = None):
        self.configure_B_for_row_operations()
        if i is None:
            return numpy.squeeze(numpy.asarray(self.B.sum(1)))
        else:
            return self.B[i,:].sum()

    def plot_row_elem_degree_distribution(self,logscale = True,bins = None):
        row_elem_degrees = self.degrees_of_row_elems()
        if bins is None:
            bins = int(numpy.max(row_elem_degrees))
        pylab.hist(row_elem_degrees,bins=bins)
        if logscale:
            pylab.gca().set_yscale("log")
            pylab.gca().set_xscale("log")
        pylab.show()

    def degrees_of_column_elems(self,i=None):
        self.configure_B_for_column_operations()
        if i is None:
            return numpy.squeeze(numpy.asarray(self.B.sum(0)))
        else:
            return self.B[:,i].sum()

    def plot_column_elem_degree_distribution(self,logscale = True,bins = None):
        column_elem_degrees = self.degrees_of_column_elems()
        if bins is None:
            bins = int(numpy.max(column_elem_degrees))
        pylab.hist(column_elem_degrees,bins=bins)
        if logscale:
            pylab.gca().set_yscale("log")
            pylab.gca().set_xscale("log")
        pylab.show()

    def get_one_mode_projection_to_gt_graph(self,directionality=1):
        if directionality:
            self.configure_B_for_column_operations()
            B = self.B
            Ar = B.transpose() * B
            G = mygt.create_gt_graph_from_sparse_adjacency_matrix(Ar,weight_type='int',list_of_vertex_labels=self.col_elem_labels)
        else:
            self.configure_B_for_row_operations()
            B = self.B
            Ar = B * B.transpose()
            G = mygt.create_gt_graph_from_sparse_adjacency_matrix(Ar,weight_type='int',list_of_vertex_labels=self.row_elem_labels)
        return G

    def spy(self):
        pylab.spy(self.B)
        pylab.show()

    @staticmethod
    def build_sparse_matrix_from_coordinate_data(x,y,num_rows,num_cols,data = None):
        if data is None:
            data =  numpy.ones(len(x))
        return scipy.sparse.coo_matrix((data,(x,y)),shape=(num_rows,num_cols))

class TechnologyMatrix(IncidenceMatrix):

    def number_of_new_technologies_per_patent(self):
        self.configure_B_for_row_operations()
        B = self.B
        number_of_new_column_elems_per_step = numpy.zeros(B.shape[0],dtype=numpy.int)
        number_of_existing_elems_prev = 0
        for n in range(0,B.shape[0]):
            row_focus = B[n,:]
            number_of_existing_elems = numpy.max([numpy.max(row_focus.nonzero()[1]) + 1,number_of_existing_elems_prev])
            number_of_new_column_elems_per_step[n] = number_of_existing_elems - number_of_existing_elems_prev
            number_of_existing_elems_prev = number_of_existing_elems
        return number_of_new_column_elems_per_step

    def technology_age_as_elapsed_number_of_patents(self,i=None):
        self.configure_B_for_column_operations()
        if i is not None:
            bi = self.B[:,i]
            return numpy.nonzero(bi)[0]
        else:
            ages = numpy.zeros(self.B.shape[1])
            for i in range(0,self.B.shape[1]):
                bi = self.B[:,i]
                ages[i] = self.B.shape[0] - bi.nonzero()[0][0]
            return ages

    def get_correlation_technology_age_vs_degree(self):
        degrees = self.degrees_of_column_elems()
        ages = self.technology_age_as_elapsed_number_of_patents()
        return numpy.corrcoef(ages,degrees)

    def get_correlation_patent_age_vs_degree(self):
        degrees = self.degrees_of_row_elems()
        ages = self.B.shape[0] - numpy.arange(0,self.B.shape[0],dtype=numpy.int)
        return numpy.corrcoef(ages,degrees)

    def get_number_of_discovered_technologies_at_patent_step(self,patent_step = None):
        self.configure_B_for_row_operations()
        B = self.B
        if patent_step is None:
            Kp = numpy.zeros(B.shape[0],dtype=numpy.int)
            Kmax = 0
            for n in range(0,B.shape[0]):
                while Kmax+1 < B.shape[1] and B[n,Kmax+1]!=0:
                    Kmax+=1
                Kp[n] = Kmax + 1
        else:
            Kp = numpy.max(B[patent_step,:].nonzero()[1]) + 1

        return Kp
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
def convert_list_to_sparse_matrix(AdjList,number_of_rows=None,number_of_columns=None):
    ### ASSUMES THAT NUMBERING STARTS FROM ZERO

    if number_of_columns is None or number_of_rows is None:
        number_of_rows = len(AdjList)
        number_of_columns = 0
        for entry in AdjList:
            if max(entry)>number_of_columns:
                number_of_columns = max(entry)


    A = sparse.lil_matrix((number_of_rows,number_of_columns+1))
    i = -1
    for entry in AdjList:
        i+=1
        for j in entry:
            A[i,j] +=1

    return A

#def convert_list_to_sparse_matrix(AdjList,number_of_rows=None,number_of_columns=None):
#    if number_of_columns is None or number_of_rows is None:
#        number_of_columns = 0
#        number_of_rows = 0
#        for entry in AdjList:
#            i = entry[0]
#            j = entry[1]
#
#            if i > number_of_rows: number_of_rows=i
#            if j > number_of_columns: number_of_columns=j
#
#    A = sparse.lil_matrix((number_of_rows,number_of_columns))
#    for entry in AdjList:
#        i = entry[0]
#        j = entry[1]
#        A[i,j] +=1
#
#    return A

def save_AdjList_to_csv(AdjList,filename):
    with open(filename,'w') as fp:
        for entry in AdjList:
            fp.write('{0},{1}\n'.format(entry[0],entry[1]))

def read_AdjList_from_csv(filename):
    A = []
    with open(filename,'r') as fp:
        for aline in fp:
            aline = aline.strip()
            entries = aline.split(',')
            A.append([int(entries[0]),int(entries[1])])
    return A

#class TwoWayDict(dict):
#    def __len__(self):
#        return dict.__len__(self) / 2
#
#    def __setitem__(self, key, value):
#        dict.__setitem__(self, key, value)
#        dict.__setitem__(self, value, key)


#
#class PatentTechContainer():
#    AdjList = None
#    Patents = None
#    Techs = None
#
#    total_patents = None
#    total_techs = None
#
#    use_codes = None
#
#    def __init__(self, AdjList, use_codes, Patents = None, Techs = None, total_patents = None, total_techs = None):
#        self.use_codes = use_codes
#        self.AdjList = AdjList
#
#        if Patents is not None:
#            self.Patents = Patents
#            self.total_patents = len(self.Patents)
#        else:
#            self.total_patents = total_patents
#
#        if Techs is not None:
#            self.Techs = Techs
#            self.total_techs = len(self.Techs)
#        else:
#            self.total_techs = total_techs
#
#        if self.total_patents is None or self.total_techs is None:
#            self.total_patents,self.total_techs = self.find_K_number_of_columns(self.AdjList)
#
#
#    def save_to_csv(self,filename):
#        if self.use_codes:
#            tech = 'Codes'
#        else:
#            tech = 'Classes'
#
#        with open(filename,'wb') as fp:
#            fp.write('* Patents {0}'.format(self.total_patents))
#            if self.Patents is not number_of_columnsone:
#                for pat in self.Patents.keys():
#                    if isinstance(pat,str):
#                        fp.write('{0} {1}'.format(self.Patents[pat],pat))
#
#            fp.write('* {0} {1}'.format(tech,self.total_patents))
#            if self.Techs is not number_of_columnsone:
#                for tech in self.Techs.keys():
#                    if isinstance(tech,str):
#                        fp.write('{0} {1}'.format(self.Techs[tech],tech))
#
#            fp.write('* Connections')
#            for entry in range(0,len(self.AdjList)):
#                i = self.AdjList[0]
#                j = self.AdjList[1]
#                fp.write('{0} {1}'.format(i,j))

    #                @staticmethod
    #def read_from_csv(filename):
    #    with open(filename,'r') as fp:
    #        for aline in fp:
    #            aline = aline.strip()
    #            entries = aline.split(' ')
    #
    #            if entries[0]=='*' and entries[1] == 'Patents':
    #
    #
    #
    #@staticmethod
    #def convert_list_to_sparse_matrix(AdjList,K=number_of_columnsone,number_of_columns=None):
    #    if number_of_columns is None or K is None:
    #        K,number_of_columns = PatentTechContainer.find_K_N(AdjList)
    #
    #    A = sparse.lil_matrix(K,number_of_columns)
    #    for entry in AdjList:
    #        i = entry[0]
    #        j = entry[1]
    #        A[i,j] +=1
    #
    #    return A
    #
    #@staticmethod
    #def find_K_N(AdjList):
    #    number_of_columns = 0
    #    K = 0
    #    for entry in AdjList:
    #        i = entry[0]
    #        j = entry[1]
    #
    #        if i > K: K=i
    #        if j > number_of_columns: number_of_columns=j
    #
    #    return K+1,number_of_columns+1
