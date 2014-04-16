__author__ = 'yannis'

import numpy
import scipy
import scipy.sparse as sparse

def iBP_random_sample(N,a,b=1):

    Z = []
    i = 0
    K = 0
    mk = dict()

    while i<N:
        total_dishes_sampled_by_current_customer = 0
        if i==0:
            # try a number of new dishes:
            while K==0:K = numpy.random.poisson(a)
            for j in range(0,K):
                Z.append([i,j])
                mk[j] = 1
                total_dishes_sampled_by_current_customer+=1
        else:
            # A) try existing dishes
            for j in range(0,K):
                p=float(mk[j])/(b-1+i+1)
                #print 'i:{0}, k:{1}, mk:{2} , b+i={3}, p = {4}, total_dishes_sampled_by_current_customer={5}'.format(i,j,mk[j],b+i,p,total_dishes_sampled_by_current_customer)
                pick_dish = numpy.random.binomial(1,p)
                if pick_dish:
                    Z.append([i,j])
                    mk[j] +=1
                    total_dishes_sampled_by_current_customer+=1
            # B) try a number of new dishes:
            new_cols = numpy.random.poisson(float(a*b)/(b-1+i+1))
            for j in range(0,new_cols):
                Z.append([i,K+j])
                mk[K+j] = 1
                total_dishes_sampled_by_current_customer+=1
            K+=new_cols

        if total_dishes_sampled_by_current_customer!=0:i+=1

    OUTPUT =  convert_list_to_sparse_matrix(Z,N,K)
    return OUTPUT

def iBP_random_sample_given_max_K(MAX_FEATURES,a,b=1):

    Z = []
    i = 0
    K = 0
    new_cols = 0
    mk = dict()

    while K<MAX_FEATURES:
        total_dishes_sampled_by_current_customer = 0
        if i==0:
            # try a number of new dishes:
            while new_cols==0:new_cols = numpy.random.poisson(a)
            for j in range(0,new_cols):
                Z.append([i,j])
                mk[j] = 1
                total_dishes_sampled_by_current_customer+=1
                K+=1
                if K==MAX_FEATURES:
                    break
        else:
            # A) try existing dishes
            for j in range(0,K):
                p=float(mk[j])/(b-1+i+1)
                #print 'i:{0}, k:{1}, mk:{2} , b+i={3}, p = {4}, total_dishes_sampled_by_current_customer={5}'.format(i,j,mk[j],b+i,p,total_dishes_sampled_by_current_customer)
                pick_dish = numpy.random.binomial(1,p)
                if pick_dish:
                    Z.append([i,j])
                    mk[j] +=1
                    total_dishes_sampled_by_current_customer+=1
            # B) try a number of new dishes:
            new_cols = numpy.random.poisson(float(a*b)/(b-1+i+1))
            for j in range(0,new_cols):
                Z.append([i,K])
                mk[K] = 1
                K+=1
                total_dishes_sampled_by_current_customer+=1
                if K == MAX_FEATURES:
                    break

        if total_dishes_sampled_by_current_customer!=0:i+=1

    OUTPUT =  convert_list_to_sparse_matrix(Z,i+1,K)
    return OUTPUT



def convert_list_to_sparse_matrix(AdjList,number_of_rows=None,number_of_columns=None):
    if number_of_columns is None or number_of_rows is None:
        number_of_columns = 0
        number_of_rows = 0
        for entry in AdjList:
            i = entry[0]
            j = entry[1]

            if i > number_of_rows: number_of_rows=i
            if j > number_of_columns: number_of_columns=j

    A = sparse.lil_matrix((number_of_rows,number_of_columns))
    for entry in AdjList:
        i = entry[0]
        j = entry[1]
        A[i,j] +=1

    return A

def iBT_pdf(Z,a,b=1):
    N,K = Z.shape
    Kplus = 0

    mk = [sum(Z[:,i]) for i in range(0,K)]

    prod = 1
    for k in range(0,K):
        if mk[k] !=0:
            prod*=scipy.special.beta(mk[k],N-mk[k]+b)
            Kplus+=1
    return 1