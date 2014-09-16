__author__ = 'yannis'

import pandas
import numpy
import scipy

###### PARSERS

def read_incidence_matrix_sequence(Bs):
    #expects: ordered list of numpy arrays
    N = Bs[0].shape[0]

    As = []
    Ds = []

    coocurrences_ij = []
    occurrences_i = []
    occurrences_j = []

    TLs = dict()

    for B in Bs:
        A = numpy.dot(B,B.T)
        # sparse?
        A = A - numpy.diag(A.diagonal())
        As.append(A)
        Ds.append(A.sum(0))

    for i in range(0,N-1):
        for j in range(i+1,N):
            ij = tuple([i,j])

            coocurrences_ij = [A[i,j] for A in As]
            occurrences_i = [d[i] for d in Ds]
            occurrences_j = [d[j] for d in Ds]

            TLs[ij] = TemporalLink(coocurrences_ij,occurrences_i,occurrences_j)
            TLs[ij].learn_all_steps()

    return TLs, As
###### MODEL

def p_exp(k,l):
    return (1.*k)/(k+l)

def p_var(k,l):
    return (1.*k*l)/ ((k+l)**2 * (k+l+1))

def a_exp(k,l,x):
    return x*p_exp(k,l)

def a_var(k,l,x):
    return (1.*x*k*l*(k+l+x)) / ((k+l)**2 * (k+l+1))

def a_std(k,l,x):
    return numpy.sqrt(a_var(k,l,x))

def a_exp_from_observed_opportunities(k,l,x):
    return a_exp(k,l,x)

def a_exp_from_predicted_opportunities(k,l,xi,xj = None):
    Ep = p_exp(k,l)
    if xj is not None:
        return (xi+xj)*(Ep/(1+Ep))
    else:
        return xi*Ep

def get_opportunities(aij,xi,xj):
    return xi + xj - aij

def sigmoid(x):
  return 1 / (1 + numpy.exp(-x))


#def watch_and_learn(a_new,xij_new,xi_new,xj_new,kappa,lambd,gamma):
#    # new value
#    p_new = (1.*a_new)/xij_new
#
#    # do prediction
#    p_pred_mean = p_exp(kappa,lambd)
#    p_pred_var = p_var(kappa,lambd)
#
#    # calculate deviation from prediction
#    err = error_function(p_new,p_pred_mean) #square error
#
#    # calculate gain
#    gain = gain_function(err,p_pred_var)
#
#    # calculate new gamma
#    gamma = fusion_function(gain,gamma)
#
#    #update hyperparameters
#    kappa = gamma*kappa + (1-gamma)*a_new
#    lambd = gamma*lambd + (1-gamma)*(xij_new - a_new)
#
#    return kappa,lambd,gamma,p_pred_mean,p_pred_var,err,gain ### ADD SQUARE ERROR, EXP_SQUARE_ERROR, GAIN,

def watch_and_learn(a_new,xij_new,xi_new,xj_new,kappa,lambd,gamma):
    # try a prediction given new xi,xj and kappa, lambda from previous step
    a_pred = a_exp_from_predicted_opportunities(kappa,lambd,xi_new,xj_new)

    # also do a prediction for pijs
    p_pred_mean = p_exp(kappa,lambd)
    p_pred_var = p_var(kappa,lambd)

    # get prediction variance (expected squared error from predictive distribution)
    exp_error = a_var(kappa,lambd,xij_new)

    # calculate deviation from prediction
    err = error_function(a_new,a_pred) #square error

    # calculate gain
    gain = gain_function(err,exp_error)

    # calculate new gamma
    gamma = fusion_function(gain,gamma)

    #update hyperparameters
    kappa = gamma*kappa + (1-gamma)*a_new
    lambd = gamma*lambd + (1-gamma)*(xij_new - a_new)

    return kappa,lambd,gamma,a_pred,exp_error,err,gain,p_pred_mean,p_pred_var

def error_function(new_val,predicted_val):
    return (1.*new_val - predicted_val)**2

def gain_function(err,exp_error):
    return  err / exp_error

def fusion_function(gain,gamma): #it cal also take old gamma
    return 1/(1+gain) #logistic

class TemporalLink:

    def __init__(self,coocurrences_ij,occurrences_i,occurrences_j,timestamps = None, use_degree_as_opportunities = False):
        self.T = len(coocurrences_ij)
        if timestamps is None:
            self.timestamps = range(0,self.T)
        else:
            self.timestamps = timestamps


        self.DATA = pandas.DataFrame({'aijs': coocurrences_ij,
                                      'xis': occurrences_i,
                                      'xjs': occurrences_j},
                                        index=self.timestamps)

        self.PARAMS = pandas.DataFrame({'kappas':  [numpy.nan]*self.T,
                                        'lambdas': [numpy.nan]*self.T,
                                        'gammas': [numpy.nan]*self.T},
                                       index=self.timestamps)

        self.AUX = pandas.DataFrame({'aijs_pred':  [numpy.nan]*self.T,
                                     'aijs_var':  [numpy.nan]*self.T,
                                     'sqr_err':  [numpy.nan]*self.T,
                                     'gain':  [numpy.nan]*self.T,
                                    'pijs_pred':[numpy.nan]*self.T,
                                    'pijs_pred_var': [numpy.nan]*self.T},
                                     index=self.timestamps)

        self.use_degree_as_opportunities = use_degree_as_opportunities

        self.k0 = 10
        self.l0 = 10
        self.g0 = .5

        self.current_step = -1
        self.current_timestamp = -1

#### TIME FUNCTIONS
    def increment_time(self):
        self.current_step +=1
        self.current_timestamp = self.timestamps[self.current_step]
    def get_step_from_timestamp(self,timestamp):
        return self.timestamps.index(timestamp)
    def next_timestamp(self):
        if self.current_step!=self.T-1:
            return self.timestamps[self.current_step+1]
        else:
            return numpy.nan
    def next_timestamp_of_given(self,timestamp):
        if timestamp!=self.timestamps[-1]:
            return self.timestamps[self.get_step_from_timestamp(timestamp) +1]
        else:
            return numpy.nan
    def previous_timestamp(self):
        if self.current_step!=0:
            return self.timestamps[self.current_step-1]
        else:
            return numpy.nan
    def previous_timestamp_of_given(self,timestamp):
        if timestamp!=self.timestamps[0]:
            return self.timestamps[self.get_step_from_timestamp(timestamp) -1]
        else:
            return numpy.nan

#### MODEL LEARNING
    def learn_one_step(self):
        if self.current_step!=-1:
            kappa = self.PARAMS.kappas[self.current_timestamp]
            lambd = self.PARAMS.lambdas[self.current_timestamp]
            gamma = self.PARAMS.gammas[self.current_timestamp]
        else:
            kappa = self.k0
            lambd = self.l0
            gamma = self.g0

        xi = self.DATA.xis[self.next_timestamp()]
        xj = self.DATA.xjs[self.next_timestamp()]

        if (not numpy.isnan(xi)) and  (not numpy.isnan(xj)):
            a_new = self.DATA.aijs[self.next_timestamp()]

            if not self.use_degree_as_opportunities:
                xij_new = get_opportunities(a_new,xi,xj)
                kappa,lambd,gamma,a_pred,exp_error,err,gain,p_pred_mean,p_pred_var = watch_and_learn(a_new,xij_new,xi,xj,kappa,lambd,gamma)
            else:
                kappa,lambd,gamma,a_pred,exp_error,err,gain,p_pred_mean,p_pred_var = watch_and_learn(a_new,xi,xi,None,kappa,lambd,gamma)

            self.AUX.aijs_pred[self.next_timestamp()] = a_pred
            self.AUX.aijs_var[self.next_timestamp()] = exp_error
            self.AUX.sqr_err[self.next_timestamp()] = err
            self.AUX.gain[self.next_timestamp()] = gain
            self.AUX.pijs_pred[self.next_timestamp()] = p_pred_mean
            self.AUX.pijs_pred_var[self.next_timestamp()] = p_pred_var

        self.increment_time()
        self.PARAMS.kappas[self.current_timestamp] = kappa
        self.PARAMS.lambdas[self.current_timestamp] = lambd
        self.PARAMS.gammas[self.current_timestamp] = gamma


    def learn_all_steps(self):
        for t in range(self.current_step,self.T-1):
            self.learn_one_step()

        return self.export_results()

    def export_results(self):
        return pandas.DataFrame({'aijs':self.DATA.aijs,
                                'xis':self.DATA.xis,
                                'xjs':self.DATA.xjs,
                                'xijs':self.get_all_opportunities(),
                                'pijs':self.get_all_simple_ratios(),
                                'pijs_pred':self.get_all_p_exp(),
                                'pijs_pred_var':self.get_all_p_var(),
                                #'a_pred': pandas.Series([self.predict_exp_coocurrences_at_timestamp(t) for t in self.timestamps],index=self.timestamps),
                                'aijs_pred': self.AUX.aijs_pred,
                                'aijs_var': self.AUX.aijs_var,
                                'sqr_err': self.AUX.sqr_err,
                                'gain': self.AUX.gain,
                                #'x_pred': pandas.Series([self.predict_exp_opportunities_at_timestamp(t) for t in self.timestamps],index=self.timestamps),
                                #'kappas': self.PARAMS.kappas,
                                #'lambdas':self.PARAMS.lambdas,
                                'gammas':self.PARAMS.gammas
                                #'SRs':self.get_all_simple_ratios()
                                },
                                index=self.timestamps)

### MODEL PREDICTIONS
    def predict_exp_cooccurrences(self):
        return self.predict_exp_coocurrences_at_timestamp(self.current_timestamp)

    def predict_exp_coocurrences_at_timestamp(self,timestamp):
        if timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[timestamp]
            xj = self.DATA.xjs[timestamp]
            kappa = self.PARAMS.kappas[self.previous_timestamp_of_given(timestamp)]
            lambd = self.PARAMS.lambdas[self.previous_timestamp_of_given(timestamp)]
            if not self.use_degree_as_opportunities:
                return a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
            else:
                return a_exp_from_predicted_opportunities(kappa,lambd,xi)
        else:
            return numpy.nan

#    def predict_var_cooccurrences(self):

    def predict_exp_opportunities(self):
        return self.predict_exp_opportunities_at_timestamp(self.current_timestamp)

    def predict_exp_opportunities_at_timestamp(self,timestamp):
        if timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[timestamp]
            xj = self.DATA.xjs[timestamp]
            if not self.use_degree_as_opportunities:
                kappa = self.PARAMS.kappas[self.previous_timestamp()]
                lambd = self.PARAMS.lambdas[self.previous_timestamp()]
                return xi + xj - a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
            else:
                return xi
        else:
            return numpy.nan

### RETURN PARAMS AND STATISTICS
    def get_opportunities_at_step(self,step):
        timestamp = self.timestamps[step]
        return self.get_opportunities_at_timestamp(timestamp)
    def get_opportunities_at_timestamp(self,timestamp):
        aij = self.DATA.aijs[timestamp]
        xi = self.DATA.xis[timestamp]
        xj = self.DATA.xjs[timestamp]
        return xi + xj - aij

    def get_a_pred_std_at_step(self,step):
        timestamp = self.timestamps[step]
        return self.get_a_pred_std_at_timestamp(timestamp)

    def get_a_pred_std_at_timestamp(self,timestamp):
        timestamp_prev = self.previous_timestamp_of_given(timestamp)
        if numpy.isnan(timestamp_prev):
            return 0
        else:
            return a_std(self.PARAMS.kappas[timestamp_prev],self.PARAMS.lambdas[timestamp_prev],self.PARAMS.xijs[timestamp_prev])

    def get_p_exp_at_step(self,step):
        timestamp = self.timestamps[step]
        return self.get_p_exp_at_timestamp(timestamp)
    def get_p_exp_at_timestamp(self,timestamp):
        return (1.*self.PARAMS.kappas[timestamp])/(self.PARAMS.kappas[timestamp] + self.PARAMS.lambdas[timestamp])

    def get_all_p_exp(self):
        return (1.*self.PARAMS.kappas)/(self.PARAMS.kappas + self.PARAMS.lambdas)

    def get_all_p_var(self):
        return (1.*self.PARAMS.kappas*self.PARAMS.lambdas)/ ((self.PARAMS.kappas+self.PARAMS.lambdas)**2 * (self.PARAMS.kappas+self.PARAMS.lambdas+1))

    def get_all_opportunities(self):
        if self.use_degree_as_opportunities:
            return self.DATA.xis
        else:
            return self.DATA.xis + self.DATA.xjs - self.DATA.aijs

    def get_all_simple_ratios(self):
        return (1.*self.DATA.aijs) / self.get_all_opportunities()

    def get_all_dependencies_i_to_j(self):
        return (1.*self.DATA.aijs) / self.DATA.xis

    def get_all_dependencies_j_to_i(self):
        return (1.*self.DATA.aijs) / self.DATA.xjs

##### DRAW SAMPLES FROM THE MODEL

    def draw_sample_coocurrences(self,xi,xj,xij = None,timestamp=None):
        if timestamp is None:
            timestamp = self.current_timestamp

        kappa = self.PARAMS.kappas[timestamp]
        lambd = self.PARAMS.lambdas[timestamp]

