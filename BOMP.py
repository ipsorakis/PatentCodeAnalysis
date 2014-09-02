__author__ = 'yannis'

import pandas
import numpy
import scipy

def p_exp(k,l):
    return (1.*k)/(k+l)

def a_exp(k,l,x):
    return x*p_exp(k,l)

def a_var(k,l,x):
    return (1.*x*k*l*(k+l+x)) / ((k+l)**2 * (k+l+1))

def a_std(k,l,x):
    return numpy.sqrt(a_var(k,l,x))

def a_exp_from_observed_opportunities(k,l,x):
    return a_exp(k,l,x)
def a_exp_from_predicted_opportunities(k,l,xi,xj):
    Ep = p_exp(k,l)
    return (xi+xj)*(Ep/(1+Ep))

def get_opportunities(aij,xi,xj):
    return xi + xj - aij

def sigmoid(x):
  return 1 / (1 + numpy.exp(-x))


def watch_and_learn(a_new,xij_new,xi_new,xj_new,kappa,lambd,gamma):
    # try a prediction given new xi,xj
    a_pred = a_exp_from_predicted_opportunities(kappa,lambd,xi_new,xj_new)
    
    # calculate deviation from prediction
    err = error_function(a_new,a_pred)

    # calculate gain
    #gamma = gain(err,xij_new)
    gamma = gain(err,gamma,xij_new,kappa,lambd)

    #update hyperparameters
    kappa = gamma*kappa + (1-gamma)*a_new
    lambd = gamma*lambd + (1-gamma)*(xij_new - a_new)

    return kappa,lambd,gamma

def error_function(a_new,a_pred):
    return (1.*a_new - a_pred)**2
    #return 1.*a_new - a_pred

def gain(err,gamma,x,kappa,lambd):
    #gamma  *= (1 + sigmoid(err/a_std(kappa,lambd,x)))/2
    #print 'err = {0:.2f}, std = {1:.2f}, change = {2:.2f}'.format(err,a_std(kappa,lambd,x),err/a_std(kappa,lambd,x))
    #gamma = (gamma + sigmoid(a_var(kappa,lambd,x)/err))/2
    #gamma = (gamma + sigmoid(a_var(kappa,lambd,x)/err - numpy.exp(1)))/2.

    #gamma = (gamma + sigmoid(numpy.log(a_var(kappa,lambd,x)/err)))/2.
    gamma = sigmoid(numpy.log(a_var(kappa,lambd,x)/err))
    print 'err = {0:.2f}, var = {1:.2f}, change = {2:.2f}, gamma = {3:.2f}'.format(err,a_var(kappa,lambd,x),a_var(kappa,lambd,x) / err,gamma)
    #if gamma>1:
    #    gamma=1-
#def gain(err,gamma,x)
    #

    return gamma

class TemporalLink:

    def __init__(self,coocurrences_ij,occurrences_i,occurrences_j,timestamps = None):
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

        self.k0 = 1
        self.l0 = 1
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
            xij_new = get_opportunities(a_new,xi,xj)

            kappa,lambd,gamma = watch_and_learn(a_new,xij_new,xi,xj,kappa,lambd,gamma)

        self.increment_time()
        self.PARAMS.kappas[self.current_timestamp] = kappa
        self.PARAMS.lambdas[self.current_timestamp] = lambd
        self.PARAMS.gammas[self.current_timestamp] = gamma

    def learn_all_steps(self):
        for t in range(self.current_step,self.T-1):
            self.learn_one_step()

        return pandas.DataFrame({'aijs':self.DATA.aijs,
                                'xis':self.DATA.xis,
                                'xjs':self.DATA.xjs,
                                'xijs':self.get_all_opportunities(),
                                'p_exp':self.get_all_p_exp(),
                                'a_pred': pandas.Series([self.predict_exp_coocurrences_at_timestamp(t) for t in self.timestamps],index=self.timestamps),
                                'x_pred': pandas.Series([self.predict_exp_opportunities_at_timestamp(t) for t in self.timestamps],index=self.timestamps),
                                'kappas': self.PARAMS.kappas,
                                'lambdas':self.PARAMS.lambdas,
                                'gammas':self.PARAMS.gammas,
                                'SRs':self.get_all_simple_ratios()
                                },
                                index=self.timestamps)


### MODEL PREDICTIONS
    def predict_exp_cooccurrences(self):
        if self.current_timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[self.current_timestamp]
            xj = self.DATA.xjs[self.current_timestamp]
            kappa = self.PARAMS.kappas[self.previous_timestamp()]
            lambd = self.PARAMS.lambdas[self.previous_timestamp()]
            return a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
        else:
            return numpy.nan

    def predict_exp_coocurrences_at_timestamp(self,timestamp):
        if timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[timestamp]
            xj = self.DATA.xjs[timestamp]
            kappa = self.PARAMS.kappas[self.previous_timestamp_of_given(timestamp)]
            lambd = self.PARAMS.lambdas[self.previous_timestamp_of_given(timestamp)]
            return a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
        else:
            return numpy.nan

#    def predict_var_cooccurrences(self):

    def predict_exp_opportunities(self):
        if self.current_timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[self.current_timestamp]
            xj = self.DATA.xjs[self.current_timestamp]
            kappa = self.PARAMS.kappas[self.previous_timestamp()]
            lambd = self.PARAMS.lambdas[self.previous_timestamp()]
            return xi + xj - a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
        else:
            return numpy.nan

    def predict_exp_opportunities_at_timestamp(self,timestamp):
        if timestamp!=self.timestamps[0]:
            xi = self.DATA.xis[timestamp]
            xj = self.DATA.xjs[timestamp]
            kappa = self.PARAMS.kappas[self.previous_timestamp_of_given(timestamp)]
            lambd = self.PARAMS.lambdas[self.previous_timestamp_of_given(timestamp)]
            return xi + xj - a_exp_from_predicted_opportunities(kappa,lambd,xi,xj)
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

    def get_all_opportunities(self):
        return self.DATA.xis + self.DATA.xjs - self.DATA.aijs

    def get_all_simple_ratios(self):
        return (1.*self.DATA.aijs) / self.get_all_opportunities()

    def get_all_dependencies_i_to_j(self):
        return (1.*self.DATA.aijs) / self.DATA.xis

    def get_all_dependencies_j_to_i(self):
        return (1.*self.DATA.aijs) / self.DATA.xjs