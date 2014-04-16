#__author__ = 'yannis'
#
#import numpy
#import scipy
#import my_stat_tools as mystats
#
#def extract_features_from_activity_series(A,decade_range = range(1790,2020,10)):
#    FEATURE_VECTOR[c][ = dict()
#
#    FEATURE_VECTOR[c][c]['Start_decade'] =  decade_range[numpy.flatnonzero(~numpy.isnan(A))[0]]
#    #FEATURE_VECTOR[c]['First_combination_year']
#    FEATURE_VECTOR[c]['Decade_of_max_activity'] = decade_range[numpy.nanargmax(A)]
#    FEATURE_VECTOR[c]['Decade_of_max_normalised_activity'] = decade_range[numpy.nanargmax(X)]
#
#    FEATURE_VECTOR[c]['Decade_of_max_activity_growth'] = decade_range[numpy.nanargmax(dA)]
#    FEATURE_VECTOR[c]['Decade_of_max_normalised_activity_growth'] = decade_range[numpy.nanargmax(dX)]
#
#    FEATURE_VECTOR[c]['Decade_of_min_activity_growth'] = decade_range[numpy.nanargmin(dA)]
#    FEATURE_VECTOR[c]['Decade_of_min_normalised_activity_growth'] = decade_range[numpy.nanargmin(dX)]
#
#    FEATURE_VECTOR[c]['Flatness_of_activity'] = mystats.time_series_flatness(A)
#    FEATURE_VECTOR[c]['Flatness_of_normalised_activity'] = mystats.time_series_flatness(X)
#
#    FEATURE_VECTOR[c]['Linear_trend'] = scipy.stats.linregress(numpy.array(decade_range)[~numpy.isnan(X)],A[~numpy.isnan(X)])[0]