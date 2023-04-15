#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def objective_wrapper(params, experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_params=True):

    kinetic_model, hybridization_model = simulate_full_model(params, kinetic_model, hybridization_model)
    resid = residuals(experiment.fret, hybridization_model.fret)
    resid = np.ravel(resid)
    rss = sum_of_squared_residuals(resid)
    
    if print_current_params == True:
        print(f'Current fit RSS: {rss:.10f}')
        print(f'Current fit parameters:')
        params.pretty_print(colwidth=10, columns=['value', 'min', 'max', 'vary'])
        print('')
    
    return resid

def residuals(ydata, predicted):

    resid = []
    for i, v in enumerate(ydata):
        resid.append((ydata[i][1:] - predicted[i][1:])) # From 1: because 0 index are dummy t=0 values in the experimental dataset that we don't actually have FRET values for due to dead time

    return resid

def sum_of_squared_residuals(residuals):

    rss = np.sum(np.square(residuals))

    return rss