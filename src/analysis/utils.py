#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys
import yaml

def load_data(configuration_file):

    config_params = yaml.safe_load(open(configuration_file,'r'))
    replicate_df = pd.read_csv(config_params['Data file to fit'])
    individual_replicates = config_params['Modeling parameters']['Use individual replicates']

    if individual_replicates == False: # Calculate average of replicates
        from copy import deepcopy
        replicate_groups = deepcopy(replicate_df.groupby(['Replicate'])) # Split dataset into replicates and average them
        avg_dict ={'Time':[],'FRET':[],'Enzyme':[],'RNA':[],'Error':[],'Replicate':[]}
        for ind, group in replicate_groups:
            avg_dict['Time'].append(group.Time.values)
            avg_dict['FRET'].append(group.FRET.values)
            avg_dict['Enzyme'].append(group.Enzyme.values)
            avg_dict['RNA'].append(group.RNA.values)
        avg_dict['Error'] = np.std(avg_dict['FRET'],0)
        avg_dict['Time'] = np.mean(avg_dict['Time'],0)
        avg_dict['FRET'] = np.mean(avg_dict['FRET'],0)
        avg_dict['Enzyme'] = np.mean(avg_dict['Enzyme'],0)
        avg_dict['RNA'] = np.mean(avg_dict['RNA'],0)
        avg_dict['Replicate'] = ['Average' for t in avg_dict['Time']]
        avg_df = pd.DataFrame(avg_dict)

        return config_params, avg_df

    else:
        return config_params, replicate_df

def setup_parameters(config_params, initial_guess_params):

    hybridization_params = {k:config_params['Experimental parameters'][k]['Value'] for k in ['QT', 'n', 'Temperature']}
    hybridization_params['dGo'] = config_params['Modeling parameters']['Fit parameters']['dGo']['Value']
    hybridization_params['alpha'] = config_params['Modeling parameters']['Fit parameters']['alpha']['Value']

    for k in config_params['Modeling parameters']['Fit parameters'].keys():
        initial_guess_params.add(k, value = config_params['Modeling parameters']['Fit parameters'][k]['Value'], vary = config_params['Modeling parameters']['Fit parameters'][k]['Vary'], 
        min = config_params['Modeling parameters']['Fit parameters'][k]['Minimum'])

    varied_params = [k for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True]
    opt_params = {k:[] for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True}

    return  hybridization_params, initial_guess_params, varied_params, opt_params

def create_experiment_dataframe(time, avgFRET, stdFRET, E0, S0):

    data_dict = {'Time':[],'FRET':[],'Error':[], 'Enzyme':[], 'RNA':[]} # Input data is list of lists, needs to be unraveled into one list for making data frame
    data_dict['Time'] = [t for time_vector in time for t in time_vector]
    data_dict['FRET'] = [fret for fret_vector in avgFRET for fret in fret_vector]
    data_dict['Error'] = [fret for fret_vector in stdFRET for fret in fret_vector]
    data_dict['Enzyme'] = [E0[i] for i,time_vector in enumerate(time) for t in time_vector] # Match each time point with corresponding enzyme and RNA concentration
    data_dict['RNA'] = [S0[i] for i,time_vector in enumerate(time)  for t in time_vector]

    return pd.DataFrame(data_dict)

def make_dictionary(key_list, value_list):

    made_dictionary = {k:v for k,v in zip(key_list,value_list)}

    return made_dictionary

def write_optimal_parameter_csv(opt_params, opt_param_units, file):

    opt_params_dict = {'Parameter':[k for k in opt_params.keys() if 'Error' not in k], 'Value':[opt_params[k] for k in opt_params if 'Error' not in k],
    'Error':[opt_params[k] for k in opt_params.keys() if 'Error' in k], 'Units':[i for i in opt_param_units]}
    
    opt_params_df = pd.DataFrame(opt_params_dict)
    opt_params_df.to_csv(file)