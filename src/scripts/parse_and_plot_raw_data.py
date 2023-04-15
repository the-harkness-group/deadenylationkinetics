#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################################################
# This is a one-time-use script to parse the raw data files and calculate FRET for the            #
# deadenylation kinetics assay.                                                                   #
#                                                                                                 #
# It is designed to parse a .csv data file from a Synergy Neo2 plate reader from BioTek using the #
# Gen5 software. When exporting results from the plate reader, export as CSV or TXT and name as:  #
#                                                                                                 #
#   CNOT[0-9][X]-##-##uM_###s_rep-#_....csv                                                       #
#   CNOTVARIANT-CONC-CONCuM_TIMEDELAYs_rep-NUMREPLICATES_ANYTHING.csv                             #
#                                                                                                 #
# Arguments include (1) the time file and (2) data files for each enzyme concentration            #
#                                                                                                 #
# Run script as: python_raw_data.py TIMEFILE.csv DATAFILE(S).csv                                  #
###################################################################################################

import sys
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from plotting import make_pdf

data = pd.DataFrame()
search_phrase = ['CNOT[0-9]+[A-Z]*-([0-9-un]*)M','_rep-([0-9]+)','_([0-9]+)s','RNA[0-9]+-([-0-9]+)nM']
all_enz_conc = []

# Load in time arrays based on shaking delays
time_arrays = pd.read_csv(sys.argv[1])
num_files = len(sys.argv)-2
channels = ['IDD','IDA','IAA']
ex = ['485','485','550']
em = ['528','590','590']
# Parse the data files 
for file_num in range(num_files):
    fluor_data = {k:[] for k in channels}   
    at_data = False
    filename = sys.argv[file_num+2]
    with open(filename, 'r') as f:
        for i,line in enumerate(f):
            if line.find("Results") != -1: # Determine if you are "at the start of the data"
                at_data = True
            if at_data == True: # Parse the data by ex/em wavelengths
                for j,channel in enumerate(channels):
                    if line[-10:].find(ex[j]) != -1 and line[-10:].find(em[j]) != -1:
                        m = re.findall(r'([0-9]+)',line)
                        fluor_data[channel].append([int(v) for v in m[:-2]])
    fluor_data ={k:np.transpose(fluor_data[k]) for k in channels} # Transpose the data

    # Calculate FRET
    fluor_data['IAA_scaling'] = [0]*len(fluor_data['IDD'])
    fluor_data['IDA_norm'] = [0]*len(fluor_data['IDD'])
    fret = [0]*len(fluor_data['IDD'])
    for i,v in enumerate(fluor_data['IDD']):
        norm_value = fluor_data['IAA'][i][0]
        fluor_data['IAA_scaling'][i] = [x/norm_value for x in fluor_data['IAA'][i]]
        fluor_data['IDA_norm'][i] = [x/z for x,z in zip(fluor_data['IDA'][i],fluor_data['IAA_scaling'][i])]
        fret[i] = [y/(x+y) for x,y in zip(v,fluor_data['IDA_norm'][i])] # FRET with normed IDA
        # fret[i] = [y/(x+y) for x,y in zip(v,fluor_data['IDA'][i])] # FRET with raw IDA

    # Extract parameters from filename
    parameters = {"enzyme_conc":[],"replicates":[],"time_delay":[],"rna_conc":[]} # Initialize dictionary
    for i,v in enumerate(parameters.keys()): # Loop through dictionary keys
        m = re.findall(search_phrase[i],filename) # Search for the parameter in the filename
        parameters[v] = [int(k) for k in re.findall(r'([0-9]+)',m[0])] # Extract the parameter value

    # Full list of enzyme concentrations
    all_enz_conc = np.concatenate((all_enz_conc,parameters['enzyme_conc']))
   
    # Create dataframe with all data
    for i in range(len(parameters['enzyme_conc'])):
        for u in range(len(parameters['rna_conc'])):
            for j in range(parameters['replicates'][0]):
                temp_data = pd.DataFrame()
                temp_data['time'] = np.concatenate(([0],time_arrays[str(parameters['time_delay'][0])])) # Add time array based on delay with time = 0
                temp_data['fret'] = np.concatenate(([0],fret[j+(i*parameters['replicates'][0])])) # Add FRET array
                temp_data['error'] = [0 for k in range(len(fret[0])+1)] # Add replicate number for each value of FRET
                temp_data['enzyme_conc'] = [parameters['enzyme_conc'][i]/1.0e6 for k in range(len(fret[0])+1)] # Add enzyme concentration for each value
                temp_data['replicate'] = [j+1 for k in range(len(fret[0])+1)] # Add replicate number for each value of FRET
                temp_data['rna_conc'] = [parameters['rna_conc'][u]/1.0e9 for k in range(len(fret[0])+1)] # Add RNA concentration for each value
                for k in channels:
                    temp_data[k] = np.concatenate(([0],fluor_data[k][j+(i*parameters['replicates'][0])])) # add fluorescence data
                data = pd.concat([data,temp_data], ignore_index=True) # Add data to dataframe

# Extract enzyme name from filename
enzyme = re.findall('CNOT[0-9]+[A-Z]*',filename)[0]

# Save data to csv
export_conc = '-'.join(str(int(v)) for v in all_enz_conc)
export_filename = f'{enzyme}_{export_conc}uM_data_for_fit'
data.to_csv(f'{export_filename}.csv')


# Plot FRET data
pdf = make_pdf(f'{export_filename}.pdf')
replicate_shapes = ['s','o','^']
num_conc = len(all_enz_conc)

fret_fig = plt.subplots(num_conc, 1, figsize=(5,num_conc*2.5))
for i in range(num_conc):
    for j in range(parameters['replicates'][0]):
        temp_data = data.loc[(data['enzyme_conc'] == all_enz_conc[i]/1e6) & (data['replicate'] == j+1)]
        p = fret_fig[1][i].plot(temp_data['time'],temp_data['fret'],replicate_shapes[j],label=f'rep. {j+1}') 
    fret_fig[1][i].set_ylabel('FRET')
    fret_fig[1][i].set_ylim(0,1)
    fret_fig[1][i].set_title(f'{all_enz_conc[i]} uM', loc = 'right')
fret_fig[1][0].legend(loc='best')
fret_fig[1][-1].set_xlabel('time (s)')
fret_fig[0].suptitle(f'{enzyme}')

# Plot raw fluorescence data
fluor_fig = plt.subplots(num_conc, 3, figsize=(10,num_conc*2))
for i in range(num_conc):
    for j in range(parameters['replicates'][0]):
        temp_data = data.loc[(data['enzyme_conc'] == all_enz_conc[i]/1e6) & (data['replicate'] == j+1)]
        for k,v in enumerate(channels):
            l=k
            fluor_fig[1][i,k].plot(temp_data['time'],temp_data[v],replicate_shapes[j],label=f'rep. {j+1}')
            if i == 0: fluor_fig[1][0,k].set_title(f"{channels[k]}$_{{{ex[k]},{em[k]}}}$", loc = 'center', y = 1.0)

    fluor_fig[1][i,0].set_ylabel('Fluorescence (a.u.)')
    fluor_fig[1][i,-1].set_title(f'{all_enz_conc[i]} uM', loc = 'right', y = 0.0)
fluor_fig[1][-1,1].set_xlabel('Time (s)')
fluor_fig[0].suptitle(f'{enzyme}')

# Save figure to PDF
fret_fig[0].tight_layout()
fluor_fig[0].tight_layout()
pdf.savefig(fret_fig[0])
pdf.savefig(fluor_fig[0])

pdf.close()
plt.close()
