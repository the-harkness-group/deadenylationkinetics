#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
import yaml

config_params = yaml.safe_load(open(sys.argv[1],'r'))

output_file_name = config_params['Output file name']
data_to_load = config_params['Data to load']
time_arrays = np.transpose(np.array(pd.read_csv(config_params['Time arrays'], header=0)))

# Initialize an empty array to store FRET values
fret_array = []

# Load CSV file

for i,file in enumerate(data_to_load):
    dataload = np.array(pd.read_csv(data_to_load[file], header=None))
    
    # Save row 2 of data starting at column 3 as enzyme_conc
    enzyme_conc = dataload[1][2:]
    unique_enzyme_conc = np.unique(enzyme_conc)

    # Save row 3 of data starting at column 3 as cap1_conc
    cap1_conc = dataload[2][2:]
    unique_cap1_conc = np.unique(cap1_conc)

    # Save row 4 of data starting at column 3 as rna_conc
    rna_conc = dataload[3][2:]
    unique_rna_conc = np.unique(rna_conc)

    # Save row 5 of data starting at column 3 as dna_conc
    dna_conc = dataload[4][2:]
    unique_dna_conc = np.unique(dna_conc)

    # Save first column of data as point from row 7 onwards
    points = np.transpose(dataload)[0][6:]

    # Convert points to integers
    points = points.astype(int)
    unique_points = np.unique(points)

    # Save row 6 of data starting at column 3 as time array index
    time_index = dataload[5][2:].astype(int)

    # Save column 2 of data as channels from row 7 onwards
    channels = np.transpose(dataload)[1][6:]
    unique_channels = np.unique(channels)

    # Save data starting at row 7 and column 3 as data
    data = np.transpose(dataload[6:,2:])
 
    for dna in unique_dna_conc:
        if dna == 0:
            continue
        else:
            for rna in unique_rna_conc:
                for cap1 in unique_cap1_conc:
                    for enzyme in unique_enzyme_conc:
                        temp_data = np.transpose(data[np.all([enzyme_conc == enzyme, cap1_conc == cap1, rna_conc == rna, dna_conc == dna],axis=0 ),:])
                        temp_time_idx = time_index[np.all([enzyme_conc == enzyme, cap1_conc == cap1, rna_conc == rna, dna_conc == dna],axis=0 )]
                        temp_time = time_arrays[temp_time_idx-1]
                        if enzyme == 0:
                            blank = np.mean(temp_data, axis=1)
                            blank_IDD_start = blank[np.all([channels == 'DD',points == 1],axis=0)]
                            blank_IDA_start = blank[np.all([channels == 'DA',points == 1],axis=0)]
                        for i,point in enumerate(unique_points):
                            IDD_idx = np.all([channels == 'DD',points == point],axis=0)
                            IDA_idx = np.all([channels == 'DA',points == point],axis=0)
                            temp_IDD = temp_data[IDD_idx, :][0]
                            temp_IDA = temp_data[IDA_idx, :][0]
                            temp_IDD_blank = blank[IDD_idx]
                            temp_IDA_blank = blank[IDA_idx]
                            norm_IDD = temp_IDD - abs(blank_IDD_start-temp_IDD_blank)
                            norm_IDA = temp_IDA - abs(blank_IDA_start-temp_IDA_blank)
                            FRET = norm_IDA/(norm_IDD+norm_IDA)
                            for j,rep in enumerate(FRET):
                                fret_array.append([temp_time[j][i],rep,0,enzyme,1,rna,dna])

# Save fret_array as CSV
temp_df = pd.DataFrame(fret_array, columns=['Time', 'FRET', 'Error','Enzyme', 'Replicate','RNA', 'DNA'])
df = temp_df.sort_values(['RNA', 'Enzyme','Time'], ascending=[True, True, True])
df.to_csv(f'{output_file_name}.csv', index=False)