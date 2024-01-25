#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################################################
# This is a one-time-use script to parse the raw data files and calculate FRET for the            #
# deadenylation kinetics assay.                                                                   #
#                                                                                                 #
# Run script as: python parse_and_plot_raw_data.py parse_data.yaml                                #
###################################################################################################

import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
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
                            enzyme = round(enzyme*1e6,2)/1e6
                            for j,rep in enumerate(FRET):
                                fret_array.append([temp_time[j][i],rep,0,enzyme,1,rna,dna])

# Save fret_array as CSV
temp_df = pd.DataFrame(fret_array, columns=['Time', 'FRET', 'Error','Enzyme', 'Replicate','RNA', 'DNA'])
df = temp_df.sort_values(['RNA', 'Enzyme','Time'], ascending=[True, True, True])
df.to_csv(f'{output_file_name}.csv', index=False)

protein = config_params['Sample name']

rnas = df["RNA"].unique()
enzymes = df['Enzyme'].unique()
times = df["Time"].unique()

mean_df = pd.DataFrame(columns=["Time", "mFRET", "Stdev", "RNA", "Enzyme"])

# Specify the RNA, Enzyme, and Time values
for rna in rnas:
    for enzyme in enzymes:
        for time in times:
            # Filter the dataframe based on the specified values
            filtered_df = df[(df["RNA"] == rna) & (df["Enzyme"] == enzyme) & (df["Time"] == time)]
            if filtered_df.empty:
                continue
            else:
                mean_fret = filtered_df["FRET"].mean()
                stdev_fret = filtered_df["FRET"].std()
                stdev_fret = stdev_fret if not np.isnan(stdev_fret) else 0
                temp_df = pd.DataFrame({"Time": time, "mFRET": mean_fret, "Stdev": stdev_fret, "RNA": rna, "Enzyme": enzyme}, index=[0])
                mean_df = pd.concat([mean_df, temp_df])

points = len(enzymes)
colormap = cm.inferno(np.linspace(1, 0, points+1))
colormap = colormap[1:]

ylim = [-0.05,1.05]
yticks = np.linspace(0,1,6)

len_rna = len(rnas)

if len_rna == 1:
    data_fig, ax = plt.subplots(1,1,figsize=(2+len_rna*5,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
    max_time = round(max(mean_df['Time']),-3)
    if max_time <= 4000:
        max_time = 4000
    elif max_time <= 8000:
        max_time = 8000
    else:
        max_time = max_time
    max_time = 2400
    xlim = [0-0.03*max_time, max_time+0.03*max_time]
    xticks = np.linspace(0,max_time,6)
    for j, enzyme in enumerate(enzymes):
        
            filtered_df = mean_df[(mean_df["RNA"] == rna) & (mean_df["Enzyme"] == enzyme)]
                      
            # color_idx = np.where(colorpoints == round(enzyme*1e5,2))
            # idx_enzyme = round(enzyme*1e5,2)
            # color_idx = np.where(colorpoints == idx_enzyme)[0]
            color_idx = j
            line_label = f"{enzyme*1e6:.1f}"

            ax.errorbar(filtered_df["Time"],filtered_df["mFRET"],yerr=filtered_df["Stdev"],fmt='o',markersize=5,mfc=colormap[color_idx],mec=colormap[color_idx],mew=1,capsize=0,capthick=1,
                            ecolor=colormap[color_idx],label = line_label)
            
            ax.set_title(f"{protein} + {rna*1e9:.0f} nM RNA")
            ax.xaxis.set_ticks(xticks)
            ax.yaxis.set_ticks(yticks)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("mean FRET")
            legend_title = f"[{protein}] (uM)"
            legend = ax.legend(title=legend_title,frameon=False, loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))

else:
    data_fig, axs = plt.subplots(len_rna,1,figsize=(7,len_rna*4)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
    for i, rna in enumerate(rnas):
        temp_data = mean_df[(mean_df["RNA"] == rna)]
        max_time = round(max(temp_data['Time']),-3)
        if max_time <= 4000:
            max_time = 4000
        elif max_time <= 8000:
            max_time = 8000
        xlim = [0-0.03*max_time, max_time+0.03*max_time]
        xticks = np.linspace(0,max_time,6)
        for j, enzyme in enumerate(enzymes):
            filtered_df = mean_df[(mean_df["RNA"] == rna) & (mean_df["Enzyme"] == enzyme)]
            color_idx = j
            line_label = f"{enzyme*1e6:.1f}"

            # FRET data plots
            # axs[i].errorbar(,,color=colormap[color_idx])
            axs[i].errorbar(filtered_df["Time"],filtered_df["mFRET"],yerr=filtered_df["Stdev"],fmt='o',markersize=5,mfc=colormap[color_idx],mec=colormap[color_idx],mew=1,capsize=0,capthick=1,
                            ecolor=colormap[color_idx],label = line_label)
            
            axs[i].set_title(f"{protein} + {rna*1e9:.0f} nM RNA")
            axs[i].xaxis.set_ticks(xticks)
            axs[i].yaxis.set_ticks(yticks)
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].set_ylabel("mean FRET")
            if i == len_rna-1:
                axs[i].set_xlabel("Time (s)")
            if i == 0:
                legend_title = f"[{protein}] (uM)"
                legend = axs[i].legend(title=legend_title,frameon=False, loc='upper left', fontsize=8, bbox_to_anchor=(1.05, 1))

data_fig.tight_layout()

if config_params['Save plot as']['PDF'] == True:
    data_fig.savefig(f"{output_file_name}_plotted.pdf", format='pdf')
if config_params['Save plot as']['PNG'] == True:   
    data_fig.savefig(f"{output_file_name}_plotted.png", format='png', transparent=True, dpi=1200)
