#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

"Ref. Banerjee, D. et al. Improved nearest-neighbor parameters for the stability of RNA/DNA hybrids under"
"a physiological condition. Nucleic Acids Res. 48(21):12042-54. 2020."

def main():
    L = 1e-7 # ligand concentration in M (100 nM)
    RNA_tag_sequence = 'CCUUUCC'
    polyA_sequence = [i*'A' for i in range(0,19)]
    numA = [i for i in range(0,19)]
    RNA_sequences = [RNA_tag_sequence + i for i in polyA_sequence]
    
    nn_pairs = make_nn_pairs(RNA_sequences)
    nn_thermo = nn_thermodynamics()
    temperature = 30 + 273.15
    dG_dict = calculate_dG(nn_pairs, numA, temperature, nn_thermo)
    # for k in dG_dict.keys():
    #     print(f"r{k}: dG RNA/DNA duplex = {dG_dict[k]['dG']} kcal/mol @ {temperature} K")
    binding_dict = fraction_bound(dG_dict, L, temperature)

    write_dG_csv(dG_dict, temperature)
    plot_df = plot_dict(dG_dict, binding_dict, temperature)
    plot_dG(plot_df)
    plot_Kd_frac(plot_df)


def make_nn_pairs(sequences):

    nn_pairs = {sequence:[sequence[i:i+2] for i in range(len(sequence[:-1]))] for sequence in sequences}

    return nn_pairs


def nn_thermodynamics():

    nn_pairs = ['rAA/dTT','rAC/dGT','rAG/dGT','rAU/dAT','rCA/dTG','rCC/dGG','rCG/dCG',
    'rCU/dAG','rGA/dTC','rGC/dGC','rGG/dCC','rGU/dAC','rUA/dTA','rUC/dGA','rUG/dCA','rUU/dAA']
    nn_dH = [-7.8,-10.1,-9.4,-5.8,-9.8,-9.5,-9.0,-6.1,-8.6,-10.6,-13.3,-9.3,-6.6,-6.5,-8.9,-7.4] # kcal/mol
    nn_dS = [-22.9,-27.3,-26.2,-17.5,-27.4,-24.8,-24.3,-17.9,-22.7,-27.7,-35.7,-25.5,-19.7,-16.3,-23.3,-24.3] # cal /mol /K
    nn_thermo = {nn_pairs[i]:{'dH':nn_dH[i],'dS':nn_dS[i]} for i in range(len(nn_pairs))}
    
    return nn_thermo


def calculate_dG(nn_pairs, numA, temperature, nn_thermo):

    dG_dict = {sequence:{'dG':None,'numA':numA[i]} for i,sequence in enumerate(nn_pairs.keys())}
    for sequence in nn_pairs.keys():

        dH_sum = 0
        dS_sum = 0

        for nn_pair in nn_pairs[sequence]:

            for k in nn_thermo.keys():

                if 'r' + nn_pair + '/' in k:
            
                    dH_sum += nn_thermo[k]['dH']
                    dS_sum += nn_thermo[k]['dS']

        dG = dH_sum - temperature*(dS_sum/1000) # /1000 to convert cal /mol /K to kcal /mol /K to match dH
        dG_dict[sequence]['dG'] = dG
    

    return dG_dict


def fraction_bound(dG_dict, L, temperature):

    R = 1.987e-3 # kcal / mol / K

    K_dict = {k:np.exp(dG_dict[k]['dG']/(R*temperature)) for k in dG_dict.keys()}

    maxy = 1
    fraction_bound_dict = {k:(maxy*L)/(L+K_dict[k]) for k in K_dict.keys()}
    binding_dict = {k:{'Kd':K_dict[k],'fraction bound':fraction_bound_dict[k]} for k in K_dict.keys()}

    csv_dict = {'polyA length':[k.count('A') for k in dG_dict.keys()],'Kd M':[K_dict[k] for k in K_dict.keys()]}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv('RNA_DNA_hybrid_duplex_Kd.csv',index=False)

    return binding_dict


def write_dG_csv(dG_dict, temperature):

    csv_dict = {'RNA/DNA duplex sequence':[k for k in dG_dict.keys()],'dG kcal/mol':[dG_dict[k]['dG'] for k in dG_dict.keys()],'Temperature K':[temperature for k in dG_dict.keys()],
    'polyA length':[k.count('A') for k in dG_dict.keys()]}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv('RNA_DNA_hybrid_duplex_dG.csv',index=False)


def plot_dict(dG_dict, binding_dict, temperature):
    plot_dict = {'RNA/DNA duplex sequence':[k for k in dG_dict.keys()],'dG kcal/mol':[dG_dict[k]['dG'] for k in dG_dict.keys()],'Temperature K':[temperature for k in dG_dict.keys()],
    'polyA length':[k.count('A') for k in dG_dict.keys()],'Kd':[binding_dict[k]['Kd'] for k in binding_dict.keys()],'fraction bound':[binding_dict[k]['fraction bound'] for k in binding_dict.keys()]}
    plot_df = pd.DataFrame(plot_dict)

    return plot_df


def plot_dG(plot_df):

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(plot_df['polyA length'],plot_df['dG kcal/mol'],color='k')
    plt.xlabel('polyA length')
    plt.ylabel('RNA/DNA duplex free energy (kcal/mol)')
    plt.title(f"RNA/DNA duplex free energy vs. polyA length @ {plot_df['Temperature K'][0]} K")
    plt.savefig('RNA_DNA_hybrid_duplex_dG.pdf')


def plot_Kd_frac(plot_df):

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.55, 0.3, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])

    ax1.plot(plot_df['polyA length'],plot_df['fraction bound'],color='k',marker='o',linestyle='none',label='RNA-DNA duplex')
    ax1.plot(plot_df['polyA length'],1-plot_df['fraction bound'],color='grey',marker='^',linestyle='none',label='Free RNA')
    ax1.set_xticks(range(0,21,4))
    ax1.set_xlabel('RNA tag with # As')
    ax1.set_ylabel('Fraction')
    ax1.tick_params(axis='both', direction='in')
    ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 0.4, 0.5, 0.5),frameon=False)


    ax2.plot(plot_df['polyA length'],plot_df['Kd'],color='k',marker='.',linestyle='none')
    ax2.set_xticks(range(0,21,4))
    ax2.set_yscale('log')
    ax2.set_xlabel('RNA tag with # As')
    ax2.set_ylabel(r'Duplex $K_{D}$ (M)')
    ax2.tick_params(axis='both', direction='in')

    plt.savefig('RNA_DNA_frac_bound_duplex_Kd.pdf')


main()
