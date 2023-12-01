#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import yaml
from utils import load_data, setup_parameters, write_optimal_parameter_csv
from experiment import FretExperiment
from models import generate_model_objects, simulate_full_model, calculate_residuals_simulate_best_fit_data
from plotting import PlotHandler
from minimization import objective_wrapper, residuals
from lmfit import Parameters, minimize, report_fit
from error_analysis import ErrorAnalysis
from icecream import ic

############################### TO DO #################################
# - Check first point in fitting
# - Deepcopy pandas df in Experiment and Models
# - Move FRET normalization to Experiment class, or somewhere smarter
# - Fix replicate stuff in don't fit, use individual replicates block
#######################################################################

def main():

    # Get data, set up fit parameters, constants, etc.
    config_params, data = load_data(sys.argv[1])
    hybridization_params, initial_guess_params, varied_params, opt_params = setup_parameters(config_params, Parameters())

    minimizer_params = []
    experiments = []
    kinetic_models = []
    hybridization_models = []

    # Run fit, either sequential fitting of individual replicates or average of replicates
    if config_params['Modeling parameters']['Fit'] == True:
        min_method = config_params['Modeling parameters']['Minimizer']['Initial']
        
        if config_params['Modeling parameters']['Use individual replicates'] == True:
            print("***** Fitting replicate experiments sequentially *****\n")
            replicate_groups = data.groupby('Replicate')
           
            for i, (ind, group) in enumerate(replicate_groups):
                experiment = FretExperiment(group, hybridization_params)
                kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'])
                experiments.append(experiment)
                kinetic_models.append(kinetic_model)
                hybridization_models.append(hybridization_model)
                
                if i != 0:
                    initial_guess_params = minimizer_params[0]
                    min_method = config_params['Modeling parameters']['Minimizer']['Subsequent']
                
                minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, args=(experiment, kinetic_model, hybridization_model, simulate_full_model))
                report_fit(minimizer_result)
                minimizer_params.append(minimizer_result.params)

            print('')
            print('### OPTIMAL REPLICATE FIT PARAMETERS, AVG +/- STD ###')
            # Extract final parameters from each replicate and average them
            for i, rep_parameters in enumerate(minimizer_params):
                for i, parameter in enumerate(rep_parameters):
                    if parameter in opt_params.keys():
                        opt_params[parameter].append(rep_parameters[parameter].value)
            for i, k in enumerate(varied_params):
                opt_params[f"{k} Error"] = np.std(np.abs(opt_params[k]))
                opt_params[k] = np.mean(np.abs(opt_params[k]))
                
                v = opt_params[k]
                err = opt_params[f"{k} Error"]
                print(f"{k}: {v} +/- {err}")
            
            # Simulate best fit data and plot, need to do it for each replicate
            resids, normalized_resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)

        else:
            print(f"***** Fitting the average of the replicates with {min_method} method *****\n")
            experiment = FretExperiment(data, hybridization_params)
            kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'])
            experiments.append(experiment)
            kinetic_models.append(kinetic_model)
            hybridization_models.append(hybridization_model)

            minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, args=(experiment, kinetic_model, hybridization_model, simulate_full_model))
            report_fit(minimizer_result)
            minimizer_params.append(minimizer_result.params)

            for i, k in enumerate(minimizer_result.params):
                if k in opt_params.keys():
                    opt_params[k] = minimizer_result.params[k].value
                    opt_params[f"{k} Error"] = minimizer_result.params[k].stderr

            if config_params['Modeling parameters']['Error estimation']['Monte Carlo']['Run'] == True:
                monte_carlo_iterations = config_params['Modeling parameters']['Error estimation']['Monte Carlo']['Iterations']
                rmsd = np.sqrt(minimizer_result.chisqr/minimizer_result.ndata)
                error_analyzer = ErrorAnalysis(minimizer_result.params, monte_carlo_iterations, rmsd, None, None)
                error_analyzer.monte_carlo_parameter_dictionary()
                error_analyzer.monte_carlo_fits(experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
                error_analyzer.monte_carlo_distributions(config_params['Sample name'])
                error_analyzer.save_monte_carlo_results(config_params['Sample name'])

            if config_params['Modeling parameters']['Error estimation']['Error surfaces']['Run'] == True:
                range_factor = config_params['Modeling parameters']['Error estimation']['Error surfaces']['Parameter range factor']
                points = config_params['Modeling parameters']['Error estimation']['Error surfaces']['Points']
                error_analyzer = ErrorAnalysis(minimizer_result.params, None, None, range_factor, points)
                error_analyzer.correlation_pairs()
                error_analyzer.parameter_correlation_fits(experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
                error_analyzer.parameter_correlation_surfaces(config_params['Sample name'])
                error_analyzer.save_parameter_correlation_results(config_params['Sample name'])

            # Simulate best fit data and plot, only for average data
            resids, normalized_resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)

        # Save best parameters in .csv in case needed for something else
        opt_param_units = [config_params['Modeling parameters']['Fit parameters'][k]['Units'] for k in config_params['Modeling parameters']['Fit parameters'].keys() if config_params['Modeling parameters']['Fit parameters'][k]['Vary'] == True]
        write_optimal_parameter_csv(opt_params, opt_param_units, config_params['Optimal fit parameter file'])
    
    # Simulate with input parameters, e.g. to check if parameters are reasonable before trying fit
    elif config_params['Modeling parameters']['Fit'] == False:
        if config_params['Modeling parameters']['Use individual replicates'] == True:
            print("***** Simulating experiment replicates *****\n")
            # DO REPLICATE STUFF

        else:
            print(f"***** Simulating average of the experiment replicates *****\n")
            # Simulate best fit data and plot, only for average data
            experiment = FretExperiment(data, hybridization_params)
            kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'])
            experiments.append(experiment)
            kinetic_models.append(kinetic_model)
            hybridization_models.append(hybridization_model)
            minimizer_params.append(initial_guess_params)
            resids, normalized_resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)
            print(f'RSS for simulated data: {np.sum(np.square(resids))}')
    
    # Do plotting

    # Plotting annealed fraction to check for good data
    # ic(best_hybr_models[1].annealed_fraction[1])
    # ic(best_kin_models[1].time[1])

    best_fit_flag = config_params['Plot parameters']['Plot best fit']
    residual_flag = config_params['Plot parameters']['Plot residuals']
    RNA_populations_flag = config_params['Plot parameters']['Plot RNA population curves']
    annealed_fraction_flag = config_params['Plot parameters']['Plot annealed fraction']
    bar_2d_flag = config_params['Plot parameters']['Plot 2D population bars']
    bar_3d_flag = config_params['Plot parameters']['Plot 3D population bars']
    sample_name = config_params['Sample name']
    plot_name = config_params['Output plot file']
    plot_handler = PlotHandler(experiments, best_kin_models, best_hybr_models, resids, normalized_resids, sample_name, plot_name, best_fit_flag, residual_flag, RNA_populations_flag, annealed_fraction_flag, bar_2d_flag, bar_3d_flag)
    plot_handler.run_plots()

if __name__ == '__main__':
    main()
