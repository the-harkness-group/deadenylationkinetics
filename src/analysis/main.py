#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import load_data, setup_parameters, write_optimal_parameter_csv
from experiment import FretExperiment
from models import generate_model_objects, simulate_full_model, calculate_residuals_simulate_best_fit_data
from plotting import PlotHandler
from minimization import objective_wrapper, residuals, sum_of_squared_residuals
from lmfit import Parameters, minimize, report_fit
from error_analysis import ErrorAnalysis
import os


def main():

    # Get data, set up fit parameters, constants, etc.
    config_params, data = load_data(sys.argv[1])
    hybridization_params, initial_guess_params, varied_params, opt_params = setup_parameters(config_params, Parameters())

    minimizer_params = []
    experiments = []
    kinetic_models = []
    hybridization_models = []

    # Create the output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run fit, either sequential fitting of individual replicates or average of replicates
    if config_params['Modeling parameters']['Fit'] == True:
        min_method = config_params['Modeling parameters']['Minimizer']

        print("\n### Running data fits ###")
        experiment = FretExperiment(data, hybridization_params)
        kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'])
        experiments.append(experiment)
        kinetic_models.append(kinetic_model)
        hybridization_models.append(hybridization_model)
        
        minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, args=(experiment, kinetic_model, hybridization_model, simulate_full_model))
        report_fit(minimizer_result)
        minimizer_params.append(minimizer_result.params)

        param_units = [config_params['Modeling parameters']['Fit parameters'][k]['Units'] for k in config_params['Modeling parameters']['Fit parameters'].keys()]

        # Simulate best fit data and plot
        resids, normalized_resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)
        
        # Save best parameters in .csv
        try:
            write_optimal_parameter_csv(minimizer_result.params, param_units, config_params['Optimal fit parameter file'])
        except Exception as e:
            print(e)

    # Simulate with input parameters, e.g. to check if parameters are reasonable before trying fit
    elif config_params['Modeling parameters']['Fit'] == False:

        print('\n### Running data simulation ###')
        # Simulate best fit data and plot
        experiment = FretExperiment(data, hybridization_params)
        kinetic_model, hybridization_model = generate_model_objects(experiment, config_params['Modeling parameters']['Kinetic model'])
        experiments.append(experiment)
        kinetic_models.append(kinetic_model)
        hybridization_models.append(hybridization_model)
        minimizer_params.append(initial_guess_params)
        resids, normalized_resids, best_kin_models, best_hybr_models = calculate_residuals_simulate_best_fit_data(experiments, minimizer_params, config_params, residuals)
        print(f'RSS for simulated data: {sum_of_squared_residuals(resids[0])}')
    
    # Do plotting
    plot_mean_flag = config_params['Plot parameters']['Plot mean data']
    best_fit_flag = config_params['Plot parameters']['Plot best fit']
    residual_flag = config_params['Plot parameters']['Plot residuals']
    RNA_populations_flag = config_params['Plot parameters']['Plot RNA population curves']
    annealed_fraction_flag = config_params['Plot parameters']['Plot annealed fraction']
    bar_2d_flag = config_params['Plot parameters']['Plot 2D population bars']
    bar_3d_flag = config_params['Plot parameters']['Plot 3D population bars']
    sample_name = config_params['Sample name']
    plot_name = config_params['Output plot file']
    plot_handler = PlotHandler(experiments, best_kin_models, best_hybr_models, resids, normalized_resids, sample_name, plot_name, plot_mean_flag, best_fit_flag, residual_flag, RNA_populations_flag, annealed_fraction_flag, bar_2d_flag, bar_3d_flag)
    plot_handler.run_plots()

    # Error analysis
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


if __name__ == '__main__':
    main()
