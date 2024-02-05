import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from multiprocessing import cpu_count
from lmfit import minimize, Parameters
from plotting import make_pdf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from copy import deepcopy
from icecream import ic


class ErrorAnalysis():

    def __init__(self, opt_params, monte_carlo_iterations=None, rmsd=None, range_factor=None, points=None):

        self.opt_params = opt_params
        self.monte_carlo_iterations = monte_carlo_iterations # For Monte carlo
        self.rmsd = rmsd
        self.range_factor = range_factor # For correlation surfaces
        self.points = points

    @staticmethod
    def parameter_range(opt_param, scaling_factor=10, num_points=10):
        # Generate parameter vectors for correlation surfaces

        opt_param_range = np.logspace(np.log10(opt_param/scaling_factor), np.log10(opt_param*scaling_factor), num_points) # +/- scaling factor orders of magnitude from optimal value

        return opt_param_range
    
    def correlation_pairs(self):

        self.correlation_pairs = {} # Big dictionary of all parameter pair combinations and their associated Parameters objects for passing to fitting routine
        params_to_correlate = [k for k in self.opt_params.keys() if self.opt_params[k].vary == True]
        from copy import deepcopy
        opt_params_copy = deepcopy(self.opt_params)

        for i in range(len(params_to_correlate) - 1): # Need to correlate ith parameter with only the parameters ahead of it, don't need to do last parameter because it gets done along the way
            param_1_range = self.parameter_range(self.opt_params[params_to_correlate[i]].value, self.range_factor, self.points)

            for j in range(i + 1, len(params_to_correlate)):
                param_2_range = self.parameter_range(self.opt_params[params_to_correlate[j]].value, self.range_factor, self.points)
                self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"] = {f"{params_to_correlate[i]}":[], f"{params_to_correlate[j]}":[], "Parameter sets":[], "RSS":[], 'Fit results':[], 'Result order':[]}

                for k, param_1 in enumerate(param_1_range): # Iterate over values for each parameter pairing, set the pairs in question to constants, allow params not in correlation pair to be varied

                    for l, param_2 in enumerate(param_2_range):

                        opt_params_copy[params_to_correlate[i]].value = param_1
                        opt_params_copy[params_to_correlate[i]].vary = False
                        opt_params_copy[params_to_correlate[j]].value = param_2
                        opt_params_copy[params_to_correlate[j]].vary = False

                        self.correlation_pairs[f"{params_to_correlate[i]},{params_to_correlate[j]}"]["Parameter sets"].append(opt_params_copy) # Parallel fit results are not in the same order as this

                        opt_params_copy = deepcopy(self.opt_params)
    
    def parameter_correlation_fits(self, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper):

        maxParallelProcesses = cpu_count()
        print('')
        print('### Running parameter correlation fits using {} CPU cores. ###'.format(maxParallelProcesses))
        start = time.time()
        for param_pairs in self.correlation_pairs.keys():
            parameter_sets = self.correlation_pairs[param_pairs]['Parameter sets']

            with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
                future_results = {}
                for x in list(np.arange(len(parameter_sets))):
                    future_result = parallelExecution.submit(self.parallel_fit_task, parameter_sets[x], experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper)
                    future_results[future_result] = x
                for future in as_completed(future_results):
                    ax = future_results[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print('%r generated an exception in parameter correlation fits: %s' % (ax, exc))
                    else:
                        print(f'Parameter pair {param_pairs} iteration {ax} completed.')
                        self.correlation_pairs[param_pairs]['Result order'].append(ax)
                        self.correlation_pairs[param_pairs]['Fit results'].append(result.params)
                        self.correlation_pairs[param_pairs]['RSS'].append(result.chisqr)
                        self.correlation_pairs[param_pairs][param_pairs.split(',')[0]].append(result.params[param_pairs.split(',')[0]].value)
                        self.correlation_pairs[param_pairs][param_pairs.split(',')[1]].append(result.params[param_pairs.split(',')[1]].value)
        
        end = time.time()
        print(f"### Elapsed parameter correlation fit time was {end-start} s ###")

    def monte_carlo_fits(self, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper):

        maxParallelProcesses = cpu_count()
        print('')
        print('### Running Monte Carlo fits using {} CPU cores. ###'.format(maxParallelProcesses))
        start = time.time()
        with ProcessPoolExecutor(max_workers = maxParallelProcesses) as parallelExecution:
            future_results = {}
            for x in list(np.arange(1, self.monte_carlo_iterations + 1)):
                future_result = parallelExecution.submit(self.monte_carlo_parallel_fit_task, self.opt_params, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, self.rmsd)
                future_results[future_result] = x
            for future in as_completed(future_results):
                ax = future_results[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('%r generated an exception in Monte Carlo fits: %s' % (ax, exc))
                else:
                    print(f'Monte Carlo iteration {ax} completed.')
                    {self.monte_carlo_parameters[k].append(result.params[k].value) for k in self.monte_carlo_parameters.keys()}
        end = time.time()
        print(f"### Elapsed parameter correlation fit time was {end-start:.2f} s ###")

        for k in self.monte_carlo_parameters.keys():
            self.monte_carlo_errors[f"{k} error"] = np.std(self.monte_carlo_parameters[k])

        print('')
        print('### Monte Carlo parameter error estimates ###')
        for k1, k2 in zip(self.monte_carlo_parameters.keys(), self.monte_carlo_errors.keys()):
            print(f"{k1} = {self.opt_params[k1].value} +/- {self.monte_carlo_errors[k2]}")

    def monte_carlo_parameter_dictionary(self):
        
        self.monte_carlo_parameters = {k:[] for k in self.opt_params.keys() if self.opt_params[k].vary == True}
        self.monte_carlo_errors = {f"{k} error":None for k in self.opt_params.keys() if self.opt_params[k].vary == True}

    @staticmethod
    def parallel_fit_task(initial_guess_params, experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, min_method='leastsq', print_current_params=False):

        minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, args=(experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_params))

        return minimizer_result
    
    @staticmethod
    def monte_carlo_parallel_fit_task(initial_guess_params, perfect_experiment, kinetic_model, hybridization_model, simulate_full_model, objective_wrapper, rmsd, min_method='leastsq', print_current_params=False):

        perturbed_experiment = deepcopy(perfect_experiment)
        for i,x in enumerate(perturbed_experiment.fret):
            perturbed_experiment.fret[i] = x + np.random.RandomState().normal(scale=rmsd,size=np.size(x, 0))
        perturbed_minimizer_result = minimize(objective_wrapper, initial_guess_params, method = min_method, 
        args=(perturbed_experiment, kinetic_model, hybridization_model, simulate_full_model, print_current_params))
        return perturbed_minimizer_result

    def parameter_correlation_surfaces(self, sample_name):

        pdf = make_pdf(f"{sample_name}_parameter_correlation_surfaces.pdf")
        fractions = [x/100 for x in range(1,101)]
        contourf_colors = cm.magma(fractions)
        contourf_colors  = contourf_colors[0::10]
        contour_colors = 'black'
        levels = 8
        alpha=0.65
        linestyles='solid'

        linewidths=0.25
        for param_pairs in self.correlation_pairs.keys():

            x = np.log10(self.correlation_pairs[param_pairs][param_pairs.split(',')[0]])
            y = np.log10(self.correlation_pairs[param_pairs][param_pairs.split(',')[1]])
            z = np.log10(self.correlation_pairs[param_pairs]['RSS'])

            fig, ax = plt.subplots(1, 1)
            a = ax.tricontourf(x, y, z, alpha=alpha, colors=contourf_colors, levels=levels)
            ax.tricontour(x, y, z, linewidths=linewidths, colors=contour_colors, linestyles=linestyles, levels=levels)
            cbar = fig.colorbar(a)
            cbar.ax.set_title('log$_{10}$(RSS)', pad=15)
            x_label = 'log$_{10}$(' + param_pairs.split(',')[0] + ')'
            y_label = 'log$_{10}$(' + param_pairs.split(',')[1] + ')'
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

        pdf.close()

    def monte_carlo_distributions(self, sample_name):

        pdf = make_pdf(f"output/{sample_name}_MonteCarlo_parameter_distributions_{self.monte_carlo_iterations}_iterations.pdf")
        for k in self.monte_carlo_parameters.keys():
            fig, ax = plt.subplots(1,1)
            ax.hist(self.monte_carlo_parameters[k], bins=100)
            # ax.set_xscale('log')
            avg = np.mean(self.monte_carlo_parameters[k])
            one_sd = self.monte_carlo_errors[f"{k} error"]
            #two_sd = 2*one_sd
            #lines = [avg, avg+one_sd, avg-one_sd, avg+two_sd, avg-two_sd]
            #ymin = [0, 0, 0, 0, 0]
            #ymax = np.repeat(self.monte_carlo_iterations/4, 5)
            #ax.vlines(lines, ymin, ymax, linestyles='--', color='r')
            ax.set_title(f"{k}: {avg} $\pm$ {one_sd} (1 std.)")
            ax.set_xlabel(k)
            ax.set_ylabel('Count')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()

    @staticmethod
    def make_grid_data(x, y, z, resolution=100, contour_method='linear'):

        x_resample = np.linspace(min(x), max(x), resolution)
        y_resample = np.linspace(min(y), max(y), resolution)

        x_grid, y_grid = np.meshgrid(x_resample, y_resample)
        z_grid = griddata((x, y), z, (x_grid, y_grid), contour_method)

        return x_grid, y_grid, z_grid
    
    def save_parameter_correlation_results(self, sample_name):

        result_dfs = []
        for param_pairs in self.correlation_pairs.keys():
            result_dict = {}
            result_dict[param_pairs.split(',')[0]] = self.correlation_pairs[param_pairs][param_pairs.split(',')[0]]
            result_dict[param_pairs.split(',')[1]] = self.correlation_pairs[param_pairs][param_pairs.split(',')[1]]
            result_dict['RSS'] = self.correlation_pairs[param_pairs]['RSS']
            result_dict['Result order'] = self.correlation_pairs[param_pairs]['Result order']
            result_dfs.append(pd.DataFrame(result_dict))

        merged_result_df = pd.concat(result_dfs, axis=1, keys=(self.correlation_pairs.keys()))
        merged_result_df.to_csv(f"output/{sample_name}_parameter_correlation_results.csv")

    def save_monte_carlo_results(self, sample_name):

        monte_carlo_results = {'Parameter':[], 'Opt Value':[], 'Error':[]}
        monte_carlo_df = pd.DataFrame(self.monte_carlo_parameters)
        for k1, k2 in zip(self.monte_carlo_parameters, self.monte_carlo_errors):
            monte_carlo_results['Parameter'].append(k1)
            monte_carlo_results['Opt Value'].append(self.opt_params[k1].value)
            monte_carlo_results['Error'].append(self.monte_carlo_errors[k2])
        monte_carlo_results = pd.DataFrame(monte_carlo_results)
        monte_carlo_df.to_csv(f"output/{sample_name}_MonteCarlo_values_{self.monte_carlo_iterations}_iterations.csv", index=False)
        monte_carlo_results.to_csv(f"output/{sample_name}_MonteCarlo_errors_{self.monte_carlo_iterations}_iterations.csv", index=False)
        
