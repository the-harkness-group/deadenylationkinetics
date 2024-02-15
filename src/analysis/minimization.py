import numpy as np


def objective_wrapper(params, experiment, kinetic_model, hybridization_model, simulate_full_model):
    kinetic_model, hybridization_model = simulate_full_model(params, kinetic_model, hybridization_model)
    
    resid = residuals(experiment.fret, hybridization_model.fret)
    concat_resid = np.concatenate(resid, axis=None)
    return concat_resid

def residuals(ydata, predicted):
    resid = []
    for i, v in enumerate(ydata):
        resid.append((ydata[i] - predicted[i])) 
    return resid

def sum_of_squared_residuals(residuals):
    resid = np.concatenate(residuals, axis=None)
    rss = np.sum(np.square(resid))
    return rss
