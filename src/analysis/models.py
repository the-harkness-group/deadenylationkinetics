#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from icecream import ic

class DistributiveDeadenylation():

    def __init__(self, fret_experiment):

        self.time = fret_experiment.time
        self.rna = fret_experiment.rna
        self.enzyme = fret_experiment.enzyme
        self.n = fret_experiment.n
        self.species_list()

    def species_list(self):

        species = ['E*','E'] # Binding incompetent and competent enzyme
        for x in range(1,self.n+1):
            species.append(f"ETA{x}") # RNA bound enzyme
        for x in range(1,self.n+1): # Must use a second loop because species is list not dict so order matters
            species.append(f"TA{x}") # Free RNA
        species.append('A1') # Free AMP arising from deadenylation
        self.species = species

    def setup_concentrations(self):

        self.concentrations = {}
        for specie in self.species:
            self.concentrations[specie] = []

    def initial_concentration_guesses(self, enzyme, rna, k1, km1, n):
        ## Make list of t=0 concentrations of enzyme and substrate.
        ## Should be all free enzyme, all full-length and free RNA substrate
        ## because no binding or cleavage has occurred yet.

        C0 = []
        C0.append((1/(1+(k1/km1)))*enzyme) # E*, initial guess is equilibrium concentration from E* <-> E with no added RNA
        C0.append(((k1/km1)/(1+(k1/km1)))*enzyme) # E
        for x in range(1,n+1): # ETAi, initially no bound complex
            C0.append(0)
        for x in range(1,n+1):
            if x == n: # TAi
                C0.append(rna) # Last element, initially all RNA is max length
            else:
                C0.append(0) # All other lengths are initially zero conc
        C0.append(0) # A1, initially zero

        self.C0 = C0

    @staticmethod
    def relaxation_matrix(C0, k1, km1, k2, km2, kcat, n):
        ## Relaxation matrix for nuclease activity, assumes just up to 3mer polyA strand length here as an example (n=3).
        ## Assumes that enzyme falls off RNA strand after catalysis and rebinds product to catalyze again,
        ## i.e. the enzyme is fully distributive. Also assumes that enzyme cannot catalyze anything below 
        ## TA2 in length, i.e. once only 1 A is left, the enzyme cannot continue remove bases from the rest 
        ## of the strand. This matrix can be extended for arbitrary length RNA, using the parameter n to set
        ## the initial RNA length prior to cleavage by the enzyme.

        ##  d/dt  C(t)  =                       R                               * C(t)

        ##       [E*]     [-k1 k-1 0 0 0 0 0 0 0                                ] [E*]
        ##       [E]      [k1 -k-1 k-2 k-2+kcat k-2+kcat -k2[E] -k2[E] -k2[E] 0 ] [E]
        ##  d/dt [TEA1] = [0 0 -k-2 0 0 k2[E] 0 0 0                             ] [TEA1]
        ##       [TEA2]   [0 0 0 -k-2-kcat 0 0 k2[E] 0 0                        ] [TEA2]
        ##       [TEA3]   [0 0 0 0 -k-2-kcat 0 0 k2[E] 0                        ] [TEA3]
        ##       [TA1]    [0 0 k-2 kcat 0 -k2[E] 0 0 0                          ] [TA1]
        ##       [TA2]    [0 0 0 k-2 kcat 0 -k2[E] 0 0                          ] [TA2]
        ##       [TA3]    [0 0 0 0 k-2 0 0 -k2[E] 0                             ] [TA3]
        ##       [A1]     [0 0 0 kcat kcat 0 0 0 0                              ] [A1]   

        ## There are a total of 2*n + 3 species, so for the relaxation matrix R there are 
        ## 2*n + 3 rows and columns, e.g. 1 E*, 1 E, 1 A1, n ETAi, and n TAi = 2*n + 3 total species.

        R = []
        R.append([-k1, km1] + [0]*n + [0]*n + [0])  # E*
        R.append([k1, -km1] + [km2] + [km2+kcat]*(n-1) + [-k2*C0[1]]*n + [0])  # E
        R.append([0]*2 + [-km2] + [0]*(n - 1) + [k2*C0[1]] + [0]*(n - 1) + [0])  # ETA1
        for y in range(2, n + 1):  # ETA2 to ETAi
            R.append([0]*(y + 1) + [-km2-kcat] + [0]*(n - y) + [0]*(y - 1) + [k2*C0[1]] + [0]*(n - y) + [0])
        R.append([0]*2 + [km2] + [kcat] + [0]*(n - 2) + [-k2*C0[1]] + [0]*(n - 1) + [0])  # TA1
        for y in range(2, n):  # TA2 to TAn-1
            R.append([0]*(y + 1) + [km2] + [kcat] + [0]*(n - 2) + [-k2*C0[1]] + [0]*(n - y) + [0])
        R.append([0]*2 + [0]*(n - 1) + [km2] + [0]*(n - 1) + [-k2*C0[1]] + [0])  # TAn
        R.append([0]*2 + [0] + [kcat]*(n - 1) + [0]*n + [0])  # A1

        return R

    def extract_solved_concentrations(self, solver_result):

        self.concentrations['E*'].append(solver_result.y[0])
        self.concentrations['E'].append(solver_result.y[1])
        for x in range(1,self.n+1): # Don't have to use two for loops because concentrations is dict not list
            self.concentrations[f"ETA{x}"].append(solver_result.y[x+1]) # x+1 to move past E* and E, ETAi are before TAi
            self.concentrations[f"TA{x}"].append(solver_result.y[x+self.n+1]) # x+n+1 because ETAi and TAi are separated by n indices
        self.concentrations['A1'].append(solver_result.y[-1]) # A1 is last


    def calculate_total_rna_concentrations(self):

        self.total_rna_concentrations = {k:[] for k in self.concentrations.keys() if 'E' not in k}
        for i, v in enumerate(self.enzyme):
            for j in range(1, self.n+1):
                self.total_rna_concentrations[f'TA{j}'].append(self.concentrations[f'TA{j}'][i] + self.concentrations[f'ETA{j}'][i]) # TAi,T = [TAi] + [ETAi]
            self.total_rna_concentrations['A1'].append(self.concentrations['A1'][i]) # TAi,T = [TAi] + [ETAi]

    def simulate_kinetics(self, params):
        ## Run numerical integration of rate equations for a given kinetic model from t=0, returns Ci(t)
        ## Needs initial guesses for concentrations of each species at t=0

        k1 = params['k1'].value
        km1 = params['km1'].value
        k2 = params['k2'].value
        km2 = params['km2'].value
        kcat = params['kcat'].value

        self.setup_concentrations()
        for i, v in enumerate(self.enzyme):
            if self.enzyme[i] == 0: # No enzyme means nothing happens, all RNA is full length at all times
                self.concentrations[f'E*'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'E'].append([0 for x in range(len(self.time[i]))])
                self.concentrations[f'A1'].append([0 for x in range(len(self.time[i]))])

                for l in range(1, self.n+1):
                    if l == self.n:
                        self.concentrations[f'TA{self.n}'].append([self.rna for x in range(len(self.time[i]))])
                        self.concentrations[f'ETA{self.n}'].append([0 for x in range(len(self.time[i]))])
                    else:
                        self.concentrations[f'TA{l}'].append([0 for x in range(len(self.time[i]))])
                        self.concentrations[f'ETA{l}'].append([0 for x in range(len(self.time[i]))])

            else:
                self.initial_concentration_guesses(self.enzyme[i], self.rna, k1, km1, self.n)
                param_args = {'k1':k1, 'km1':km1, 'k2':k2, 'km2':km2, 'kcat':kcat, 'n':self.n}
                time_span = (self.time[i][0],self.time[i][-1])
                initial_concs = self.C0
                rate_func = self.relaxation_matrix
                t_return = np.unique(np.array(self.time[i]))
                solver_result = solve_ivp(propagator,time_span,initial_concs,t_eval=t_return,method='BDF',first_step=1e-12,atol=1e-12,args=(rate_func, param_args))
                self.extract_solved_concentrations(solver_result)

class DuplexHybridization:

    def __init__(self, fret_experiment):

        self.experimental_fret = fret_experiment.fret # Needed for solving baseline params with Ax = B
        self.fret_error = fret_experiment.fret_error
        self.dGo = fret_experiment.dGo
        self.alpha = fret_experiment.alpha
        self.n = fret_experiment.n
        self.temperature = fret_experiment.temperature
        self.time = fret_experiment.time
        self.QT = fret_experiment.QT
        self.enzyme = fret_experiment.enzyme
        self.rna = fret_experiment.rna
        self.species_list()
        self.initial_concentration_guesses()
        self.calculate_kq()

    def species_list(self):

        species = []
        for x in range(1,self.n+1):
            species.append(f'TA{x}') # Free RNA
        for x in range(1,self.n+1):
            species.append(f'TA{x}Q') # RNA hybridized to DNA quencher strand
        species.append('Q') # Free quencher strand
        self.species = species

    def setup_concentrations(self):

        self.concentrations = {}
        for specie in self.species:
            self.concentrations[specie] = [[] for concentration in self.enzyme]
        
        self.annealed_fraction  =[[] for x in self.enzyme]

    def initial_concentration_guesses(self):
        
        self.C0= []
        for x in range(self.n):
            self.C0.append(1e-7) # [TAi], free RNA
        for x in range(self.n):
            self.C0.append(1e-7) # [TAiQ], RNA annealed to DNA quencher strand
        self.C0.append(self.QT) # [Q], free DNA quencher

    def get_total_rna_concentrations(self, prior_to_hybridization_concentrations):
        # ic(prior_to_hybridization_concentrations)
        self.total_concentrations = [prior_to_hybridization_concentrations[f'TA{x}'] + prior_to_hybridization_concentrations[f'ETA{x}'] for x in range(1,self.n+1)] # TAi,T = [TAi] + [ETAi]

    def calculate_kq(self):

        i = np.array([I for I in np.arange(1,self.n+1)])
        dG = self.dGo + self.alpha*i # dG for forming hybrid RNA:DNA duplex as function of RNA length
        R = 8.3145e-3 # units of kJ/mol for dG, change to 1.987e-3 if you like kcal/mol but then also need to change dGo and alpha inputs to kcal/mol
        self.KQ = np.exp(-dG/(R*self.temperature))

    @staticmethod
    def hybrid_duplex_equations(C0, n, QT, TAiT, KQ):
        ## This takes the concentrations of each RNA species at each time point and calculates how much
        ## of each becomes annealed to the capture strand Q according to the affinity constant KQ.
        ## Essentially, this calculates the concentrations of a series of hybrid duplexes over time,
        ## once the deadenylation reaction is quenched and capture strand is added to probe the FRET value.

        eqs = []
        eqs.append(-QT + np.sum(C0[n:])) # 0 = -QT + [Q] + [TA1Q] + ... + [TAnQ], mass conservation DNA quencher
        for x in range(n):
            eqs.append(-TAiT[x] + C0[x] + C0[x+n]) # 0 = -TAiT + [TAi] + [TAiQ], mass conservation each RNA length
        for x in range(n):
            eqs.append(KQ[x]*C0[x]*C0[-1] - C0[x+n]) # KQi*[TAi]*[Q] - [TAiQ] = 0, affinity constant of each hybrid duplex

        return eqs

    def extract_solved_concentrations(self, solver_result, ei):

        for index, value in enumerate(self.concentrations.keys()):
            self.concentrations[value][ei].append(solver_result.x[index])

    def generate_baseline_matrix(self):

        self.baseline_matrix  =[[] for x in self.enzyme]
        for i, v in enumerate(self.enzyme):
            for fraction in self.annealed_fraction[i]:
                self.baseline_matrix[i].append([fraction, 1])

    def solve_fret_baseline_params(self):
    # Baseline parameters are solved according to Ax = B, where A is a matrix with rows = [Pi, 1], and Pi is the ith annealed fraction time point 1 is a factor accounting for the baseline offset.
    # B is the experimental FRET data vector for a given enzyme concentration, and x is a vector of baseline parameters for that enzyme concentration, i.e. dF and F. Each enzyme conc has a unique dF and F.
    # The result of Ax = B maps the simulated hybridized RNA populations onto the experimental data using the optimal baseline parameters in x.

        self.baseline_params  =[[] for x in self.enzyme]
        for i, v in enumerate(self.enzyme):
            baseline_solutions = np.linalg.lstsq(self.baseline_matrix[i], self.experimental_fret[i], rcond=None)  # Don't use time zero dummy point for calculating baselines
            self.baseline_params[i].append([baseline_solutions[0][0]])
            self.baseline_params[i].append([baseline_solutions[0][1]])

    def calculate_fret(self):
    ## FRET curve starts high and ends low as full length RNA is converted to
    ## shorter RNA products which have weaker affinities for the capture strand
    ## denoted by Q, thus they cannot form stable hybrid RNA:DNA duplexes that
    ## would otherwise give rise to high FRET due to the proximity of the donor
    ## and acceptor dyes.

        self.fret = [[] for x in self.enzyme]
        for i, v in enumerate(self.enzyme):
            fret_column_vector = np.matmul(self.baseline_matrix[i], self.baseline_params[i]) # Don't use time zero dummy point for calculating simulated FRET
            self.fret[i] = np.ravel(np.reshape(fret_column_vector, (1, len(fret_column_vector))))
            # self.fret[i] = np.insert(self.fret[i], 0, 0) # Put time zero dummy point back in to make simulated FRET the same length as the experimental

    def normalize_fret(self): # Normalize FRET profiles to run from 1 to 0

        self.normalized_experimental_fret = []
        self.normalized_fret = []
        self.normalized_fret_error = []
        for i, v in enumerate(self.enzyme):
            self.normalized_fret.append((self.fret[i] - self.baseline_params[i][1])/self.baseline_params[i][0])
            self.normalized_experimental_fret.append((self.experimental_fret[i] - self.baseline_params[i][1])/self.baseline_params[i][0])
            self.normalized_fret_error.append(np.sqrt((self.fret_error[i]/self.baseline_params[i][0])**2))

    def simulate_hybridization(self, kinetic_model):
    ## Solve for concentrations of free and hybridized RNA after stopping reaction and adding quencher DNA strand
    ## Needs initial guesses for concentrations as in the kinetic part

        self.setup_concentrations()
        for i, v in enumerate(kinetic_model.enzyme):
            for z,t in enumerate(kinetic_model.time[i]):
                # ic({k:kinetic_model.concentrations[k][i][z] for k in kinetic_model.concentrations.keys()})
                self.get_total_rna_concentrations({k:kinetic_model.concentrations[k][i][z] for k in kinetic_model.concentrations.keys()})
                solver_result = root(self.hybrid_duplex_equations, self.C0, args=(self.n, self.QT, self.total_concentrations, self.KQ), method='lm')
                self.extract_solved_concentrations(solver_result, i) # Need enzyme index to extend concentration list for each enzyme concentration
                self.annealed_fraction[i].append(np.sum([self.concentrations[k][i][z]/self.rna for k in self.concentrations if ('Q' in k) & (k[0] != 'Q')])) # Want everything annealed to Q, i.e. TAiQ, but not free Q
            # for j,k in enumerate(kinetic_model.concentrations.keys()):
            #     ic({k:kinetic_model.concentrations[k][i][z] for z,t in enumerate(kinetic_model.time[i])})
            #     self.get_total_rna_concentrations({k:kinetic_model.concentrations[k][i][z] for z,t in enumerate(kinetic_model.time[i])})
            #     solver_result = root(self.hybrid_duplex_equations, self.C0, args=(self.n, self.QT, self.total_concentrations, self.KQ), method='lm')
            #     self.extract_solved_concentrations(solver_result, i) # Need enzyme index to extend concentration list for each enzyme concentration
            #     self.annealed_fraction[i].append(np.sum([self.concentrations[k][i][z]/self.rna for k in self.concentrations if ('Q' in k) & (k[0] != 'Q')])) # Want everything annealed to Q, i.e. TAiQ, but not free Q

def propagator(t, C, func, constants): # Used in scipy.integrate.solve_ivp, general propagation function for use by kinetic model objects

    R = func(C, **constants) # Make relaxation matrix

    return np.matmul(R,C) # Calculates concentration fluxes, d/dt C

def generate_model_objects(fret_experiment, fit_model):

    if fit_model == 'Distributive':
        kinetic_model = DistributiveDeadenylation(fret_experiment)
    
    hybridization_model = DuplexHybridization(fret_experiment)

    return kinetic_model, hybridization_model

def simulate_full_model(params, kinetic_model, hybridization_model):

    kinetic_model.simulate_kinetics(params)
    hybridization_model.simulate_hybridization(kinetic_model)
    hybridization_model.generate_baseline_matrix()
    hybridization_model.solve_fret_baseline_params()
    hybridization_model.calculate_fret()

    return kinetic_model, hybridization_model

def calculate_residuals_simulate_best_fit_data(fret_expts, opt_params, config_params, residuals):
    # Calculate fit residuals and simulate data over finely sampled experimental temperature range to generate smooth best fit data and population plots
    
    from copy import deepcopy
    best_kin_models = []
    best_hybr_models = []
    resid = []
    normalized_resid = []
    for i, fret_expt in enumerate(fret_expts):
        kinetic_model, hybridization_model = generate_model_objects(fret_expt, config_params['Modeling parameters']['Kinetic model'])
        kinetic_model, hybridization_model = simulate_full_model(opt_params[i], kinetic_model, hybridization_model)
        hybridization_model.normalize_fret()
        resid.append(residuals(fret_expt.fret, hybridization_model.fret))
        normalized_resid.append(residuals(hybridization_model.normalized_experimental_fret, hybridization_model.normalized_fret))

        max_time = max(np.array([max(time_vector) for time_vector in fret_expt.time])) # Re-simulate with finely sampled experimental time vectors
        sim_time = [np.linspace(0, max_time, 300) for time_vector in fret_expt.time]
        sim_fret_expt = deepcopy(fret_expt)
        sim_fret_expt.time = sim_time
        sim_kinetic_model, sim_hybridization_model = generate_model_objects(sim_fret_expt, config_params['Modeling parameters']['Kinetic model'])
        sim_kinetic_model.simulate_kinetics(opt_params[i])
        sim_hybridization_model.simulate_hybridization(sim_kinetic_model)
        sim_hybridization_model.baseline_params = hybridization_model.baseline_params # Copy best baseline params for simulating best fit data
        sim_hybridization_model.generate_baseline_matrix() # Don't recalculate baseline params because this was already done above with the optimal params and copied, just make new baseline matrix
        sim_hybridization_model.calculate_fret()
        sim_hybridization_model.normalize_fret()

        best_kin_models.append(sim_kinetic_model)
        best_hybr_models.append(sim_hybridization_model)
        
    return resid, normalized_resid, best_kin_models, best_hybr_models