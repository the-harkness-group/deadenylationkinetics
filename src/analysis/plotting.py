import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.backends.backend_pdf
import matplotlib.patheffects as path_effects
import matplotlib.colors as colors
import numpy as np
from scipy import interpolate


class PlotHandler:

    def __init__(self, experiments, kinetic_models, hybridization_models, resids, normalized_resids, sample_name, plot_name, plot_mean_flag, best_fit_flag, residual_flag, RNA_populations_flag, annealed_fraction_flag, bar_2d_flag, bar_3d_flag):

        self.experiments = experiments
        self.kinetic_models = kinetic_models
        self.hybridization_models = hybridization_models
        self.residuals = resids
        self.normalized_residuals = normalized_resids
        self.sample_name = sample_name
        self.timesample = [0,30,60,120,300,600,900,1200,1500,1800,2100,2400,3600] # Time points at which to make bar plots of the RNA populations

        self.plot_mean_flag = plot_mean_flag
        self.best_fit_flag = best_fit_flag
        self.residual_flag = residual_flag
        self.RNA_populations_flag = RNA_populations_flag
        self.annealed_fraction_flag = annealed_fraction_flag
        self.bar_2d_flag = bar_2d_flag
        self.bar_3d_flag = bar_3d_flag

        unique_enzyme = self.experiments[0].enzyme # FRET plots
        slice = 1
        points = len(unique_enzyme)
        colormap = cm.inferno
        map_name = 'enzyme_colors'
        reversed = True
        self.get_colors(points, slice, colormap, map_name, reversed)
        self.enzyme_colors = self.enzyme_colors
        self.alphas = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3]

        t_pop_keys = [k for k in kinetic_models[0].concentrations.keys() if k not in ['E', 'E*']] # 2D and 3D bar plots
        slice = 1
        points = len(t_pop_keys)
        colormap = cm.coolwarm
        map_name = 't_pop_colors'
        reversed = False
        self.get_colors(points, slice, colormap, map_name, reversed)

        self.enzyme_bar_colors = ['#80cdc1', '#c7eae5'] # 2D bar plots

        self.pdf = make_pdf(f"output/{plot_name}")

    def addattr(self, x, v):
        self.__dict__[x] = v

    @staticmethod
    def plot_best_fit(experiments, kinetic_models, hybridization_models, enz_colors, sample_name, pdf, plot_mean_flag):

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            
            kinetic_model = kinetic_models[j]
            hybridization_model = hybridization_models[j]            
            if plot_mean_flag == True:
                for r, rna in enumerate(experiment.rna):
                    for e,enzyme in enumerate(experiment.enzyme):
                        experiment.unique_time.append([])
                        experiment.mean_fret.append([])
                        experiment.fret_std.append([])
                        experiment.mean_norm_fret.append([])
                        experiment.norm_fret_std.append([])                    
                        for t,time in enumerate(np.unique(experiment.time[e])):
                            filtered_fret = experiment.fret[e][np.where(experiment.time[e] == time)[0]]
                            mean_fret = filtered_fret.mean()
                            stdev_fret = filtered_fret.std()
                            stdev_fret = stdev_fret if not np.isnan(stdev_fret) else 0

                            filtered_norm_fret = hybridization_model.normalized_experimental_fret[e][np.where(experiment.time[e] == time)[0]]
                            mean_norm_fret = filtered_norm_fret.mean()
                            stdev_norm_fret = filtered_norm_fret.std()
                            stdev_norm_fret = stdev_norm_fret if not np.isnan(stdev_norm_fret) else 0

                            experiment.unique_time[e].append(time)
                            experiment.mean_fret[e].append(mean_fret)
                            experiment.mean_norm_fret[e].append(mean_norm_fret)
                            experiment.fret_std[e].append(stdev_fret)
                            experiment.norm_fret_std[e].append(stdev_norm_fret)
            else:
                continue

            data_fit_fig = plt.subplots(1,1,figsize=(7,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
            normalized_data_fit_fig = plt.subplots(1,1,figsize=(7,5))

            max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
            xlim = [0-0.03*max_time, max_time+0.03*max_time]
            xticks = np.linspace(0,max_time,5)

            for i, enzyme in enumerate(experiment.enzyme):
                color_idx = i
                rep_idx = j
                line_label = f"{enzyme*1e6}"

                # FRET data plots
                data_fit_fig[1].plot(hybridization_model.time[i],hybridization_model.fret[i],color=enz_colors[color_idx],label=line_label)
                normalized_data_fit_fig[1].plot(hybridization_model.time[i],hybridization_model.normalized_fret[i],color=enz_colors[color_idx],label=line_label)
                if plot_mean_flag == True:
                    data_fit_fig[1].errorbar(experiment.unique_time[i],experiment.mean_fret[i],yerr=experiment.fret_std[i],fmt='o',markersize=5,mfc='w',
                                             mec=enz_colors[color_idx],mew=2,capsize=3,capthick=1.5,ecolor=enz_colors[color_idx])
                    normalized_data_fit_fig[1].errorbar(experiment.unique_time[i],experiment.mean_norm_fret[i],yerr=experiment.norm_fret_std[i],fmt='o',markersize=5,mfc='w',
                                                        mec=enz_colors[color_idx],mew=2,capsize=3,capthick=1.5,ecolor=enz_colors[color_idx])
                else:
                    data_fit_fig[1].scatter(experiment.time[i],experiment.fret[i],s=30,color=enz_colors[color_idx],alpha=0.8,linewidth=0)
                    normalized_data_fit_fig[1].scatter(experiment.time[i],hybridization_model.normalized_experimental_fret[i],s=30,color=enz_colors[color_idx],alpha=0.8,linewidth=0)                
            
            for plot in [data_fit_fig, normalized_data_fit_fig]:
                plot[1].set_xlim(xlim)
                plot[1].set_xticks(xticks)
                plot[1].set_xticklabels(xticks,rotation=45)
                plot[1].set_xlabel('Time (s)')
                if plot == data_fit_fig:
                    plot[1].set_ylabel('FRET')
                elif plot == normalized_data_fit_fig:
                    plot[1].set_ylabel('Normalized FRET')
                    plot[1].set_ylim([-0.25, 1.25])
                if len(experiments) > 1:
                    plot[1].set_title(f"Replicate {j+1}")
                else:
                    plot[1].set_title(f"Average of replicates")

                L = plot[1].legend(loc='upper right',title=f"[{sample_name}] $\mu$M",frameon=False,handlelength=0,handletextpad=0,markerscale=0)
                for k,text in enumerate(L.get_texts()):
                    text.set_color('white')
                    text.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground=enz_colors[k]),path_effects.Normal()])
                for item in L.legendHandles:
                    item.set_visible(False)            
            
                plot[0].tight_layout()
                pdf.savefig(plot[0])
        plt.close()

    @staticmethod
    def plot_residuals(experiments, kinetic_models, hybridization_models, normalized_residuals, enz_colors, pdf, plot_mean_flag):

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            kinetic_model = kinetic_models[j]
            hybridization_model = hybridization_models[j]
            normalized_residual = normalized_residuals[j]

            if plot_mean_flag == True:
                for r, rna in enumerate(experiment.rna):
                    for e,enzyme in enumerate(experiment.enzyme):
                        experiment.unique_time.append([])
                        experiment.mean_resid.append([])
                        experiment.resid_std.append([])                
                        for t,time in enumerate(np.unique(experiment.time[e])):
                            filtered_resid = normalized_residual[e][np.where(experiment.time[e] == time)[0]]
                            mean_resid = filtered_resid.mean()
                            stdev_resid = filtered_resid.std()
                            stdev_resid = stdev_resid if not np.isnan(stdev_resid) else 0

                            experiment.mean_resid[e].append(mean_resid)
                            experiment.resid_std[e].append(stdev_resid)
            else:
                continue

            resid_fig = plt.subplots(len(experiment.enzyme),1,figsize=(7,len(experiment.enzyme))) # One subplot for each set of residuals to see more clearly

            max_time = round(np.max([np.max(v) for v in experiment.time]),-2)
            xlim = [0-0.03*max_time, max_time+0.03*max_time]
            xticks = np.linspace(0,max_time,5)

            for subfig in resid_fig[1]:
                subfig.axhline(y=0, color='k',lw=1,ls='--',zorder=0) # Data fit residual plots

            for i, enzyme in enumerate(experiment.enzyme):
                color_idx = i
                line_label = f"{enzyme*1e6} $\mu$M"

                if plot_mean_flag == True:
                    resid_fig[1][i].errorbar(experiment.unique_time[i],experiment.mean_resid[i],yerr=experiment.resid_std[i],fmt='o',markersize=5,mfc='w',mec=enz_colors[color_idx],mew=2,capsize=3,capthick=1.5,ecolor=enz_colors[color_idx],label=line_label)
                else:
                    resid_fig[1][i].scatter(experiment.time[i],normalized_residual[i],color=enz_colors[color_idx],alpha=0.8,linewidth=0,label=line_label)
                
                resid_fig[1][i].set_ylim([-0.28, 0.28])
                resid_fig[1][i].set_xlim(xlim)
                resid_fig[1][i].set_xticks(xticks)

                # get handles
                handles, labels = resid_fig[1][i].get_legend_handles_labels()
                # remove the errorbars
                handles = [h[0] for h in handles]
                # use them in the legend
                L = resid_fig[1][i].legend(handles, labels, loc='upper right',frameon=False,handlelength=0,handletextpad=0,markerscale=0)
                for k,text in enumerate(L.get_texts()):
                    text.set_color('white')
                    text.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground=enz_colors[i]),path_effects.Normal()])

                if i < len(experiment.enzyme) - 1:
                    resid_fig[1][i].set_xticklabels([])
                else:
                    resid_fig[1][i].set_xticklabels(xticks, rotation=45)

            if len(experiments) > 1:
                resid_fig[1][0].set_title(f"Replicate {j+1}")
            else:
                resid_fig[1][0].set_title(f"Average of replicates")

            resid_fig[1][int(round(len(experiment.enzyme)/2))].set_ylabel('Residual')
            resid_fig[1][-1].set_xlabel('Time (s)')
            resid_fig[0].tight_layout()
            resid_fig[0].subplots_adjust(hspace=0.45)
            pdf.savefig(resid_fig[0])
        plt.close()

    @staticmethod
    def plot_all_populations(experiments, kinetic_models, sample_name, pdf):
            
        # !! still needs colorbar legend
    
        for j, experiment in enumerate(experiments): # Only make plot for one replicate
            if j == 0:
                kinetic_model = kinetic_models[j]
                kinetic_model.calculate_total_rna_concentrations()
                color_values = cm.coolwarm(np.linspace(0,1,len(kinetic_model.total_rna_concentrations.keys())))

                all_populations_fig, axs = plt.subplots(len(experiment.enzyme),2,figsize=(9,len(experiment.enzyme)*2), gridspec_kw={'width_ratios': [3,1]}) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
                
                for i, enzyme in enumerate(experiment.enzyme):

                    # all populations plot
                    for q, k in enumerate(kinetic_model.total_rna_concentrations.keys()):
                        if k != 'A1': 
                            axs[i][0].plot(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i], label=k, color=color_values[q], alpha=0.8)                
                        else: # Plot A1 on separate axis since it gets very large
                            axs[i][1].plot(kinetic_model.time[i], kinetic_model.total_rna_concentrations[k][i], label=k, color=color_values[0], alpha=0.8)
                    
                    if i == len(experiment.enzyme) - 1:  # Put x-axis label below last plot
                        axs[i][0].set_xlabel('Time (s)')
                        axs[i][1].set_xlabel('Time (s)')
                    else: # No x-tick labels for plots that aren't on bottom row
                        axs[i][0].set_xticklabels([])
                        axs[i][1].set_xticklabels([])
                        
                    axs[i][0].set_ylabel('Concentration (M)')
                    axs[i][0].set_title(f"RNA species for {enzyme*1e6} $\mu$M {sample_name}")
                    axs[i][1].set_title(f"[A1] (M)")
                all_populations_fig.tight_layout()
                pdf.savefig(all_populations_fig)
            plt.close()

    @staticmethod
    def plot_annealed_fraction(experiments, hybridization_models, enz_colors, sample_name, pdf):

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            if j == 0:
                hybridization_model = hybridization_models[j]
                annealed_fraction_fig = plt.subplots(1,1,figsize=(7,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]

                for i, enzyme in enumerate(experiment.enzyme):
                    color_idx = i
                    line_label = f"{enzyme*1e6}"

                    # Annealed Fraction plot
                    annealed_fraction_fig[1].scatter(hybridization_model.time[i],hybridization_model.annealed_fraction[i],s=30,color=enz_colors[color_idx],label=line_label,alpha=0.8,linewidth=0)
                
                annealed_fraction_fig[1].set_xlabel('Time (s)')
                annealed_fraction_fig[1].set_ylabel('Annealed fraction')
                annealed_fraction_fig[1].set_title(f"Annealed fraction")
                annealed_fraction_fig[1].set_ylim([-0.1, 1.1])
                L = annealed_fraction_fig[1].legend(frameon=False,handlelength=0,handletextpad=0,loc='upper right',title=f"[{sample_name}] $\mu$M",markerscale=0)
                for k,text in enumerate(L.get_texts()):
                    text.set_color(enz_colors[k])
                    text.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground=enz_colors[k]),path_effects.Normal()])
                annealed_fraction_fig[0].tight_layout()
                pdf.savefig(annealed_fraction_fig[0])
            plt.close()

    @staticmethod
    def plot_2d_population_bars(experiments, kinetic_models, hybridization_models, enzyme_colors, t_pop_colors, pdf, timesample):

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            if j == 0:
                kinetic_model = kinetic_models[j]
                for i, enzyme in enumerate(experiment.enzyme):
                    # Species concentration bar plots at desired time points
                    fig, ax = plt.subplots(len(timesample), 3, figsize=(11, len(timesample)*2), gridspec_kw={'width_ratios': [1, 1, 6]})

                    for ti, time in enumerate(timesample):
                        tindex = (np.abs(kinetic_model.time[i] - time)).argmin()
                        kinetic_model.calculate_total_rna_concentrations()
                        ax[ti][0].bar(0.0, kinetic_model.concentrations['E*'][i][tindex]/kinetic_model.enzyme[i], color=enzyme_colors[1], label='E*', width=0.5)
                        ax[ti][0].bar(0.75, kinetic_model.concentrations['E'][i][tindex]/kinetic_model.enzyme[i], color=enzyme_colors[0], label='E', width=0.5)
                        ax[ti][1].bar(0.0, kinetic_model.total_rna_concentrations[f'TA{kinetic_model.n}'][i][tindex]/kinetic_model.rna, color=t_pop_colors[-1], label=f"TA$_{{{kinetic_model.n}}}$", width=0.5)
                        ax[ti][1].bar(0.75, kinetic_model.total_rna_concentrations['A1'][i][tindex]/(kinetic_model.rna * (kinetic_model.n - 1)), color=t_pop_colors[0], label="A$_{{{1}}}$", width=0.5)

                        for q, k in enumerate(kinetic_model.total_rna_concentrations.keys()):
                            if k not in ['A1', f"TA{kinetic_model.n}"]:
                                alen = int(k.split('TA')[1])
                                ax[ti][2].bar(q, kinetic_model.total_rna_concentrations[k][i][tindex]/kinetic_model.rna, color=t_pop_colors[q], label=f'TA$_{{{alen}}}$')
                        
                        ax[ti][2].invert_xaxis()
                        ax[ti][0].set_ylabel(f"Fraction at t: {np.round(kinetic_model.time[i][tindex],0)} s")
                        if ti == len(timesample) - 1:
                            ax[ti][0].set_xlabel('Enzyme states')
                            ax[ti][1].set_xlabel('Initial RNA and AMP')
                            ax[ti][2].set_xlabel('Product RNA')
                        
                        ax[ti][0].set_xticks([0, 0.75])
                        ax[ti][1].set_xticks([0, 0.75])                        
                        ax[ti][2].set_xticks([x for x in range(0, kinetic_model.n - 1)])
                        rna_lens = [int(k.split('TA')[1]) for k in kinetic_model.total_rna_concentrations.keys() if k not in [f'TA{kinetic_model.n}', 'A1']]
                        if ti < len(timesample) - 1:
                            ax[ti][0].set_xticklabels([])
                            ax[ti][1].set_xticklabels([])
                            ax[ti][2].set_xticklabels([])
                        else:
                            ax[ti][0].set_xticklabels(['E*', 'E'], rotation=45)
                            ax[ti][1].set_xticklabels([f"TA$_{{{kinetic_model.n}}}$", 'A$_{{{1}}}$'], rotation=45)
                            ax[ti][2].set_xticklabels(['TA' + f"$_{{{v}}}$" for v in rna_lens], rotation=45)
                       
                        ax[ti][0].set_xlim([-0.5, 1.25])
                        ax[ti][1].set_xlim([-0.5, 1.25])
                        ax[ti][2].set_xlim([kinetic_model.n-0.75, -0.75])
                        ax[ti][0].set_ylim([-0.02, 1.02])
                        ax[ti][1].set_ylim([-0.02, 1.02])
                        ax[ti][2].set_ylim([-0.02, 1.02])
                        if ti == 0:
                            ax[ti][2].set_title(f"$E_{0}$: {np.round(kinetic_model.enzyme[i]*1e6,1)}, $RNA_{0}$: {np.round(kinetic_model.rna*1e6,1)} $\mu$M")
                    fig.tight_layout()
                    pdf.savefig(fig)
        plt.close()

    @staticmethod
    def plot_3d_population_bars(experiments, kinetic_models, hybridization_models, pdf, timesample, sample_name):

        for j, experiment in enumerate(experiments): # Plot individual replicates on separate plots to see fits more clearly
            kinetic_model = kinetic_models[j]
            kinetic_model.calculate_total_rna_concentrations()
            hybridization_model = hybridization_models[j]

            # 3D surface/bar plots, e.g. x = species, y = enzyme conc., z = species fraction
            surf_figs = []
            for ti, time in enumerate(timesample):
                species_matrix = []
                resampled_species_matrix = []
                enzyme_matrix = []
                resampled_enzyme_matrix = []
                fraction_matrix = []
                rna_species = [f'TA{x}' for x in range(1, experiment.n + 1)]
                fine_enz = np.linspace(kinetic_model.enzyme[1]*1e6, kinetic_model.enzyme[-1]*1e6, 10)
                resampled_species_vector = np.linspace(1, kinetic_model.n, kinetic_model.n)
                for ei, enz in enumerate(kinetic_model.enzyme):
                    if ei == 0:
                        pass
                    else:
                        tindex = (np.abs(kinetic_model.time[ei] - time)).argmin()
                        species_vector = [int(x.split('TA')[1]) for x in rna_species]
                        enzyme_vector = [enz*1e6 for x in species_vector]
                        fraction_vector = [kinetic_model.total_rna_concentrations[x][ei][tindex]/kinetic_model.rna for x in rna_species]

                        species_matrix.append(species_vector)
                        enzyme_matrix.append(enzyme_vector)
                        fraction_matrix.append(fraction_vector)
                
                for ei, enz in enumerate(fine_enz):
                    resampled_enzyme_vector = [enz for x in resampled_species_vector]
                    resampled_species_matrix.append(resampled_species_vector)
                    resampled_enzyme_matrix.append(resampled_enzyme_vector)

                surf_fig = plt.figure(figsize=(11, 7))
                surf_ax = surf_fig.gca(projection='3d')
                X = np.array(resampled_species_matrix)
                Y = np.array(resampled_enzyme_matrix)
                Z = interpolate.griddata((np.ravel(species_matrix), np.ravel(enzyme_matrix)), np.ravel(fraction_matrix), (X, Y), method='linear')

                X = np.ravel(X)
                Y = np.ravel(Y)
                Z = np.ravel(Z)

                x = np.full_like(X, 0.8)
                y = np.full_like(Y, 0.4)
                z = np.full_like(Z, 0)

                fracs = np.ravel(X.astype(float))/np.ravel(X.max())
                norm = colors.Normalize(fracs.min(), fracs.max())
                color_values = cm.coolwarm(norm(fracs.tolist()))

                surf_ax.bar3d(Y, X, z, y, x, Z, color=color_values, edgecolor='w', linewidth=0.1, shade=False)
                surf_ax.set_ylabel('RNA polyA length', labelpad=10)
                surf_ax.set_xlabel(f'[{sample_name}] $\mu$M', labelpad=10)
                surf_ax.set_zlabel('Fraction', labelpad=10)
                surf_ax.set_yticks(np.linspace(2, kinetic_model.n, 9))
                surf_ax.set_xticks(np.linspace(Y.min(), Y.max(), 10))
                surf_ax.set_ylim([X.max()+0.05*X.max(), X.min()-0.05*X.max()])
                surf_ax.set_xlim([Y.min()-0.05*Y.max(), Y.max()+0.05*Y.max()])
                surf_ax.set_zlim([-0.01, 1.01])
                surf_ax.set_title(f"Time: {time} s")

                surf_ax.xaxis.pane.fill = False
                surf_ax.yaxis.pane.fill = False
                surf_ax.zaxis.pane.fill = False
                surf_ax.xaxis.pane.set_edgecolor('w')
                surf_ax.yaxis.pane.set_edgecolor('w')
                surf_ax.zaxis.pane.set_edgecolor('w')

                surf_ax.set_box_aspect(aspect=(1.8,1.8,1))
                surf_ax.view_init(azim=-60, elev=30)
                surf_fig.tight_layout()
                surf_figs.append(surf_fig)
                plt.close(surf_fig)

            for fig in surf_figs:
                pdf.savefig(fig)
        plt.close()

    def run_plots(self):

        if self.best_fit_flag == True:
            self.plot_best_fit(self.experiments, self.kinetic_models, self.hybridization_models, self.enzyme_colors, self.sample_name, self.pdf, self.plot_mean_flag)

        if self.residual_flag == True:
            self.plot_residuals(self.experiments, self.kinetic_models, self.hybridization_models, self.normalized_residuals, self.enzyme_colors, self.pdf, self.plot_mean_flag)
                
        if self.RNA_populations_flag == True:
            self.plot_all_populations(self.experiments, self.kinetic_models, self.sample_name, self.pdf)
                
        if self.annealed_fraction_flag == True:
            self.plot_annealed_fraction(self.experiments, self.hybridization_models, self.enzyme_colors, self.sample_name, self.pdf)
                
        if self.bar_2d_flag == True:
            self.plot_2d_population_bars(self.experiments, self.kinetic_models, self.hybridization_models, self.enzyme_bar_colors, self.t_pop_colors, self.pdf, self.timesample)
                
        if self.bar_3d_flag == True:
            self.plot_3d_population_bars(self.experiments, self.kinetic_models, self.hybridization_models, self.pdf, self.timesample, self.sample_name)                
        self.pdf.close()

    def get_colors(self, points=100, slice=1, colormap=cm.coolwarm, map_name='plot_colors', reversed=False):

        color_values = colormap(np.linspace(0, 1, points+2))
        color_values = color_values[1:-1]

        if reversed == True:
            color_values = color_values[::-1]

        self.addattr(map_name, color_values)


def make_pdf(pdf_name):
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    return pdf