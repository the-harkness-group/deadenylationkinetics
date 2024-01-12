import pandas as pd
import icecream as ic
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys

# Specify the file path
file_path = sys.argv[1]

file_name = file_path.replace(".csv", "")
protein = sys.argv[2]

# Load the CSV file into a dataframe
df = pd.read_csv(file_path)

rnas = df["RNA"].unique()
enzymes = sorted(df["Enzyme"].unique())
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
                temp_df = pd.DataFrame({"Time": time, "mFRET": mean_fret, "Stdev": stdev_fret, "RNA": rna, "Enzyme": enzyme}, index=[0])
                mean_df = pd.concat([mean_df, temp_df])



points = len(enzymes)
colormap = cm.inferno(np.linspace(1, 0, points+1))
colormap = colormap[1:]

max_time = round(np.max([np.max(v) for v in mean_df.Time]),-3)
xlim = [0-0.03*max_time, max_time+0.03*max_time]
xticks = np.linspace(0,max_time,6)
ylim = [-0.05,1.05]
yticks = np.linspace(0,1,6)

len_rna = len(rnas)


if len_rna == 1:
    data_fig, ax = plt.subplots(1,1,figsize=(2+len_rna*5,5)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
    for j, enzyme in enumerate(enzymes):
            filtered_df = mean_df[(mean_df["RNA"] == rna) & (mean_df["Enzyme"] == enzyme)]
            color_idx = j
            line_label = f"{enzyme*1e6:.1f}"

            # FRET data plots
            # axs[i].errorbar(,,color=colormap[color_idx])
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
            # plt.setp(legend.get_title(), fontsize=12)
else:
    data_fig, axs = plt.subplots(len_rna,1,figsize=(7,len_rna*4)) # Access fig with data_fit_fig[0], axis with data_fit_fig[1]
    for i, rna in enumerate(rnas):
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
                # plt.setp(legend.get_title(), fontsize=12)
data_fig.tight_layout()
data_fig.savefig(f"{file_name}_plotted.pdf", format="pdf")
data_fig.savefig(f"{file_name}_plotted.png", format="png", dpi=600)
data_fig.savefig(f"{file_name}_plotted.svg", format="svg")
