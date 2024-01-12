import pandas as pd
import sys
import numpy as np


# Load CSV file
dataload = np.array(pd.read_csv(sys.argv[1], header=None))

# Save first row of data starting at column 3 as enzyme_conc
enzyme_conc = dataload[0][2:]
unique_enzyme_conc = np.unique(enzyme_conc)

# Get enzyme name from row 1 column 2
enzyme_name = dataload[0][1]

# Save second row of data starting at column 3 as cap1_conc
cap1_conc = dataload[1][2:]
unique_cap1_conc = np.unique(cap1_conc)

# Save third row of data starting at column 3 as rna_conc
rna_conc = dataload[2][2:]
unique_rna_conc = np.unique(rna_conc)

# Save fourth row of data starting at column 3 as dna_conc
dna_conc = dataload[3][2:]
unique_dna_conc = np.unique(dna_conc)

# Save first column of data as point
points = np.transpose(dataload)[0][4:]
unique_points = np.unique(points)

# Save second column of data as channel
channels = np.transpose(dataload)[1][4:]
unique_channels = np.unique(channels)

# Save data starting at row 5 and column 3 as data
data = np.transpose(dataload[4:,2:])

# for w, x, y, z in zip(enzyme_conc, cap1_conc, rna_conc, dna_conc):
#     print(w, x, y, z)
#     temp_data = data.all(enzyme_conc == w, cap1_conc == x, rna_conc == y,dna_conc == z)
#     print(temp_data)

for dna in unique_dna_conc:
    if dna == 0:
        continue
    else:
        for rna in unique_rna_conc:
            for cap1 in unique_cap1_conc:
                for enzyme in unique_enzyme_conc:
                    temp_data = data[np.all([enzyme_conc == enzyme, cap1_conc == cap1, rna_conc == rna, dna_conc == dna],axis=0 ),:]
                    if enzyme == 0:
                        blank = np.mean(temp_data, axis=0)
                        blank_IDD_start = blank[np.all([channels == 'IDD',points == 1],axis=0)]
                        blank_IDA_start = blank[np.all([channels == 'IDA',points == 1],axis=0)]
                    for point in points:
                        temp_IDD = temp_data[np.all([channels == 'IDD',points == point],axis=1),:]
                        temp_IDA = temp_data[np.all([channels == 'IDA',points == point],axis=1),:]
                        norm_IDD = temp_IDD - abs(blank_IDD_start-blank[np.all([channels == 'IDD',points == point],axis=1)])
                        norm_IDA = temp_IDA - abs(blank_IDA_start-blank[np.all([channels == 'IDA',points == point],axis=1)])

            


