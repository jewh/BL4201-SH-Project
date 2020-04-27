

# File for processing data from .csv format into .txt readable by banjo.

import pandas as pd
import numpy as np

# List the desired traps from which you'd like to see data from
# Can find this as the Rothamsted trap number
# moorland traps = [45, 544, 582, 590, 592, 636, 163, 597, 139, 584, 226, 140]
traps = [114, 638, 599, 67, 186, 626, 595, 379, 493, 1, 2, 180, 662, 328, 630, 631, 629, 145, 18, 88, 381, 216, 666,
109, 212, 414, 528, 126, 653, 560, 257, 160, 664, 520, 596, 465, 278, 529, 576, 469, 480, 60, 492]

# Specify the species that you'd like to extract data for
# For user ease, using the vernacular name - but make sure the spelling and capitalisation is correct!
# heather-feeding species = ["Satyr Pug", "Heath Rustic", "True Lover's Knot", "Neglected Rustic", "Grey Scalloped Bar",
#            "Marsh Oblique-barred", "Pale Eggar", "Northern Eggar", "Fox Moth", "Emperor Moth", "Smoky Wave",
#            "Grey Mountain Carpet", "Dark Marbled Carpet", "Common Marbled Carpet", "July Highflyer", "Winter Moth",
#            "Autumnal Moth", "Twin-spot Carpet", "Narrow-Winged Pug", "Doubled-striped Pug", "Magpie Moth",
#            "Dotted Border", "Mottled Beauty", "Annulet", "Grass Wave", "Dark Tussock", "Four-dotted Footman",
#            "Clouded Buff", "Broom Moth", "Northern Deep-brown Dart", "Black Rustic", "Golden-rod Brindle",
#            "Dark Brocade", "Pinion-streaked Snout", "Autumnal Rustic", "Ingrailed Clay", "Purple Clay",
#            "Small Square-spot", "Fen Square-spot", "Beautiful Brocade", "Glaucous Shears", "Red Sword-grass",
#            "Yellow-line Quaker", "Flounced Chestnut", "Light Knot Grass", "Scarce Silver Y", "Pinion-streaked Snout",
#            "Wood Tiger"
#            ]
species = ['Anomalous', 'Black Rustic', 'Brown-Line Bright-Eye', 'Clay', 'Cloaked Minor','Clouded-Bordered Brindle',
'Coast Dart',
'Common Wainscot',
'Cosmopolitan',
'Dark Arches',
'Delicate',
'Drinker',
'Dusky Brocade',
'Dusky Sallow',
'Flounced Rustic',
'Gatekeeper',
'Grayling',
'Large Nutmeg',
'Large Skipper',
'Lesser Common Rustic',
'Lunar Yellow Underwing',
'Marbled Minor',
'Marbled White',
'Meadow Brown',
'Middle-barred Minor',
'Ringlet',
'Rosy Minor',
'Rustic Shoulder-knot',
'Shoulder-striped Wainscot',
'Slender Brindle',
'Small Dotted Buff',
'Small Heath',
'Small Skipper',
'Smoky Wainscot',
'Speckled Wood',
'Straw Dot',
'Tawny Marbled Minor',
'Wall',
'White-Speck'
]
# Now load in the data for extraction
# Just need to specify the relative path to the file
data = pd.read_csv("RISData/RISMothData.csv", encoding='unicode_escape')
print("Loaded data")
# Specify in this file which column trap numbers are stored in
trap_column = 'RIS-TrapCode'
# Now specify where the species common names are stored
species_column = 'Common Name'
# Create two output dataframes
# This is probably a terrible way of doing this but if it works once it's fine


def get_traps(trap_column, trap_list, species_column, species_list):
    out1 = pd.DataFrame()
    out2 = pd.DataFrame()
    for trap_number in trap_list:
        out1 = out1.append(data.loc[data[trap_column] == trap_number])
        for species in species_list:
            out2 = out2.append(out1.loc[out1[species_column] == species])
    return out2


out2 = get_traps(trap_column, traps, species_column, species)
# Now define a new data frame for storage
out3 = pd.DataFrame()
print("Got out3")
# Now save the data frame as a .txt file with species id in the row across the top, and observations moving down
# Now we want to get yearly totals for each species
# First set each CalDate to a pandas date type object
out2['CalDate'] = pd.to_datetime(out2.CalDate)
# Remove useless columns
headings_to_remove = ['binomial', 'County', 'TrapName', 'CountType', 'Code-RISgeneric', 'Count', 'DaysForCount']
out2.drop(headings_to_remove, axis=1, inplace=True)
# Check the data looks good
print("__________________________________________________________________________________________________")
print("                                         Input Data")
print("__________________________________________________________________________________________________")
print(out2.head())
print(out2.tail())
print("__________________________________________________________________________________________________")

# Now sum for each year, for each trap, for each species
out3 = out2.groupby([trap_column, species_column, out2.CalDate.dt.year])[['DailyCount']].sum()
out3.reset_index(inplace=True)
# Pivot so in Banjo format, and fill in NaNs with zeros
out3 = out3.pivot_table(values='DailyCount',
                        index=[trap_column, 'CalDate'],
                        columns=species_column)
out3 = out3.fillna(0)
# Inspect the data to make sure it's correct
print("__________________________________________________________________________________________________")
print("                                         Processed Data")
print("__________________________________________________________________________________________________")
print(out3.head())
print(out3.tail())
print("__________________________________________________________________________________________________")

# The output will list these species in alphabetical order as column headers,
# so the following shows which columns correspond to which species:
print("__________________________________________________________________________________________________")
print("                                       Moth Species Indices")
print("__________________________________________________________________________________________________")
for moth in list(out3):
    print("{0} = {1}".format(list(out3).index(moth), moth))
print("__________________________________________________________________________________________________")
# Finally give the number of entries in the data (observation = year's total sample at a trap)
print("                                       Data over {0} Observations ".format(len(out3.index)))
# Now, save as a .txt file for Banjo analysis
# This file is tab-delimited, and counts are recorded to 6 decimal places
np.savetxt("RISData\RISgrassdata.txt", out3, fmt='%.d', delimiter='\t')
# Also save as a .csv file so as to allow easy reference
out3.to_csv(path_or_buf="RISData\BanjoRISgrassdata.csv")


