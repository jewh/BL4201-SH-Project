# Meant to generate Banjo settings files by reading the file names in a given directory


import os as os
from os.path import isfile, join
import shutil

instances = 10  # number of instances generated by Network Generator.py

current_directory = os.path.dirname(os.path.realpath(__file__))

files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
files.remove("Settings_Generator.py")  # so returns a list of all files in directory except itself

# ___________________________________________Above works as intended __________________________________________________

# Now begin to iteratively change Banjo settings files with the names of each file in the directory:

# Define a function that removes .txt suffices from file names

def remove_txt(path):
    file = os.path.basename(path)
    out = os.path.splitext(file)
    return out[0]

# Define a function that replaces ' ' in file names with '_'

def replace_space(file):
    out = file.replace(' ', '_')
    return out


# Specify the location of the template settings file using this variable:

template = "/Users/James/settings.txt"

# First open the banjo settings template file, make a copy, edit that copy

for i in files:
    filein = open(template, 'r')
    shutil.copy2(template,
                 "{0}/settings/settings{1}.txt".format(current_directory, remove_txt(i)))  # This works
    fileout = open("{0}/settings/settings{1}.txt".format(current_directory, remove_txt(i)), 'w')
    for line in filein:
        fileout.write(line.replace("Positive control extinction network n6 L15 N4.0 I1000 in0", remove_txt(i)).replace(
            "inputDirectory = /Users/james/",
            "inputDirectory = {0}".format(current_directory)).replace(
            "observationsFile = Extinction Network Positive Control extinction network with n6 L15 N4 I1000 in0.txt",
            "observationsFile = {0}".format(i)).replace(
            "observationsFile = Extinction Network Positive Control extinction network with n6 L15 N4 I1000 in0.txt",
            "observationsFile = {0}".format(i)).replace("outputDirectory = Users/james",
            "{0}/BanjoOutputs".format(current_directory)).replace("reportFile = static.report.txt", "reportFile = static.{0}Report.txt".format(replace_space(remove_txt(i)))))
    fileout.close()
    filein.close()

# Works! Might need to put "" around the observation file name 