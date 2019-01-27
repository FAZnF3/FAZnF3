import csv
import os

# convert txt to csv file
header = ["Name", "Cation_formula", "Num_Atom_Types", "Num_Atoms", "Atom_Types", "Bandgap_GGA_(eV)", "Dielectric_Constant_electronic", "Dielectric_constant_ionic", "Dielectric_constant_total", "Refractive_index", "Atomization_Energy(eV/atom)", "Relative_energy1(eV/atom)", "Relative energy2(eV/atom)", "Unit_cell_volume_(A^3)", "Density_(g/cm^3)", "Tool", "Pseudopotential", "Note"]
with open('nomad_data.csv', 'w', newline='') as f_output:
   csv_output = csv.writer(f_output)
   csv_output.writerow(header)

   for num in range(1, 1347):
      strn = '{:d}'.format(num).zfill(4)
      infile = open("G:/hoip_data/HOIP_cif/cif_merge/%s.txt" %strn, "r")
      with open("G:/hoip_data/HOIP_cif/cif_merge/%s.txt" %strn, 'r', newline='') as f_text:
         csv_text = csv.reader(f_text, delimiter=':', skipinitialspace=True)
         csv_output.writerow(row[1] for row in csv_text)
