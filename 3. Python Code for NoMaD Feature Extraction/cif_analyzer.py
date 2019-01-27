strn = ""

for num in range(1, 1347):
    strn = '{:d}'.format(num).zfill(4)
    infile = open("G:/hoip_data/HOIP_cif/cif_merge/%s.cif" %strn, "r")
    outfile = open("G:/hoip_data/HOIP_cif/all_properties_txt_merge/%s.txt" %strn, "w")
    for line in infile:
        outfile.write(line)