
#purest fragment maker
import random
import re

species = input("Human or virus? [H/V]: ")
if species == "H": fname = "hg38.fa"
else: fname = "HIV-1.fasta"

seq = ""
with open(fname) as infile:
    for line in infile:
        if line.startswith('>'): continue
        elif line in ['\n','\r\n']: continue
        else:
            addition = re.sub("(?i)N","",line)
            seq = seq+addition

seq = re.sub('[\r\n]',"",seq)
start_indices = [ random.randint(0,(len(seq)-100)) for i in range(0,10)]
for i in range(0,len(start_indices)):
    fname = species+"_read"+str(i+1)+".txt"
    fhand = open(fname, 'w')
    fhand.write(seq[start_indices[i]:(start_indices[i]+100)])
    fhand.close()
