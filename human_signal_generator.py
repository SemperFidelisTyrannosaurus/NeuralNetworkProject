

import re
import random
import itertools as it

"""
human hg38.fa reference has 64185939 non-header lines
each line in 50 characters across
therefore, as long as we start our subsample two lines short of the end, we can get a subsample
"""

def generate(line_numbers):
    with open("hg38.fa") as fhand:
        for i in range(len(line_numbers)):
            start_line = line_numbers[i]
            seq= ""
            end_line = start_line + 2
            while len(seq) < 100:
                lines = it.islice(fhand, start_line, end_line) #only examines whole lines
                for line in lines:
                    if (line.startswith(">")) or (line in ["\n","\r\n"]): continue
                    else: seq = seq + re.sub("(?i)N","",line)
                end_line += 1
                if end_line >= 64185939: break
            ofile = open("human_"+str(i)+".txt","w")
            seq = re.sub("\n","",seq)
            ofile.write(seq[0:100])
            ofile.close()
    return
                    
line_nums = 64185939 - 2
start_lines = [random.randint(0,line_nums) for i in range(10)]
start_lines.sort()
print("Start lines are "+str(start_lines))
generate(start_lines)

