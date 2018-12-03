
from cnn import dna_dataset as dd
import re
import numpy as np

""" 
 FFBP approach to classification- utlizies learning of top 64 7-mers (of each) species.
 utilizes bipartite graph- two output classification nodes connected to the same set of 128 input nodes.
 input nodes are fed values by filtering operation- each node corresponding to the number of each type of 7mer in the data.
 
 The main place this method loses out is that it examines the 7mer composition and not the order.
 Where the cnn method aims to place the location of the read in the reference, this merely examines if the 7mer composition is similar
 to that of target.
"""

class Network:
    kmers = dd.load_unique_kmers(128,7)    
    def __init__(self):
        self.input_extract = [ 0 for kmer in Network.kmers]
        self.human_weight = [ 1 for i in range(128)]
        self.viral_weight = [ 1 for i in range(128)]
        self.delta_human_w = [ 0 for i in range(128)]
        self.delta_viral_w = [ 0 for i in range(128)]
        self.human_bias = 0
        self.viral_bias = 0
        self.is_human = 0
        self.is_virus = 0
        self.human_err = 0
        self.viral_err = 0
        self.human_delta = 0
        self.viral_delta = 0

    def capture_input(self,string):
        for kmer in Network.kmers:
            search_str = "(?=("+kmer+"))"
            hits = len(re.findall(search_str, string))
            self.input_extract[Network.kmers.index(kmer)] = hits
        return

    def activation_fxn(self):
        self.is_human = 1 / ( 1 + np.exp(-np.dot(self.human_weight,self.input_extract)+self.human_bias)) 
        self.is_virus = 1 / ( 1 + np.exp(-np.dot(self.viral_weight,self.input_extract)+self.viral_bias))
        return

    def forward_prop(self,string):
        self.capture_input(string)
        self.activation_fxn()

    def calculate_error(self, species):
        if species == "virus":
            viral_targ,human_targ = (1,0)
        else:
            viral_targ,human_targ=(0,1)
        self.human_err = human_targ - self.is_human
        self.viral_err = viral_targ - self.is_virus
        return

    def get_delta_value(self):
        self.human_delta = self.human_err * (1-self.is_human) * self.is_human
        self.viral_delta = self.viral_err * (1-self.is_virus) * self.is_virus
        return

    def get_delta_weight(self, eta):
        hu_step = eta * self.human_delta
        self.delta_human_w = list(np.multiply(hu_step,self.input_extract))
        vi_step = eta * self.viral_delta
        self.delta_viral_w = list(np.multiply(vi_step,self.input_extract))
        return

    def update_weights(self):
        self.human_weight = np.add(self.human_weight,self.delta_human_w)
        self.viral_weight = np.add(self.viral_weight,self.delta_viral_w)
        return

    def backpropagation(self, species, eta):
        self.calculate_error(species)
        self.get_delta_value()
        self.get_delta_weight(eta)
        self.update_weights()
        self.input_extract = [ 0 for kmer in kmers]
        self.delta_human_w = [ 0 for i in range(128)]
        self.delta_viral_w = [ 0 for i in range(128)]
        self.human_delta = 0
        self.viral_delta = 0
        self.human_err = 0
        self.viral_err = 0
        return

    def evaluate(self, dna, species):
        self.forward_prop(dna)
        self.calculate_error(species)
        self.report()
        return

    def report(self):
        print("Error in human classification: "+str(self.human_err))
        print("Error in viral classification: "+str(self.viral_err))
        return

    def train_network(self, string, species, eta=0.1):
        self.forward_prop(string)
        self.backpropagation(species,eta)
        return


    def classify(self, seq):
        self.forward_prop(seq)
        if self.is_human < 0.8 and self.is_virus < 0.8 : return "unclear"
        if self.is_human > self.is_virus: return "human"
        elif self.is_human < self.is_virus: return "virus"
        else: return "unclear"

#provisional main method
viral_train = "cnn/HIV-1_train.txt"
human_train = "cnn/hg38_train.txt"
viral_test = "cnn/HIV-1_test.txt"
human_test = "cnn/hg38_test.txt"
classifier = Network()

#train
with open(viral_train, "r") as vtrain:
    with open(human_train, "r") as htrain:
        vtrain.readline()
        htrain.readline()
        while True:
            choice = random.randint(1,2)
            if choice == 1:
                v = vtrain.readline()
                if v is None: break
                classifier.train_network(v,"virus")
            else:
                h = htrain.readline()
                if h is None: break
                classifier.train_network(htrain.readline(), "human")

#test:
with open("evaluation.csv", "w") as outfile:
    with open(viral_test, "r") as vtest:
        with open(human_test, "r") as htest:
            vtest.readline()
            htest.readline()
            while True:
                choice = random.randint(1,2)
                if choice == 1:
                    seq = vtest.readline()
                    if seq is None: break
                    truth = "virus"
                    label = classifer.classify(seq)
                    outfile.write(truth+","+label+","+seq+"\n")
                else:
                    seq = htest.readline()
                    if seq is None: break
                    truth = "human"
                    label = classifier.classify(seq)
                    outfile.write(truth+","+label+","+seq+"\n")
