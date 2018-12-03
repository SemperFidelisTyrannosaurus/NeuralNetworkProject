
from cnn import dna_dataset as dd
import re
import numpy as np

class Network:
    kmers = dd.load_unique_kmers(128,7)    
    def __init__(self):
        self.input_extract = [ 0 for kmer in kmers]
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
        for kmer in kmers:
            search_str = "(?=("+kmer+"))"
            hits = len(re.findall(search_str, string))
            self.input_extract[kmers.index(kmer)] = hits
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
        self.det_delta_weight(eta)
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
        self.backpropagation(self,species,eta)
        return

