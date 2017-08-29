import RDF
import mmh3
import numpy as np
import os.path
from scipy.sparse import dok_matrix
import cPickle as pickle 

RDF_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
FOAF_name = "http://xmlns.com/foaf/0.1/name" 

class RDF_processor: 
    def __init__(self):
        self.name_list = []
        self.is_type = []
        self.subj_list= []

    def parse_identifiers(self, ident_file, obj):
        if not os.path.isfile(ident_file):
           raise IOError(ident_file + " could not be found") 
        parser = RDF.TurtleParser()
        identifiers = RDF.Model(RDF.HashStorage('ident_hash', options="hash-type='memory'"))
        parser.parse_into_model(identifiers, "file:./" + ident_file)

        self.name_list = set(identifiers.get_sources(RDF.Uri(RDF_type), RDF.Uri(obj)))
        
    def map(self, map_file, balance=True):
        if not os.path.isfile(map_file):
            raise IOError(map_file + " could not be found")
        parser = RDF.TurtleParser()
        mappings = RDF.Model(RDF.HashStorage('map_hash', options="hash-type='memory'"))
        parser.parse_into_model(mappings, "file:./" + map_file)

        subj_list = []
        other_list = []
        query = RDF.Statement(None, RDF.Uri(FOAF_name), None)
        for state in mappings.find_statements(query):
            mapped_subj = state.object.__str__()
            if state.subject in self.name_list:
                subj_list.append(mapped_subj)
            else:
                other_list.append(mapped_subj)
        if balance:
            subj_list, other_list = self.__balance(subj_list, other_list)

        self.is_type = [1] * len(subj_list) + [0] * len(other_list)
        self.subj_list = subj_list + other_list

    def __balance(self, subj_list, other_list):
        if len(subj_list) < len(other_list):
            other_list = other_list[:len(subj_list)]
        elif len(subj_list) > len(other_list):
            subj_list = subj_list[:len(other_list)]
        return subj_list, other_list

    def hash(self, mapping_size=10000):
        self.features = dok_matrix((len(self.is_type), mapping_size))
        index = 0
        for subj in self.subj_list:
            token_arr = self.__hash_tokens(subj, mapping_size)
            for hash in token_arr:
                self.features[index, hash] = 1
            index += 1

    def __hash_tokens(self, subject, mapping_size):
        subject_tokens = subject.split()
        token_arr = [mmh3.hash(token) % mapping_size for token in subject_tokens]
        return token_arr
    
    def get_features(self):
        return self.features

    def get_subjects(self):
        return self.subj_list

    def get_targets(self):
        return self.is_type

    def save(self, filename):
        pickle.dump(self.features, open(filename + ".feat", "wb"))
        pickle.dump(self.is_type, open(filename + ".type", "wb"))
        pickle.dump(self.subj_list, open(filename + ".subj", "wb"))

    def load(self, filename):
        try:
                self.features = pickle.load(open(filename + ".feat", "rb"))
                self.is_type= pickle.load(open(filename + ".type", "rb"))
                self.subj_list = pickle.load(open(filename + ".subj", "rb"))
        except IOError:
            print filename + "could not be found"
