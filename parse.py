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
        self.subjects= []
        self.features = dok_matrix((len(self.is_type), 0))

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

        subjects = []
        not_subjects = []
        query = RDF.Statement(None, RDF.Uri(FOAF_name), None)
        for state in mappings.find_statements(query):
            mapped_subj = state.object.__str__()
            if state.subject in self.name_list:
                subjects.append(mapped_subj)
            else:
                not_subjects.append(mapped_subj)
        if balance:
            subjects, not_subjects = self.__balance(subjects, not_subjects)

        self.is_type = [1] * len(subjects) + [0] * len(not_subjects)
        self.subjects = subjects + not_subjects

    def __balance(self, subjects, not_subjects):
        if len(subjects) < len(not_subjects):
            not_subjects = not_subjects[:len(subjects)]
        elif len(subjects) > len(not_subjects):
            subjects = subjects[:len(not_subjects)]
        return subjects, not_subjects

    def hash(self, mapping_size=10000):
        self.features = dok_matrix((len(self.is_type), mapping_size))
        index = 0
        for subj in self.subjects:
            token_arr = self.__hash_tokens(subj, mapping_size)
            for hash in token_arr:
                self.features[index, hash] = 1
            index += 1

    def __hash_tokens(self, subject, mapping_size):
        subject_tokens = subject.split()
        token_arr = [mmh3.hash(token) % mapping_size for token in subject_tokens]
        return token_arr
    
    def shuffle(self):
        permutation = np.random.permutation(self.features.shape[0])
        self.features = self.features.asformat("csr")
        self.features = self.features[permutation, :]
        self.is_type = np.asarray(np.array(self.is_type)[permutation])
        self.subjects = np.asarray(np.array(self.subjects)[permutation])

    def get_features(self):
        return self.features

    def get_subjects(self):
        return self.subjects

    def get_targets(self):
        return self.is_type

    def save(self, filename):
        pickle.dump(self.features, open(filename + ".feat", "wb"))
        pickle.dump(self.is_type, open(filename + ".type", "wb"))
        pickle.dump(self.subjects, open(filename + ".subj", "wb"))

    def load(self, filename):
        try:
                self.features = pickle.load(open(filename + ".feat", "rb"))
                self.is_type= pickle.load(open(filename + ".type", "rb"))
                self.subjects = pickle.load(open(filename + ".subj", "rb"))
        except IOError:
            print filename + "could not be found"
