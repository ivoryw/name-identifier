from math import ceil
from sklearn.utils import shuffle
import parse
import model
import cPickle as pickle

FOAF_Person = "http://xmlns.com/foaf/0.1/Person"

c = parse.RDF_processor()
#c.parse_identities("instance_types_transitive_en.ttl", FOAF_Person)
#c.map("mappingbased_literals_en.ttl")
#c.hash()
c.load("sample")
size = 100000
c.hash(mapping_size=size)

X = c.get_hashes()
Y = c.get_targets()

X, Y = shuffle(X, Y)

n_train= 4 * int(ceil(X.shape[0]/5))
X_train = X[:n_train, :]
X_test = X[n_train:, :]
Y_train = Y[:n_train]
Y_test = Y[n_train:]

log_r = model.log_reg(size)
for i in range(0, 100):
    log_r.fit(X_train, Y_train)
pickle.dump(log_r, open("model", "wr"))
print log_r.score(X_test, Y_test)
