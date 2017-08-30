# Name Classifier
## Usage
The program consists of two modules, a RDF_processor and a model.
It also contains two demonstration scripts, `build_model.py`, which can be used to create a model from serialized hash arrays or RDF triples and `predict.py` which provides predictions for a stored model.

The `model` file contains a prebuilt model which can be used by the predict script

### `RDF_processor`

##### `RDF_processor.parse_identifiers(ident_file, object)`
Constructs and stores a set of `<subjects>` from `ident_file` with RDF:type `object` using the Redland Python bindings.
The format of the RDF file should be:
	
    <subject> RDF:type <object>.

* `ident_file`: Name of the RDF turtle file to be parsed.
* `object`: URI of the parsing `<object>`

##### `RDF_processor.map(map_file, balance=True)`
Constructs and stores an array of `<object>` strings for `FOAF:Name` predicates, and a corresponding array describing the `<object>`'s presence in the stored set of subjects.

The format of the RDF file should be:

	<subject> FOAF:Name <object>.

* `map_file`: Name of the RDF turtle file to be parsed & mapped
* `balance`: If `True`, balances the arrays by downsampling the more prevelant category.

##### `RDF_processor.hash(mapping_size=1000)`
Performs feature hashing on the stored object strings using mmh3

* `mapping_size`: The range of the hashes, between [-_mapping\_size_, +_mapping\_size_]

##### `RDF_processor.shuffle()`
Shuffles the subject, features and identity arrays

##### `RDF_processor.get_features()`
Returns the current feature array

##### `RDF_processor.get_identifiers()`
Returns the array of identifiers

##### `RDF_processor.get_subject()`
Returns the array of subject strings


### `log_reg`

##### `log_reg.log_reg(size, batch_size=1000, alpha=0.2, C=0.0)`

* `size`: Number of features in the array to be modelled
* `batch_size`: The maximum size of each batch
* `alpha`: The learning rate for batch SGD
* `C`: The L2 regularization term

##### `log_reg.fit(X, Y)`
Fits dataset `X` to target `Y` by minimizing the logistic cost function using Mini-batch Gradient Descent

* `X`: The array to be fitted. Of shape (_n\_samples_, _n\_features_)
* `Y`: The target array for `X`. Of shape (_n\_samples_)

##### `log_reg.predict(X)`
Predicts the value of `X` using the fitted model
 
* `X`: Value to be fitted

##### `log_reg.score(X, Y)`
Returns the mean successful prediction rate for `X` on the fitted model 

* `X`: The array to be predicted
* `Y`: The targets to be compared against
