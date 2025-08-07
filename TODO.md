
---

## Core Interfaces

- [ ] Base classes
  - [ ] `BaseEstimator`: fit(), get_params(), set_params()
  - [ ] `ClassifierMixin`, `RegressorMixin`
  - [ ] `TransformerMixin`: fit_transform()
  - [ ] Input validation module (`core/validator.py`)

---

## Models

### Supervised Models (`prediction/`)
- [x] Linear models
  - [x] Linear Regression
  - [x] Logistic Regression
- [ ] K-Nearest Neighbors
  - [x] KNeighborsClassifier
  - [ ] KNeighborsRegressor
- [ ] Decision Trees
  - [x] Classifier
  - [x] Regressor
  - [ ] Random Forest
- [x] Support Vector Machines 

### Unsupervised Models (`clustering/`)
- [ ] KMeans
- [ ] DBSCAN
- [ ] Hierarchical clustering (agglomerative)
- [ ] Gaussian Mixture Models (GMM)

### Neural Models (Optional) (`neural/`)
- [ ] Base neural architecture
- [ ] Autoencoder
- [ ] Activation functions (ReLU, Sigmoid, Tanh)
- [ ] Loss functions and backpropagation

---

## Preprocessing Transformers (`preprocessing/`)

- [ ] Scaling
  - [ ] StandardScaler
  - [ ] MinMaxScaler
- [ ] Encoding
  - [ ] LabelEncoder
  - [ ] OneHotEncoder
- [ ] Imputation
  - [ ] SimpleImputer (mean, median, most_frequent)
- [ ] Feature Engineering
  - [ ] PolynomialFeatures
  - [ ] Binning (e.g., KBinsDiscretizer)

---

## Evaluation Metrics (`metrics/`)

- [ ] Classification metrics
  - [ ] accuracy_score
  - [ ] precision_score
  - [ ] recall_score
  - [ ] f1_score
- [ ] Regression metrics
  - [ ] mean_squared_error
  - [ ] mean_absolute_error
  - [ ] r2_score
- [ ] Clustering metrics
  - [ ] silhouette_score
  - [ ] adjusted_rand_index

---

## Model Selection (`model_selection/`)

- [ ] Data splitting
  - [ ] train_test_split
  - [ ] KFold
  - [ ] StratifiedKFold
- [ ] Validation
  - [ ] cross_val_score
- [ ] Search
  - [ ] GridSearchCV
  - [ ] RandomSearchCV
- [ ] Pipeline
  - [ ] Transformer → Model composition

---

## Visualizations (Optional)

- [ ] Decision boundary plot
- [ ] Confusion matrix
- [ ] Elbow method for KMeans
- [ ] Dendrogram for clustering

---

## Utility Modules (`utils/`)

- [ ] Math utilities
  - [ ] dot, norm, vector math
- [ ] Distance metrics
  - [ ] Euclidean, Manhattan, Cosine
- [ ] Statistical utilities
  - [ ] Gini, entropy
- [ ] Type & shape checks

---

## Built-in Datasets (`datasets/`)

- [ ] iris
- [ ] digits
- [ ] blobs (synthetic clustering)
- [ ] utility loader functions

---

## Testing (`tests/`)

- [ ] Unit tests for every component
- [ ] Comparison tests with Scikit-learn on toy datasets
- [ ] Code coverage checks using `pytest` + `coverage`

---

## Developer Utilities

- [ ] Package setup
  - [ ] `setup.py`
  - [ ] `pyproject.toml`
  - [ ] `requirements.txt`
- [ ] Linting with `flake8` or `ruff`
- [ ] Typing with `mypy`
- [ ] Auto-formatting with `black`

---

## Documentation (`docs/`)

- [ ] API reference with docstrings
- [ ] Markdown-based usage examples
- [ ] Optionally generate HTML docs with Sphinx or MkDocs

---

## Examples (`examples/`)

- [ ] Classification demo (Logistic Regression on Iris)
- [ ] Regression demo (Linear Regression on Boston-style data)
- [ ] Clustering demo (KMeans on blobs)

---

## Benchmarks (Optional)

- [ ] Benchmark basic models vs scikit-learn (accuracy, speed)
- [ ] Plot timing/memory usage

---

## Stretch Features (Advanced)

- [ ] Sparse matrix support (`scipy.sparse`)
- [ ] Model serialization (save/load with `joblib`)
- [ ] Web UI demo with Gradio or Streamlit
- [ ] Multi-output classification/regression
- [ ] Custom kernels for SVM or distances for KNN

---

## Algorithms (`algorithms/;`)

--- 
