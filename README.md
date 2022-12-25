Gaussian Mixture Model build in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- gaussian mixture model
- GMM
- clustering
- SVD
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- docker
- fastapi
- uvicorn

This is a Clustering Model that uses Guassian Mixtures implemented through Sklearn.

The algorithm aims to partition n observations into k clusters in which each observation belongs to the cluster with highest probability, assuming each of the k clusters has a multivariable gaussian distribution.

The data preprocessing step includes:

- for numerical variables
  - Standard scale data
  - TruncatedSVD

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT.

The model includes an inference service with 3 endpoints:

- /ping for health check and
- /infer for predictions with JSON input for instances, and JSON output of predictions
- /infer_file for predictions with multi-part CSV file input for instances, and JSON output for predictions

The inference service is implemented using fastapi+uvicorn.
