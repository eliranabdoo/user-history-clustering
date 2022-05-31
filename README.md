Install requirements and run `main.py`.
Results are saved in the results directory with split_name (test/train), silhouette score and a uid.
Each result contains the clustered dataset, the configuration and optionally an html plotting the clusters.

ExperimentConfig contains all the experiment hyper-parameters together with ClusteringConfig and DimReductionConfig
Plotting the clusters is enabled only when choosing dim reduction of 3 or 2.
