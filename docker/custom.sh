# only needed for the autonomy_datasets_msgs package, which carries the dataset
# meta information that perception_msgs/Object cannot express
git clone --branch v1.5.0 https://github.com/thinking-cars/autonomy_datasets.git src/upstream/autonomy_datasets
rm -r src/upstream/autonomy_datasets/autonomy_datasets/
