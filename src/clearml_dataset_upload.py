from clearml import Dataset

dataset = Dataset.create(
    dataset_name="pizza_not_pizza_full",
    dataset_project="pizza_binary_classification"
)

dataset.add_files(path="../pizza_nopizza_dataset_full/pizza_not_pizza")
dataset.upload()
dataset.finalize()
