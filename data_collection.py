import opendatasets as od
import legacy_cgi as cgi

# Define dataset URL
dataset_url = "https://www.kaggle.com/wordsforthewise/lending-club"

# Download dataset
od.download(dataset_url)

print("Dataset downloaded successfully.")
