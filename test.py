import kagglehub

handle = 'khxjamohammed/'
local_dataset_dir = 'path/to/local/dataset/dir'

# Create a new dataset
kagglehub.dataset_upload(handle, local_dataset_dir)

# You can then create a new version of this dataset and include version notes.
kagglehub.dataset_upload(handle, local_dataset_dir, version_notes='improved data')

# You can also specify a list of patterns for files/dirs to ignore.
# These patterns are combined with 'kagglehub.datasets.DEFAULT_IGNORE_PATTERNS'
# to determine which files and directories to exclude. 
# To ignore entire directories, include a trailing slash (/) in the pattern.
kagglehub.dataset_upload(handle, local_dataset_dir, ignore_patterns=["original/", "*.tmp"])
