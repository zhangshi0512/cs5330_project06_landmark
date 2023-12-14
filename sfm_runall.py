import sys
from sfm_feature_matching import main as feature_matching
from sfm_reconstruction import main as reconstruction
from sfm_dense_reconstruction import main as dense_reconstruction

def run_all(dataset_name):
    feature_matching(dataset_name)
    reconstruction(dataset_name)
    dense_reconstruction(dataset_name)

# TODO2: not hard code dataset

print('sys.argv', sys.argv)

# TODO3: visulization output screenshot
if __name__ == "__main__":
    if (len(sys.argv) < 2):
        raise Exception("You must input a dataset's name")
    
    # TBD: check dataset's name
    run_all(sys.argv[1])