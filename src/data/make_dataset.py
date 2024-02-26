import pandas as pd
import pathlib
import yaml
import sys
import os
from sklearn.model_selection import train_test_split


def load_data(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset


def split_data(dataset, test_split_size,seed):
    train,test = train_test_split(dataset, test_size=test_split_size, random_state=seed)
    return train,test # now train,test data store in these variable which(train&test)

 
'''def save_split_data(train,test,save_output_dir):
    # save spliting data into specific path(output path) which we provide
    pathlib.Path(save_output_dir).mkdir(parents=True, exist_ok=True) # make directory if not exists
    #After giving path now above line make dir and save both variable as csvFile
    train.to_csv(save_output_dir + "\train.csv", index=False)# to_csv(require file name but we make
    test.to_csv(save_output_dir + "\test.csv",index=False) # cvsFile path like:given path with name
'''
def save_split_data(train, test, save_output_dir):
    # Create directory if it does not exist
    os.makedirs(save_output_dir, exist_ok=True)
    
    # Define file paths
    train_file_path = os.path.join(save_output_dir, "train.csv")
    test_file_path = os.path.join(save_output_dir, "test.csv")
    
    # Save train and test data as CSV files
    train.to_csv(train_file_path, index=False)
    test.to_csv(test_file_path, index=False)
    
    return train_file_path, test_file_path

def main():                          # WindowsPath('f:\vs codes\mlops_x_dvc_deep_dive\src\data')
    curr_dir = pathlib.Path.cwd() # this line give current working directory: which above
    home_dir = curr_dir.parent.parent# WindowsPath('f:\vs codes\mlops_x_dvc_deep_dive')
    
    # now we know all file location because at this point we are in main folder: home_dir

    params_file = home_dir.as_posix() + "\params.yaml" # as_posix () method is used to convert Path object: which start windowsPath: into simple string from 
    params = yaml.safe_load(open(params_file))["make_dataset"]
    
    '''sys.argv[1] retrieves the  command-line argument passed to the script when it is executed
     in the terminal. 
    assumes that the script is executed from the terminal with the input file path provided as
    the first argument after the script name. Then, the rest of the process remains the 
    same as before: parameters are loaded from params.yaml, data is loaded, split, and saved 
    accordingly.'''

    input_file_path =  params['dataset_path'] # '\dataa\raw\creditcard.csv' 
    print(input_file_path)
    dataset_file_path = home_dir.as_posix() + input_file_path # 'f:\vs codes\mlops_x_dvc_deep_dive + \data\raw\creditcard.csv'
    save_dirPath = home_dir.as_posix() +"\dataa\processed" # Save processed data in processed folder

    #add value in above all function
    data = load_data(dataset_file_path)   # load data from given file
    train_data, test_data = split_data(dataset=data,test_split_size=params["test_split_size"],seed=params["seed"])
    save_split_data(train=train_data,test=test_data,save_dirPath=save_dirPath)

    if  __name__ == "__main__":
        main()
