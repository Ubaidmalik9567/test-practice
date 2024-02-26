import pandas as pd
import pathlib
import yaml
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

def train_model(training_features,target_feature,n_estimator,max_depth,seed):
    model = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth,random_state=seed)
    model.fit(training_features, target_feature)
    return model

def save_pickle_model(model,save_dirPath):
    with open(save_dirPath + "\model.pkl","wb") as file:
        pickle.dump(model,file)
'''
def save_model(model, output_path): # also can use joblib to saving model
    joblib.dump(model, output_path + '\model.joblib')'''

def main():
    # get parameters
    curr_dir = pathlib.Path.cwd()
    home_dir = curr_dir.parent.parent

    params_file_location = home_dir.as_posix() + "\params.yaml" # F:\vs codes\mlops_x_dvc_deep_dive\dvc.yaml
    params = yaml.safe_load(open(params_file_location))['train_model']
    
     
    # Find path address  
    input_file_address = params["file_path"] # \dataa\processed\
    curr_dataset_dir = home_dir.as_posix() + input_file_address # F:\vs codes\mlops_x_dvc_deep_dive\dataa\processed\

    # save dir Address
    save_dirPath_location = home_dir.as_posix() + "\models"
    pathlib.Path(save_dirPath_location).mkdir(parents=True, exist_ok=True) # not necessory to write it if the directory already exists it will pass otherwise it will create a new one.
    
   
    train_dataset = pd.read_csv(curr_dataset_dir + '\train.csv') 
    x_feature = train_dataset.drop(columns=["Class"], axis=1)
    y_feature = train_dataset["Class"]
    
    # sign values to fucntion
    trained_model = train_model(x_feature,y_feature,params["n_estimators"],params["max_depth"],params["seed"])
    save_pickle_model(model=trained_model, save_dirPath=save_dirPath_location)

    '''save_model(model = trained_model, output_path = save_dirPath_location)'''
if __name__ == "__main__":
    main()
