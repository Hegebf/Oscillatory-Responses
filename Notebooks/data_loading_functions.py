# estimation

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import os
import xarray as xr
from scipy import optimize

def load_longrunmip_data(model, exp = 'abrupt2x', length_restriction = None):
    directory = '../../longrunmip_data/'
    file_list = [ f.name for f in os.scandir(directory) if f.is_file()]
    file_list_namesplits = [file.rsplit("_") for file in file_list]
    files_model_names = [file_list_namesplits[k][1] for k in range(len(file_list_namesplits))]
    
    model_index = [i for i,x in enumerate(files_model_names) if x==model]
    model_files = [file_list[model_index[k]] for k in range(len(model_index))]
    model_exp_files = [file_list[model_index[k]] for k in range(len(model_index)) if file_list_namesplits[model_index[k]][2] == exp]
    model_exp_files.sort() # make sure tas is the last file

    ds_tas = xr.open_dataset(directory + model_exp_files[-1])
    ds_toarad = xr.open_mfdataset(directory + model_exp_files[0])
    deltaT = ds_tas.tas.values
    toarad = ds_toarad['netTOA'].values
    if length_restriction != None:
        deltaT = deltaT[:length_restriction]
        toarad = toarad[:length_restriction]
    return [deltaT, toarad]

def find_members(model, exp, datatype = 'anomalies'):
    if datatype == 'anomalies':
        directory = '../../Processed_data/Global_annual_anomalies/'
    elif datatype == 'means':
        directory = '../../Processed_data/Global_annual_means/'
    elif datatype == 'forcing':
        directory = '../../Estimates/Transient_forcing_estimates/'
    modelexpdirectory = os.path.join(directory, model, exp)
    filenames = [f.name for f in os.scandir(modelexpdirectory) if f.name not in ['.ipynb_checkpoints', '.DS_Store']]

    members = [file.rsplit('_')[2] for file in filenames]
    members.sort()
    return members

def load_anom(model, exp, member, length_restriction = None):
    filename = model + '_' + exp + '_' + member + '_anomalies.csv'
    file = os.path.join('../../Processed_data/Global_annual_anomalies/', model, exp, filename)
    
    data = pd.read_csv(file, index_col=0)
    if model != 'AWI-CM-1-1-MR': # maybe not neccessary after updating the data archive?
        data = data.dropna() #.reset_index()
    if length_restriction != None:
        data = data[:length_restriction]
    return data

def member_mean_tas(model, members, length_restriction = None, exp = 'abrupt-4xCO2'):
    # add also 0 response at time 0
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = length_restriction)
        deltaT0 = np.concatenate([[0],data['tas']])
        if mb == 0:
            if length_restriction == None:
                length_restriction = len(deltaT0)-1 # full length of data
            T_array = np.full((len(members), length_restriction + 1), np.nan) 
        T_array[mb,:len(deltaT0)] = deltaT0
    meanT0 = np.nanmean(T_array, axis=0)
    return meanT0

def mean_4xCO2toarad(model, members, length_restriction = None, exp = 'abrupt-4xCO2'):
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = length_restriction)
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        if mb == 0:
            if length_restriction == None:
                length_restriction = len(toarad) # full length of data
            N_array = np.full((len(members), length_restriction), np.nan) 
        N_array[mb,:len(toarad)] = toarad
    meanN = np.nanmean(N_array, axis=0)
    return meanN
