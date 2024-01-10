# estimation

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd
import os
import xarray as xr
from scipy import optimize

def load_longrunmip_data(model, exp = 'abrupt2x', length_restriction = None):
    directory = '../longrunmip_data/'
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
        directory = '../Processed_data/Global_annual_anomalies/'
    elif datatype == 'means':
        directory = '../Processed_data/Global_annual_means/'
    elif datatype == 'forcing':
        directory = '../Estimates/Transient_forcing_estimates/'
    modelexpdirectory = os.path.join(directory, model, exp)
    filenames = [f.name for f in os.scandir(modelexpdirectory) if f.name not in ['.ipynb_checkpoints', '.DS_Store']]

    members = [file.rsplit('_')[2] for file in filenames]
    members.sort()
    return members

def load_anom(model, exp, member, length_restriction = None):
    filename = model + '_' + exp + '_' + member + '_anomalies.csv'
    file = os.path.join('../Processed_data/Global_annual_anomalies/', model, exp, filename)
    
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

def Gregory_linreg(model, exp, members, startyear = 1, stopyear = 150):
    alltas = []; alltoarad = []
    # use datapoints from all specified members in the regression
    for (mb, member) in enumerate(members):
        data = load_anom(model, exp, member, length_restriction = stopyear)
        data = data[(startyear-1):]
        tas = data['tas']
        toarad = data['rsdt'] - data['rsut'] - data['rlut']
        if exp == 'abrupt-0p5xCO2': # change sign of data
            tas = -tas; toarad = -toarad
        alltas = np.append(alltas, tas)
        alltoarad = np.append(alltoarad,toarad)
        
    regpar = np.polyfit(alltas, alltoarad, 1)
    if exp == 'abrupt-4xCO2':
        gF2x = regpar[1]/2; gT2x = -regpar[1]/(2*regpar[0])
        linfit = np.polyval(regpar, [0, gT2x*2])
    else:
        gF2x = regpar[1]; gT2x = -regpar[1]/(regpar[0])
        linfit = np.polyval(regpar, [0, gT2x])
    return gF2x, gT2x, linfit

def etminan_co2forcing(C, C0 = 284.3169998547858, N_bar = 273.021049007482):
    a1 = -2.4*10**(-7)
    b1 = 7.2*10**(-4)
    c1 = -2.1*10**(-4)
    # Etminan paper says: The expressions are valid in the ranges 180–2000ppm for CO2,
    return (a1*(C-C0)**2 + b1*np.abs(C-C0) + c1*N_bar + 5.36)*np.log(C/C0)

def recT_from4xCO2(T4xCO2, F4xCO2, forcing_rec_exp):
    # Use what we know about 4xCO2 forcing and temperature response T4xCO2
    # to reconstruct the temperature response to the forcing "forcing_rec_exp"
    # index 0 should correspond to year 0, where T4xCO2[0]=0,
    # and we set the forcing difference in year 0 to 0
    lf = len(forcing_rec_exp)
    delta_f_vector = np.concatenate(([0], np.diff(forcing_rec_exp)))
    
    if lf > len(T4xCO2):
        print('Cannot compute reconstruction for an experiment longer than the 4xCO2 experiment')
    else:
        W = np.full((lf,lf),0.) # will become a lower triangular matrix
        for i in range(0,lf):
            for j in range(0,i):
                W[i,j] = delta_f_vector[i-j]/F4xCO2
        #print(W)
        T_rec_exp = W@(T4xCO2[:lf])
        return T_rec_exp
        #return W