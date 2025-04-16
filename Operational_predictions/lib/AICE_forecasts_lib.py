# -*- coding: utf-8 -*-
from global_vars import *
import metecflow
import hi_util
import os
import sys
import time
import h5py
import datetime
import numpy as np
###################################################################################################

# # Load predictors

def load_predictors(paths, today_date):
    filename = paths["predictors"] + "AICE_predictors_" + today_date + "T000000Z.h5"
    with h5py.File(filename, "r") as file:
        Dataset = {}
        Dataset["x"] = file["x"][:]
        Dataset["y"] = file["y"][:]
        Dataset["lat"] = file["lat"][:,:]
        Dataset["lon"] = file["lon"][:,:]
        Dataset["LSM"] = file["LSM"][:,:]
        Dataset["SICobs_AMSR2"] = file["SICobs_AMSR2"][:,:]
        Dataset["ECMWF_T2M"] = file["ECMWF_T2M"][:,:,:]
        Dataset["ECMWF_wind_x"] = file["ECMWF_x_wind"][:,:,:]
        Dataset["ECMWF_wind_y"] = file["ECMWF_y_wind"][:,:,:]
    return(Dataset)


# # Standardization data

def load_standardization_data(file_standardization, lead_time):
    standard = {}
    hf = h5py.File(file_standardization, "r")
    for var in hf:
        if "ECMWF" in var:
            standard[var] = np.array(hf[var])[lead_time]
        else:
            standard[var] = hf[var][()]
    hf.close()
    return(standard)


# # Data generator

class Data_generator():
    def __init__(self, list_predictors, lead_time, standard, Dataset, dim):
        self.list_predictors = list_predictors
        self.lead_time = lead_time
        self.standard = standard
        self.Dataset = Dataset
        self.dim = dim
        self.n_predictors = len(list_predictors)
    #
    def normalize(self, var, var_data):
        norm_data = (var_data - self.standard[var + "_min"]) / (self.standard[var + "_max"] - self.standard[var + "_min"])
        return(norm_data)
    #
    def data_generation(self): # Generates data containing batch_size samples
        # Initialization
        X = np.full((1, *self.dim, self.n_predictors), np.nan)
        #
        # Generate data
        for v, var in enumerate(self.list_predictors):
            if var == "LSM":
                var_data = self.Dataset["LSM"]
            elif var == "SICobs_AMSR2_SIC":
                var_data = self.Dataset["SICobs_AMSR2"]
            elif "ECMWF" in var:
                var_data = self.Dataset[var.replace("_cum", "")][self.lead_time,:,:]
            #
            X[0,:,:,v] = self.normalize(var, var_data)
        #
        return(X)


# # Function make_predictions

class make_predictions:
    def __init__(self, Dataset, Att_Res_UNet, model_params, filename_standardization, paths):
        self.Dataset = Dataset
        self.Att_Res_UNet = Att_Res_UNet
        self.model_params = model_params
        self.filename_standardization = filename_standardization
        self.paths = paths
    #
    def SIC_from_normalized_SIC(self, variable_name, field, standard):
        Predicted_SIC = field * (standard[variable_name + "_max"] - standard[variable_name + "_min"]) + standard[variable_name + "_min"]
        Predicted_SIC[Predicted_SIC > 100] = 100
        Predicted_SIC[Predicted_SIC < 0] = 0
        return(Predicted_SIC)
    #
    def predict(self):
        lead_times = np.linspace(0, 9, 10, dtype = int)
        SIC_pred = np.full((len(lead_times), self.model_params["patch_dim"][0], self.model_params["patch_dim"][1]), np.nan)
        #
        for leadtime in lead_times:
            file_model_weights = self.paths["static"] + "UNet_leadtime_" + str(leadtime) + "_days.h5"
            standard = load_standardization_data(self.filename_standardization, leadtime)
            unet_model = self.Att_Res_UNet(**self.model_params).make_unet_model()
            unet_model.load_weights(file_model_weights)
            #
            params_test = {"list_predictors": self.model_params["list_predictors"],
                           "lead_time": leadtime,
                           "standard": standard,
                           "Dataset": self.Dataset,
                           "dim": self.model_params["patch_dim"],
                          }
            #
            pred_sample = Data_generator(self.model_params["list_predictors"], leadtime, standard, self.Dataset, self.model_params["patch_dim"]).data_generation()
            predictions_SIC = np.squeeze(unet_model.predict(pred_sample))
            predictions_SIC = self.SIC_from_normalized_SIC("TARGET_AMSR2_SIC", predictions_SIC, standard)
            predictions_SIC[:,:][self.Dataset["LSM"] == 0] = np.nan
            predictions_SIC[predictions_SIC < 3] = 0
            SIC_pred[leadtime,:,:] = np.copy(predictions_SIC)
            del unet_model
        return(SIC_pred)


# # Function write_hdf5_forecasts

def write_hdf5_forecasts(Dataset, paths, today_date):
    timestamps = []
    #
    for lt in range(0, 10):
        timestamps.append((datetime.datetime.strptime(today_date, "%Y%m%d") + datetime.timedelta(days = lt)).timestamp())
    #
    path_output = paths["forecasts_temp"] 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "AICE_forecasts_" + today_date + "T000000Z.h5"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    #
    hf = h5py.File(output_filename, 'w')
    hf.create_dataset("time", data = timestamps)
    hf.create_dataset("x", data = Dataset["x"])
    hf.create_dataset("y", data = Dataset["y"])
    hf.create_dataset("lat", data = Dataset["lat"])
    hf.create_dataset("lon", data = Dataset["lon"])
    hf.create_dataset("SIC", data = Dataset["SIC_pred"])
    hf.close()
