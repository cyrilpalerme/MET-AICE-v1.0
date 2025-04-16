#!/usr/bin/env python
# coding: utf-8

# In[278]:


import os
import time
import scipy
import netCDF4
import datetime
import numpy as np
import matplotlib.pyplot as plt


# # Constants

# In[279]:


date_min = "20240401"
date_max = "20250331"
distance_threshold = 20 * 1000 # meters
#
paths = {}
paths["Barents"] = "/lustre/storeB/project/copernicus/cosi/PCAPS/Barents_AICE_grid/"
paths["AICE"] = "/lustre/storeB/project/copernicus/cosi/AICE/archive/"
paths["Anomaly_persistence"] = "/lustre/storeB/project/copernicus/cosi/PCAPS/AMSR2_Anomaly_persistence/"
paths["ice_charts"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Ice_charts/"
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/AMSR2_obs/"
paths["distance_to_land"] = "/lustre/storeB/project/copernicus/cosi/AICE/Data/Distance_to_land/"
paths["output"] = "/lustre/storeB/users/cyrilp/AICE/Stats/Barents_AICE_domain_distance_to_land_" + str(int(0.001 * distance_threshold)) + "km/"
#
if os.path.exists(paths["output"]) == False:
    os.system("mkdir -p " + paths["output"])
#
threshold_ice_edge = 10
lead_times = np.arange(10)
spatial_resolution = 5000
N_Barents_members = 6
#
list_forecasts = ["Barents", "Barents_bias_corrected", "AICE"]


# # List dates

# In[280]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Load datasets

# In[281]:


def load_datasets(list_datasets, date_task, paths, distance_threshold = distance_threshold, N_Barents_members = N_Barents_members):
    Datasets = {}
    for ds in list_datasets:
        if ds == "Barents_bias_corrected":

            filename_distance_to_land = paths["distance_to_land"] + "AICE_LSM_distance_to_land.nc"
            with netCDF4.Dataset(filename_distance_to_land, "r") as nc:
                distance_to_land = nc.variables["distance_to_land"][:,:]

            date_pers = (datetime.datetime.strptime(date_task, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
            filename_AMSR2 = paths["AMSR2"] + date_pers[0:4] + "/" + date_pers[4:6] + "/" + "AMSR2_SIC_AICE_grid_" + date_pers + "T000000Z.nc"
            filename_Barents = paths["Barents"] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Barents_on_AICE_grid_" + date_task + ".nc"
            if (os.path.isfile(filename_AMSR2) == True) and (os.path.isfile(filename_Barents) == True):
                
                with netCDF4.Dataset(filename_AMSR2, "r") as nc:
                    AMSR2_SIC_persistence = nc.variables["SIC"][0,:,:] 

                with netCDF4.Dataset(filename_Barents, "r") as nc:
                    Barents_SIC = nc.variables["ice_concentration"][:] * 100
                    Barents_bias_corrected = np.full(np.shape(Barents_SIC), np.nan)
                    for me in range(0, N_Barents_members):
                        Barents_SIC_t0 = nc.variables["ice_concentration_first_hour"][me,0,:,:] * 100
                        ini_bias = Barents_SIC_t0 - AMSR2_SIC_persistence
                        ini_bias[distance_to_land <= distance_threshold] = 0
                        Barents_bias_corrected[me,:,:,:] = Barents_SIC[me,:,:,:] - ini_bias
                    Barents_bias_corrected[Barents_bias_corrected < 0] = 0
                    Barents_bias_corrected[Barents_bias_corrected > 100] = 100
                    Datasets[ds] = {}
                    Datasets[ds]["SIC"] = np.copy(Barents_bias_corrected)       
        else:
            if ds == "Barents":
                filename = paths[ds] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Barents_on_AICE_grid_" + date_task + ".nc"
            elif ds == "AICE":
                filename = paths[ds] + "AICE_forecasts_" + date_task + "T000000Z.nc"
            elif ds == "Anomaly_persistence":
                filename = paths[ds] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Anomaly_persistence_SIC_" + date_task + ".nc"
            elif ds == "ice_charts":
                filename = paths[ds] + date_task[0:4] + "/" + date_task[4:6] + "/" + "Ice_charts_AICE_grid_" + date_task + ".nc"
            elif ds == "AMSR2":
                filename = paths[ds] + date_task[0:4] + "/" + date_task[4:6] + "/" + "AMSR2_SIC_AICE_grid_" + date_task + "T000000Z.nc"
            #
            if os.path.isfile(filename) == True:
                with netCDF4.Dataset(filename, "r") as nc:
                    Datasets[ds] = {}
                    for var in nc.variables:
                        if ds == "Barents" and var == "ice_concentration":
                            Datasets[ds]["SIC"] = nc.variables[var][:] * 100
                        else:
                            Datasets[ds][var] = nc.variables[var][:]          
    return(Datasets)   


# # Land sea mask

# In[282]:


def make_common_land_sea_mask(paths, distance_threshold):
    filename_distance_to_land = paths["distance_to_land"] + "AICE_LSM_distance_to_land.nc"
    with netCDF4.Dataset(filename_distance_to_land, "r") as nc:
        LSM_including_coastlines = nc.variables["LSM"][:,:]
        distance_to_land = nc.variables["distance_to_land"][:,:]

    LSM_excluding_coastlines = np.copy(LSM_including_coastlines)
    LSM_excluding_coastlines[distance_to_land <= distance_threshold] = 0  # These grid points are considered as land
    
    return(LSM_including_coastlines, LSM_excluding_coastlines)


# # Calculate ice edge length

# In[283]:


class calculate_ice_edge_length():
    def __init__(self, SIC, LSM, threshold_ice_edge, spatial_resolution):
        self.SIC = np.squeeze(SIC)
        self.LSM = np.squeeze(LSM)
        self.threshold_ice_edge = threshold_ice_edge
        self.spatial_resolution = spatial_resolution
        SIE = np.zeros(np.shape(self.SIC))
        SIE[self.SIC >= self.threshold_ice_edge] = 1
        self.SIE = SIE
    #
    def ice_edge_position(self):
        # Shift the arrays to get the neighboring values
        SIE_up = np.roll(self.SIE, -1, axis = 0)
        SIE_down = np.roll(self.SIE, 1, axis = 0)
        SIE_left = np.roll(self.SIE, -1, axis = 1)
        SIE_right = np.roll(self.SIE, 1, axis = 1)
        #
        LSM_up = np.roll(self.LSM, -1, axis = 0)
        LSM_down = np.roll(self.LSM, 1, axis = 0)
        LSM_left = np.roll(self.LSM, -1, axis = 1)
        LSM_right = np.roll(self.LSM, 1, axis = 1)
        #
        # Mask the borders 
        SIE_up[-1, :] = np.nan
        SIE_down[0, :] = np.nan
        SIE_left[:, -1] = np.nan
        SIE_right[:, 0] = np.nan
        #
        LSM_up[-1, :] = np.nan
        LSM_down[0, :] = np.nan
        LSM_left[:, -1] = np.nan
        LSM_right[:, 0] = np.nan
        #
        neighbors_SIE = np.stack([SIE_up, SIE_down, SIE_left, SIE_right], axis = 0)
        neighbors_LSM = np.stack([LSM_up, LSM_down, LSM_left, LSM_right], axis = 0)
        #
        open_ocean_neighbors = np.logical_and(neighbors_SIE == 0, neighbors_LSM == 1)
        nb_neighbors_open_ocean = np.nansum(open_ocean_neighbors, axis = 0)
        #
        ice_edge = np.zeros(np.shape(self.SIE))
        ice_edge[np.logical_and(nb_neighbors_open_ocean >= 1, self.SIE == 1)] = 1
        return(ice_edge)
    #
    def length_sea_ice_edge(self, ice_edge):
        # Convolution kernel to count the neighbors
        kernel = np.array([[0, 1, 0], 
                           [1, 0, 1], 
                           [0, 1, 0]])
        # Count neighbors using convolution
        neighbor_count = scipy.ndimage.convolve(ice_edge, kernel, mode = "constant", cval = 0)
        length_sie = np.zeros(np.shape(ice_edge))
        length_sie[np.logical_and(neighbor_count == 0, ice_edge == 1)] = np.sqrt(2) * self.spatial_resolution
        length_sie[np.logical_and(neighbor_count == 1, ice_edge == 1)] = 0.5 * (self.spatial_resolution + np.sqrt(2) * self.spatial_resolution)
        length_sie[np.logical_and(neighbor_count >= 2, ice_edge == 1)] = self.spatial_resolution
        sie_length = np.sum(length_sie)
        return(sie_length)
    #
    def __call__(self):
        ice_edge = self.ice_edge_position()
        ice_edge_length = self.length_sea_ice_edge(ice_edge)
        return(ice_edge_length, ice_edge)


# # Verification scores

# In[284]:


class ice_edge_verification_scores():
    def __init__(self, SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False):
        self.SIC_obs = np.squeeze(SIC_obs)  # If deterministic "SIC_obs and SIC_forecasts" must be 2D arrays (y, x). If probabilistic, they must be 3D arrays (member, y x)
        self.SIC_forecast = np.squeeze(SIC_forecast)
        self.ice_edge_length = ice_edge_length
        self.threshold_ice_edge = threshold_ice_edge
        self.spatial_resolution = spatial_resolution
        self.LSM = np.squeeze(LSM)
        self.probabilistic = probabilistic
    #
    def sea_ice_extent(self, SIC):
        SIE = np.zeros(np.shape(SIC))
        SIE[SIC >= self.threshold_ice_edge] = 1
        return(SIE)
    #
    def sea_ice_probability(self, SIC):
        SIE = np.zeros(np.shape(SIC))
        SIE[SIC >= self.threshold_ice_edge] = 1
        SIP = np.sum(SIE, axis = 0) / np.shape(SIE)[0]
        return(SIP)
    #
    def Root_Mean_Square_Error(self):
        # SIC_forec and SIC_obs must be 2D arrays (y, x)
        SIC_forec = np.ndarray.flatten(self.SIC_forecast[self.LSM == 1])
        SIC_ob = np.ndarray.flatten(self.SIC_obs[self.LSM == 1])
        MSE = np.sum((SIC_forec - SIC_ob) ** 2) / len(SIC_ob)
        RMSE = np.sqrt(MSE)
        return(RMSE)
    #
    def IIEE(self):
        SIE_obs = self.sea_ice_extent(self.SIC_obs)
        SIE_forecast = self.sea_ice_extent(self.SIC_forecast)
        SIE_obs[self.LSM < 1] = 0
        SIE_forecast[self.LSM < 1] = 0
        Flag_SIE = np.full(np.shape(SIE_obs), np.nan)
        Flag_SIE[SIE_forecast == SIE_obs] = 0
        Flag_SIE[SIE_forecast < SIE_obs] = -1
        Flag_SIE[SIE_forecast > SIE_obs] = 1
        Underestimation = np.sum(Flag_SIE == -1) * self.spatial_resolution ** 2
        Overestimation = np.sum(Flag_SIE == 1) * self.spatial_resolution ** 2
        IIEE_metric = Underestimation + Overestimation
        return(IIEE_metric, Underestimation, Overestimation)
    #
    def SPS(self):
        SIP_obs = self.sea_ice_extent(self.SIC_obs)
        SIP_forecast = self.sea_ice_probability(self.SIC_forecast)
        SIP_obs[self.LSM < 1] = 0
        SIP_forecast[self.LSM < 1] = 0
        SPS_metric = np.nansum((self.spatial_resolution ** 2) * (SIP_forecast - SIP_obs)**2)
        return(SPS_metric)
    #
    def __call__(self):
        if self.probabilistic == False:
            IIEE_distance = self.IIEE()[0] / self.ice_edge_length
            RMSE = self.Root_Mean_Square_Error()
            if np.ma.isMaskedArray(RMSE) == True:
                RMSE = np.nan
            return(IIEE_distance, RMSE)
        #
        elif self.probabilistic == True:
            N_members = np.shape(SIC_forecast)[0]
            SPS_distance = self.SPS() / self.ice_edge_length
            return(SPS_distance)


# # Write_scores function

# In[285]:


def save_scores(Metrics, ds_obs, lt, date_min, date_max, start_date, paths = paths):
    header = ""
    scores = ""
    for vi, var in enumerate(sorted(Metrics.keys(), reverse = True)):
        header = header + "\t" + var   
        scores = scores + "\t" + str(Metrics[var]) 
    #
    output_file = paths["output"] + "Scores_ice_edge_reference_" + ds_obs + "_lead_time_" + str(lt) + ".txt"
    if start_date == date_min:
        if os.path.isfile(output_file) == True:
            os.system("rm " + output_file)
    #
    if os.path.isfile(output_file) == False:
        output = open(output_file, 'a')
        output.write(header + "\n")
        output.close()
    #
    output = open(output_file, 'a')
    output.write(scores + "\n")
    output.close()


# # Verification using AMSR2 observations as reference

# In[286]:


def verification_AMSR2_as_reference(list_dates, list_forecasts, paths, spatial_resolution, LSM):
    for sdi, start_date in enumerate(list_dates):
        persistence_date = (datetime.datetime.strptime(start_date, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
        Forecasts = load_datasets(list_forecasts, start_date, paths)

        if all(key in Forecasts for key in list_forecasts) == True:
            Obs_persistence = load_datasets(["AMSR2"], persistence_date, paths)
            Anomaly_persistence = load_datasets(["Anomaly_persistence"], persistence_date, paths)

            for lt in lead_times:
                forec_date = (datetime.datetime.strptime(start_date, "%Y%m%d") + datetime.timedelta(days = int(lt))).strftime("%Y%m%d")
                try:
                    Obs_target = load_datasets(["AMSR2"], forec_date, paths)

                    Metrics = {}
                    Metrics["start_date"] = start_date
                    Metrics["forecast_date"] = forec_date
                    
                    SIC_obs = np.squeeze(Obs_target["AMSR2"]["SIC"])
                    SIC_obs[LSM == 0] = 0
                    ice_edge_length = calculate_ice_edge_length(SIC_obs, LSM, threshold_ice_edge, spatial_resolution)()[0]
                    Metrics["Ice_edge_length"] = ice_edge_length

                    if "Anomaly_persistence" in Anomaly_persistence:
                        SIC_anomaly_pers = Anomaly_persistence["Anomaly_persistence"]["SIC"][lt + 1,:,:]   
                        SIC_anomaly_pers[LSM == 0] = 0
                        Metrics["IIEE_distance_Anomaly_persistence"], Metrics["RMSE_Anomaly_persistence"] = ice_edge_verification_scores(SIC_obs, SIC_anomaly_pers, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                    else:
                        Metrics["IIEE_distance_Anomaly_persistence"] = np.nan
                        Metrics["RMSE_Anomaly_persistence"] = np.nan
                    
                    SIC_pers = np.squeeze(Obs_persistence["AMSR2"]["SIC"])
                    SIC_pers[LSM == 0] = 0
                    Metrics["IIEE_distance_Persistence_AMSR2"], Metrics["RMSE_Persistence_AMSR2"] = ice_edge_verification_scores(SIC_obs, SIC_pers, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()  
                
                    for ds_forec in Forecasts:
                        if ds_forec == "AICE":
                            SIC_forecast = Forecasts[ds_forec]["SIC"][lt,:,:]
                            SIC_forecast[LSM == 0] = 0
                            Metrics["IIEE_distance_" + ds_forec], Metrics["RMSE_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                        elif (ds_forec == "Barents") or (ds_forec == "Barents_bias_corrected"):
                            if lt < 4:
                                SIC_ensemble_mean = np.nanmean(Forecasts[ds_forec]["SIC"][:,lt,:,:], axis = 0)
                                SIC_ensemble_mean[LSM == 0] = 0
                                SIC_forecast = Forecasts[ds_forec]["SIC"][:,lt,:,:]
                                LSM_extend = np.repeat(np.expand_dims(LSM, axis = 0), N_Barents_members, axis = 0)
                                SIC_forecast[LSM_extend == 0] = 0
                                Metrics["IIEE_distance_ensemble_mean_" + ds_forec], Metrics["RMSE_ensemble_mean_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_ensemble_mean, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                                #Metrics["SPS_distance_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = True)()
                                #
                                for me in range(0, N_Barents_members):
                                    member = "{:02d}".format(me)
                                    Metrics["IIEE_distance_" + ds_forec + "_member_" + member], Metrics["RMSE_" + ds_forec + "_member_" + member] = ice_edge_verification_scores(SIC_obs, SIC_forecast[me,:,:], ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)() 
                    
                    for var in Metrics:
                        if "date" not in var:
                            Metrics[var] = np.round(Metrics[var], 3)
               
                    save_scores(Metrics, "AMSR2", lt, date_min, date_max, start_date = start_date, paths = paths)
                except:
                    pass


# # Verification using ice charts as reference

# In[287]:


def verification_ice_charts_as_reference(list_dates, list_forecasts, paths, spatial_resolution, LSM):
    for sdi, start_date in enumerate(list_dates):
        persistence_date = (datetime.datetime.strptime(start_date, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
        Forecasts = load_datasets(list_forecasts, start_date, paths)

        if all(key in Forecasts for key in list_forecasts) == True:
            Obs_persistence = load_datasets(["ice_charts", "AMSR2"], persistence_date, paths)
            Anomaly_persistence = load_datasets(["Anomaly_persistence"], persistence_date, paths)
            #
            for lt in lead_times:
                forec_date = (datetime.datetime.strptime(start_date, "%Y%m%d") + datetime.timedelta(days = int(lt))).strftime("%Y%m%d")
                try:
                    Obs_target = load_datasets(["ice_charts"], forec_date, paths)

                    Metrics = {}
                    Metrics["start_date"] = start_date
                    Metrics["forecast_date"] = forec_date
                    
                    SIC_obs = np.squeeze(Obs_target["ice_charts"]["SIC"]) 
                    SIC_obs[LSM == 0] = 0
                    ice_edge_length = calculate_ice_edge_length(SIC_obs, LSM, threshold_ice_edge, spatial_resolution)()[0]
                    Metrics["Ice_edge_length"] = ice_edge_length

                    if "Anomaly_persistence" in Anomaly_persistence:
                        SIC_anomaly_pers = Anomaly_persistence["Anomaly_persistence"]["SIC"][lt + 1,:,:]   
                        SIC_anomaly_pers[LSM == 0] = 0
                        Metrics["IIEE_distance_Anomaly_persistence"], Metrics["RMSE_Anomaly_persistence"] = ice_edge_verification_scores(SIC_obs, SIC_anomaly_pers, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                    else:
                        Metrics["IIEE_distance_Anomaly_persistence"] = np.nan
                        Metrics["RMSE_Anomaly_persistence"] = np.nan
                    #
                    for ds_pers in Obs_persistence:
                        SIC_pers = np.squeeze(Obs_persistence[ds_pers]["SIC"])
                        SIC_pers[LSM == 0] = 0
                        Metrics["IIEE_distance_Persistence_" + ds_pers], Metrics["RMSE_Persistence_" + ds_pers] = ice_edge_verification_scores(SIC_obs, SIC_pers, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()  

                    for ds_forec in Forecasts:
                        if ds_forec == "AICE":
                            SIC_forecast = Forecasts[ds_forec]["SIC"][lt,:,:]
                            SIC_forecast[LSM == 0] = 0
                            Metrics["IIEE_distance_" + ds_forec], Metrics["RMSE_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                        elif (ds_forec == "Barents") or (ds_forec == "Barents_bias_corrected"):
                            if lt < 4:
                                SIC_ensemble_mean = np.nanmean(Forecasts[ds_forec]["SIC"][:,lt,:,:], axis = 0)
                                SIC_ensemble_mean[LSM == 0] = 0
                                SIC_forecast = Forecasts[ds_forec]["SIC"][:,lt,:,:]
                                LSM_extend = np.repeat(np.expand_dims(LSM, axis = 0), N_Barents_members, axis = 0)
                                SIC_forecast[LSM_extend == 0] = 0
                                Metrics["IIEE_distance_ensemble_mean_" + ds_forec], Metrics["RMSE_ensemble_mean_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_ensemble_mean, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                                #Metrics["SPS_distance_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = True)()
                                #
                                for me in range(0, N_Barents_members):
                                    member = "{:02d}".format(me)
                                    Metrics["IIEE_distance_" + ds_forec + "_member_" + member], Metrics["RMSE_" + ds_forec + "_member_" + member] = ice_edge_verification_scores(SIC_obs, SIC_forecast[me,:,:], ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                    
                    if "ice_charts" not in Obs_persistence:
                        Metrics["IIEE_distance_Persistence_ice_charts"] = np.nan
                        Metrics["RMSE_Persistence_ice_charts"] = np.nan
                    if "AMSR2" not in Obs_persistence:                
                        Metrics["IIEE_distance_Persistence_AMSR2"] = np.nan   
                        Metrics["RMSE_Persistence_AMSR2"] = np.nan   
                    #
                    for var in Metrics:
                        if "date" not in var:
                            Metrics[var] = np.round(Metrics[var], 3)
                    #
                    save_scores(Metrics, "ice_charts", lt, date_min, date_max, start_date = start_date, paths = paths)
                except:
                    pass


# # Main

# In[288]:


list_dates = make_list_dates(date_min, date_max)

LSM_including_coastlines, LSM_excluding_coastlines = make_common_land_sea_mask(paths, distance_threshold)

verification_AMSR2_as_reference(list_dates = list_dates, 
                                list_forecasts = list_forecasts, 
                                paths = paths, 
                                spatial_resolution = spatial_resolution, 
                                LSM = LSM_excluding_coastlines)

verification_ice_charts_as_reference(list_dates = list_dates, 
                                     list_forecasts = list_forecasts, 
                                     paths = paths, 
                                     spatial_resolution = spatial_resolution, 
                                     LSM = LSM_including_coastlines)

