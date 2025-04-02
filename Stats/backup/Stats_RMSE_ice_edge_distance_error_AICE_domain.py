#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import scipy
import netCDF4
import datetime
import numpy as np
import matplotlib.pyplot as plt


# # Constants

# In[ ]:


date_min = "20240321"
date_max = "20250313"
#
paths = {}
paths["AICE"] = "/lustre/storeB/project/fou/hi/oper/aice/archive/"
paths["Anomaly_persistence"] = "/lustre/storeB/project/copernicus/cosi/PCAPS/AMSR2_Anomaly_persistence/"
paths["ice_charts"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Ice_charts/"
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/AMSR2_obs/"
paths["output"] = "/lustre/storeB/users/cyrilp/AICE/Stats/AICE_domain/"
#
if os.path.exists(paths["output"]) == False:
    os.system("mkdir -p " + paths["output"])
#
threshold_ice_edge = 10
lead_times = np.arange(10)
spatial_resolution = 5000
N_Barents_members = 6
#
list_forecasts = ["AICE"]


# # List dates

# In[3]:


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

# In[4]:


def load_datasets(list_datasets, date_task, paths, N_Barents_members = N_Barents_members):
    Datasets = {}
    for ds in list_datasets:      
        if ds == "AICE":
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
                    Datasets[ds][var] = nc.variables[var][:]          
    return(Datasets)   


# # Land sea mask

# In[5]:


def make_common_land_sea_mask(Forecasts, Observations):
    # 1 ocean, 0 land
    #
    LSM_AICE = np.ones(np.shape(Forecasts["AICE"]["SIC"][0,:,:]))
    LSM_AICE[np.isnan(Forecasts["AICE"]["SIC"][0,:,:]) == True] = 0
    #
    LSM_ice_charts = np.ones(np.shape(Observations["ice_charts"]["SIC"][0,:,:]))
    LSM_ice_charts[np.isnan(Observations["ice_charts"]["SIC"][0,:,:]) == True] = 0
    #
    LSM = np.zeros(np.shape(LSM_AICE))
    LSM[np.logical_and(LSM_ice_charts == 1, LSM_AICE == 1)] = 1
    #
    return(LSM)


# # Calculate ice edge length

# In[6]:


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

# In[7]:


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

# In[8]:


def save_scores(Metrics, ds_obs, lt, date_min, date_max, paths = paths):
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


# # Data processing

# In[9]:


list_dates = make_list_dates(date_min, date_max)
for sdi, start_date in enumerate(list_dates):
    print(start_date)
    persistence_date = (datetime.datetime.strptime(start_date, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
    Forecasts = load_datasets(list_forecasts, start_date, paths)
    if all(key in Forecasts for key in list_forecasts) == True:
        Obs_persistence = load_datasets(["ice_charts", "AMSR2"], persistence_date, paths)
        Anomaly_persistence = load_datasets(["Anomaly_persistence"], persistence_date, paths)
        #
        if "LSM" not in locals():
            LSM = make_common_land_sea_mask(Forecasts, Obs_persistence)
        #
        for lt in lead_times:
            forec_date = (datetime.datetime.strptime(start_date, "%Y%m%d") + datetime.timedelta(days = int(lt))).strftime("%Y%m%d")
            Obs_target = load_datasets(["AMSR2", "ice_charts"], forec_date, paths)
            #
            for ds_obs in Obs_target:
                Metrics = {}
                Metrics["start_date"] = start_date
                Metrics["forecast_date"] = forec_date
                #
                SIC_obs = np.squeeze(Obs_target[ds_obs]["SIC"])
                SIC_obs[LSM == 0] = 0
                ice_edge_length = calculate_ice_edge_length(SIC_obs, LSM, threshold_ice_edge, spatial_resolution)()[0]
                Metrics["Ice_edge_length"] = ice_edge_length
                #
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
                #
                for ds_forec in Forecasts:
                    if ds_forec == "AICE":
                        SIC_forecast = Forecasts[ds_forec]["SIC"][lt,:,:]
                        SIC_forecast[LSM == 0] = 0
                        Metrics["IIEE_distance_" + ds_forec], Metrics["RMSE_" + ds_forec] = ice_edge_verification_scores(SIC_obs, SIC_forecast, ice_edge_length, threshold_ice_edge, spatial_resolution, LSM, probabilistic = False)()
                #
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
                save_scores(Metrics, ds_obs, lt, date_min, date_max, paths = paths)


# In[ ]:




