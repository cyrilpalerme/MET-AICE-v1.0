#!/bin/bash -f
#$ -N AMSR2_ice_edge_lengths
#$ -l h_rt=00:03:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=1G,mem_free=1G,h_data=1G
#$ -q research-r8.q
#$ -t 1-31
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-10-2022

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/AMSR2_ice_edge_lengths_""$SGE_TASK_ID"".py" << EOF
###################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import os
import h5py
import datetime
import netCDF4
import numpy as np


# In[2]:

#
date_min = "20240101"
date_max = "20240131"
#
paths = {}
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP2/SIC/v0.1/"
paths["UNet"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Training/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/AMSR2_ice_edge_lengths/"
#
SIC_thresholds = ["10", "15", "20"]
#
grid_resolution = 5000 # meters


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


# In[4]:


def ice_edge_position(SIE, LSM):
    # LSM => 1 ocean  / 0 land and on the same grid as the SIE
    xdim, ydim = np.shape(SIE)
    nb_neighbors_open_ocean = np.zeros((xdim, ydim))
    for i in range(0, xdim):
        for j in range(0, ydim):
            if (i > 0 and i < xdim-1 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i == 0 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i == xdim-1 and j > 0 and j < ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j-1], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j-1], LSM[i, j+1]]
            elif (i > 0 and i < xdim-1 and j == 0):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j+1]]
            elif (i > 0 and i < xdim-1 and j == ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i+1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i+1,j], LSM[i, j-1]]
            elif (i == 0 and j == 0):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j+1]]
            elif (i == 0 and j == ydim-1):
                neighbors_SIE = [SIE[i+1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i+1,j], LSM[i, j-1]]
            elif (i == xdim-1 and j == 0):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j+1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j+1]]
            elif (i == xdim-1 and j == ydim-1):
                neighbors_SIE = [SIE[i-1,j], SIE[i, j-1]]
                neighbors_ocean = [LSM[i-1,j], LSM[i, j-1]]
            #
            neighbors_SIE = np.array(neighbors_SIE)
            neighbors_ocean = np.array(neighbors_ocean)
            neighbors_open_ocean = np.zeros(len(neighbors_SIE))
            neighbors_open_ocean[np.logical_and(neighbors_SIE == 0, neighbors_ocean == 1)] = 1
            nb_neighbors_open_ocean[i,j] = np.nansum(neighbors_open_ocean)
    ###
    ice_edge = np.logical_and(nb_neighbors_open_ocean >= 1, SIE == 1)
    return(ice_edge)


# In[5]:


def length_sea_ice_edge(ice_edge, spatial_resolution):
    xdim, ydim = np.shape(ice_edge)
    length_sie = np.zeros(np.shape(ice_edge))
    for i in range(0, xdim):
        for j in range(ydim):
            if ice_edge[i,j] == 1:
                if (i > 0 and i < xdim-1 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i,j-1], ice_edge[i,j+1]]))
                elif (i == 0 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j-1], ice_edge[i, j+1]]))
                elif (i == xdim-1 and j > 0 and j < ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j-1], ice_edge[i, j+1]]))
                elif (i > 0 and i < xdim-1 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i, j+1]]))
                elif (i > 0 and i < xdim-1 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i+1,j], ice_edge[i, j-1]]))
                elif (i == 0 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j+1]]))
                elif (i == 0 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i+1,j], ice_edge[i, j-1]]))
                elif (i == xdim-1 and j == 0):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j+1]]))
                elif (i == xdim-1 and j == ydim-1):
                    nb_neighbors_sie = np.nansum(np.array([ice_edge[i-1,j], ice_edge[i, j-1]]))
                #
                nb_neighbors_sie = np.array(nb_neighbors_sie)
                if np.sum(nb_neighbors_sie) == 0:
                    length_sie[i,j] = np.sqrt(2) * spatial_resolution
                elif np.sum(nb_neighbors_sie) == 1:
                    length_sie[i,j] = 0.5 * (spatial_resolution + np.sqrt(2) * spatial_resolution)
                elif np.sum(nb_neighbors_sie) >= 2:
                    length_sie[i,j] = spatial_resolution
    #
    sie_length = np.sum(length_sie)
    return(sie_length)


# In[6]:


def calculate_ice_edge_length(date_obs, SIC_thresholds, paths):
    Ice_edge_lengths = {}
    #
    file_UNet = paths["UNet"] + "2021/01/Dataset_20210101.nc"
    nc_UNet = netCDF4.Dataset(file_UNet, "r")
    LSM = nc_UNet.variables["LSM"][:,:]
    nc_UNet.close()
    #
    xmin = 909
    xmax = 1453
    ymin = 1075
    ymax = 1555
    day_after = (datetime.datetime.strptime(date_obs, "%Y%m%d") + datetime.timedelta(days = 1)).strftime("%Y%m%d")
    file_AMSR2 = paths["AMSR2"] + date_obs[0:4] + "/" + date_obs[4:6] + "/" + "sic_cosi-5km_" + date_obs + "0000-" + day_after + "0000.nc"
    nc_AMSR2 = netCDF4.Dataset(file_AMSR2, "r")
    SIC = nc_AMSR2.variables["ice_conc"][0, ymin:ymax, xmin:xmax]
    nc_AMSR2.close()
    #
    for thresh_SIC in SIC_thresholds:
        SIE = np.zeros(np.shape(SIC))
        SIE[np.logical_and(SIC >= int(thresh_SIC), SIC <= 100)] = 1
        ice_edge = ice_edge_position(SIE, LSM)
        Ice_edge_lengths["SIC" + thresh_SIC] = length_sea_ice_edge(ice_edge, grid_resolution)
    #
    return(Ice_edge_lengths)


# In[7]:


def write_output(Ice_edge_lengths, date_obs, paths, SIC_thresholds):
    path_output = paths["output"] + date_obs[0:4] + "/" + date_obs[4:6] + "/"
    if os.path.isdir(path_output) == False:
        os.system("mkdir -p " + path_output)
    #
    filename_output = path_output + "Ice_edge_lengths_" + date_obs + ".h5"
    hf = h5py.File(filename_output, "w")
    for var in Ice_edge_lengths:
        print(var)
        output_var = "Ice_edge_lengths_" + var
        hf.create_dataset(output_var, data = Ice_edge_lengths[var])
    hf.close()


# In[8]:


t0 = time.time()
list_dates = make_list_dates(date_min, date_max)
date_obs = list_dates[$SGE_TASK_ID -1]
Ice_edge_lengths = calculate_ice_edge_length(date_obs, SIC_thresholds, paths)
write_output(Ice_edge_lengths, date_obs, paths, SIC_thresholds)
tf = time.time() - t0
print("Computing time", tf)
###############################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/AMSR2_ice_edge_lengths_""$SGE_TASK_ID"".py"
