#$ -N Training_data_AICE
#$ -l h_rt=00:20:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=4G,mem_free=4G,h_data=10G
#$ -q research-r8.q
#$ -t 1-10
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-08-2023

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/Training_data_AICE_""$SGE_TASK_ID"".py" << EOF
####################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[71]:


import os 
import scipy
import netCDF4
import numpy as np
import pyproj
import datetime 
import time


# Constants

# In[72]:
#
date_min = "20241222"
date_max = "20241231"
#
lead_time_max = 10
#
paths = {}
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Training/"
paths["ECMWF"] = "/lustre/storeB/project/copernicus/cosi/AICE/Data/ECMWF_daily_time_steps/"
paths["AMSR2"] = "/lustre/storeB/project/copernicus/cosi/WP2/SIC/v0.1/"
#
proj = {}
proj["ECMWF"] = "+proj=latlon"
proj["AMSR2"] = "+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0"
#
crs = {}
for var in proj:
    crs[var] = pyproj.CRS.from_proj4(proj[var])
#
variables = {}
variables["LSM"] = ["LSM"]
variables["ECMWF"] = ["U10M", "V10M", "T2M"]
variables["AMSR2"] = ["ice_conc", "total_standard_uncertainty"] 
#
Dates_AMSR2_missing_data = ["20151204", "20160415", "20170928", "20171125", "20181216", "20210203", "20210620", "20210816", "20211102", "20220324", "20220413", "20220418", "20220729", "20221122", "20230212", "20230301", "20230401", "20230424", "20230822", "20230828", "20230901", "20240417", "20240919", "20240920"]


# task_date function
# 
#     date_min: earliest forecast start date to process
#     date_max: latest forecast start date to process
#     task_ID: task ID when parallelizing (SGE_TASK_ID)

# In[73]:


def task_date(date_min, date_max, task_ID):
    current_date = datetime.datetime.strptime(date_min, '%Y%m%d')
    end_date = datetime.datetime.strptime(date_max, '%Y%m%d')
    list_date = []
    while current_date <= end_date:
        list_date.append(current_date.strftime('%Y%m%d'))
        current_date = current_date + datetime.timedelta(days = 1)
    date_task = list_date[task_ID - 1]
    return(date_task)


# In[74]:


def extract_domain_and_LSM(paths):
    xmin = 909
    xmax = 1453
    ymin = 1075
    ymax = 1555
    #
    Domain_data = {}
    #
    file_AMSR2 = "/lustre/storeB/project/copernicus/cosi/WP2/landmasks/LandOceanLakeMask_cosi-ease2-050.nc"
    nc = netCDF4.Dataset(file_AMSR2, "r")
    Domain_data["x"] = nc.variables["xc"][xmin:xmax] * 1000
    Domain_data["y"] = nc.variables["yc"][ymin:ymax] * 1000
    Domain_data["lat"] = nc.variables["lat"][ymin:ymax, xmin:xmax]
    Domain_data["lon"] = nc.variables["lon"][ymin:ymax, xmin:xmax]
    smask = nc.variables["smask"][ymin:ymax, xmin:xmax]
    nc.close()
    #
    LSM = np.zeros(np.shape(smask))
    LSM[smask == 0] = 1
    Domain_data["LSM"] = np.copy(LSM)
    #
    print(np.shape(LSM))
    #
    return(Domain_data)


# # Padding function (make_padding)
# 
#     x and y must be vectors (can be latitude / longitude if the data are on a regular grid)  
#     field must be either a 2D array (y, x) or a 3D array (time, y, x)

# In[75]:


def make_padding(x, y, field):
    dx = x[1] - x[0]
    x_extent = np.pad(x, (1, 1), constant_values = np.nan)    
    x_extent[0] = x_extent[1] - dx
    x_extent[-1] = x_extent[-2] + dx
    #
    dy = y[1] - y[0]
    y_extent = np.pad(y, (1, 1), constant_values = np.nan)
    y_extent[0] = y_extent[1] - dy
    y_extent[-1] = y_extent[-2] + dy
    #
    if field.ndim == 2:
        field_extent = np.pad(field, (1,1), constant_values = np.nan)
    elif field.ndim == 3:
        time_dim = len(field[:,0,0])
        field_extent = np.full((time_dim, len(y_extent), len(x_extent)), np.nan)
        #
        for t in range(0, time_dim):
            field_extent[t,:,:] = np.pad(field[t,:,:], (1,1), constant_values = np.nan)
    #
    return(x_extent, y_extent, field_extent)


# # Regridding functions (nearest_neighbor_indexes and nearest_neighbor_interp)
# 
#     xx_input and yy_input must be 2D arrays
#     x_output and y_output must be vectors  
#     field must be either a 2D array with dimensions (y, x) or a 3D array with dimensions (time, y, x) 
#     invalid_values = fill value to replace by 0. Land is therefore considered as open ocean.

# In[76]:


def nearest_neighbor_indexes(x_input, y_input, x_output, y_output):
    x_input = np.expand_dims(x_input, axis = 1)
    y_input = np.expand_dims(y_input, axis = 1)
    x_output = np.expand_dims(x_output, axis = 1)
    y_output = np.expand_dims(y_output, axis = 1)
    #
    coord_input = np.concatenate((x_input, y_input), axis = 1)
    coord_output = np.concatenate((x_output, y_output), axis = 1)
    #
    tree = scipy.spatial.KDTree(coord_input)
    dist, idx = tree.query(coord_output)
    #
    return(idx)


# In[77]:


def nearest_neighbor_interp(xx_input, yy_input, x_output, y_output, field, fill_value = None):
    xx_input_flat = np.ndarray.flatten(xx_input)
    yy_input_flat = np.ndarray.flatten(yy_input)
    #
    if fill_value is not None:
        if field.ndim == 2:
            idx_fill_value = np.ndarray.flatten(field) == fill_value
        elif field.ndim == 3:
            idx_fill_value = np.ndarray.flatten(field[0,:,:]) == fill_value
        #
        xx_input_flat = xx_input_flat[idx_fill_value == False]
        yy_input_flat = yy_input_flat[idx_fill_value == False]
    #
    xx_output, yy_output = np.meshgrid(x_output, y_output)
    xx_output_flat = np.ndarray.flatten(xx_output)
    yy_output_flat = np.ndarray.flatten(yy_output)
    #
    idx = nearest_neighbor_indexes(xx_input_flat, yy_input_flat, xx_output_flat, yy_output_flat)
    #
    if field.ndim == 2:
        field_flat = np.ndarray.flatten(field)
        if fill_value is not None:
            field_flat = field_flat[idx_fill_value == False]
        #
        field_interp = field_flat[idx]
        field_regrid = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #    
    elif field.ndim == 3:
        time_dim = len(field[:,0,0])
        field_regrid = np.full((time_dim, len(y_output), len(x_output)), np.nan)
        #
        for t in range(0, time_dim):
            field_flat = np.ndarray.flatten(field[t,:,:])
            if fill_value is not None:
                field_flat = field_flat[idx_fill_value == False]
            #
            field_interp = field_flat[idx]
            field_regrid[t,:,:] = np.reshape(field_interp, (len(y_output), len(x_output)), order = "C")
    #
    return(field_regrid)


# # Rotate wind function
#     x_wind, y_wind, lats, lons must be numpy arrays

# In[78]:


def rotate_wind(x_wind, y_wind, lats, lons, proj_str_from, proj_str_to):
    if np.shape(x_wind) != np.shape(y_wind):
        raise ValueError(f"x_wind {np.shape(x_wind)} and y_wind {np.shape(y_wind)} arrays must be the same size")
    if len(lats.shape) != 1:
        raise ValueError(f"lats {np.shape(lats)} must be 1D")
    if np.shape(lats) != np.shape(lons):
        raise ValueError(f"lats {np.shape(lats)} and lats {np.shape(lons)} must be the same size")
    if len(np.shape(x_wind)) == 1:
        if np.shape(x_wind) != np.shape(lats):
            raise ValueError(f"x_wind {len(x_wind)} and lats {len(lats)} arrays must be the same size")
    elif len(np.shape(x_wind)) == 2:
        if x_wind.shape[1] != len(lats):
            raise ValueError(f"Second dimension of x_wind {x_wind.shape[1]} must equal number of lats {len(lats)}")
    else:
        raise ValueError(f"x_wind {np.shape(x_wind)} must be 1D or 2D")
    #
    proj_from = pyproj.Proj(proj_str_from)
    proj_to = pyproj.Proj(proj_str_to)
    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)
    #
    orig_speed = np.sqrt(x_wind**2 + y_wind**2)
    #
    x0, y0 = proj_from(lons, lats)
    if proj_from.name != "longlat":
        x1 = x0 + x_wind
        y1 = y0 + y_wind
    else:
        factor = 3600000.0
        x1 = x0 + x_wind / factor / np.cos(lats * 3.14159265 / 180)
        y1 = y0 + y_wind / factor
    #
    X0, Y0 = transformer.transform(x0, y0)
    X1, Y1 = transformer.transform(x1, y1)
    #
    new_x_wind = X1 - X0
    new_y_wind = Y1 - Y0
    #
    if proj_to.name == "longlat":
        new_x_wind *= np.cos(lats * 3.14159265 / 180)
    #
    if proj_to.name == "longlat" or proj_from.name == "longlat":
        curr_speed = np.sqrt(new_x_wind**2 + new_y_wind**2)
        new_x_wind *= orig_speed / curr_speed
        new_y_wind *= orig_speed / curr_speed
    #
    return(new_x_wind, new_y_wind)


# # Read_netCDF functions
#     filename: filename including the path
#     variables: list of variables (excluding time, x, y, lat, lon) to extract (list of strings)
#     paths: dictionary defined in the Constants section

# In[79]:


def read_netCDF(filename, variables, paths = paths):
    Dataset = {}
    nc = netCDF4.Dataset(filename, "r")
    Dataset["time"] = nc.variables["time"][:]
    #
    if paths["ECMWF"] in filename:
        Dataset["lat"] = nc.variables["lat"][:]
        Dataset["lon"] = nc.variables["lon"][:]
    #
    elif paths["AMSR2"] in filename:
        xmin = 909
        xmax = 1453
        ymin = 1075
        ymax = 1555
        Dataset["x"] = nc.variables["xc"][xmin:xmax] * 1000
        Dataset["y"] = nc.variables["yc"][ymin:ymax] * 1000
        Dataset["lat"] = nc.variables["lat"][ymin:ymax, xmin:xmax] 
        Dataset["lon"] = nc.variables["lon"][ymin:ymax, xmin:xmax]
    #
    for var in variables:
        vardim = nc.variables[var].ndim
        if vardim == 1:
            Dataset[var] = nc.variables[var][:]
        elif vardim == 2:
            Dataset[var] = nc.variables[var][:,:]
        elif vardim == 3:
            Dataset[var] = nc.variables[var][:,:,:]
            if paths["AMSR2"] in filename:
                xmin = 909
                xmax = 1453
                ymin = 1075
                ymax = 1555
                Dataset[var] = Dataset[var][:, ymin:ymax, xmin:xmax]
                #
                filename_date = os.path.basename(filename)[13:21]
                if filename_date in Dates_AMSR2_missing_data:
                    Dataset[var] = np.full((1, 480, 544), np.nan)
                else:
                    if ("ice_conc" in var) or ("total_standard_uncertainty" in var):
                        idx_invalid_values = Dataset[var] < 0
                        Dataset[var][idx_invalid_values == True] = -32767
        else:
            print("ERROR. Number of dimensions higher than 3.")
    nc.close()
    #
    return(Dataset)


# # extract_ECMWF_data function
# 
#     filename: filename (including path) containing ECMWF data 
#     ndays: maximum lead time in days
#     TOPAZ: TOPAZ dataset (dictionary)   
#     proj: dictionary of proj4 strings
#     variables: list of variables to extract (variables["ECMWF"])

# In[ ]:


def extract_ECMWF_data(filename, ndays, domain, proj = proj, variables = variables["ECMWF"], crs = crs):
    Data_ECMWFgrid = read_netCDF(filename, variables)
    Data_AMSR2grid = {}
    transform_ECMWF_to_AMSR2 = pyproj.Transformer.from_crs(crs["ECMWF"], crs["AMSR2"], always_xy = True)
    lons, lats = np.meshgrid(Data_ECMWFgrid["lon"], Data_ECMWFgrid["lat"])
    xx_ECMWF_AMSR2proj, yy_ECMWF_AMSR2proj = transform_ECMWF_to_AMSR2.transform(lons, lats)
    #
    if ("U10M" in Data_ECMWFgrid) and ("V10M" in Data_ECMWFgrid):
        x_wind = np.full((lead_time_max, len(Data_ECMWFgrid["lat"]), len(Data_ECMWFgrid["lon"])), np.nan)
        y_wind = np.full((lead_time_max, len(Data_ECMWFgrid["lat"]), len(Data_ECMWFgrid["lon"])), np.nan)
        #
        for ts in range(0, lead_time_max):
            x_wind_rot, y_wind_rot = rotate_wind(np.ndarray.flatten(Data_ECMWFgrid["U10M"][ts,:,:]), 
                                                 np.ndarray.flatten(Data_ECMWFgrid["V10M"][ts,:,:]),
                                                 np.ndarray.flatten(lats), 
                                                 np.ndarray.flatten(lons), 
                                                 proj["ECMWF"], 
                                                 proj["AMSR2"]
                                                )
            #
            x_wind[ts,:,:] = np.reshape(x_wind_rot, (len(Data_ECMWFgrid["lat"]), len(Data_ECMWFgrid["lon"])), order = "C")
            y_wind[ts,:,:] = np.reshape(y_wind_rot, (len(Data_ECMWFgrid["lat"]), len(Data_ECMWFgrid["lon"])), order = "C")            
            #
        Data_ECMWFgrid["wind_x"] = np.copy(x_wind)
        Data_ECMWFgrid["wind_y"] = np.copy(y_wind)
        Data_ECMWFgrid.pop("U10M")
        Data_ECMWFgrid.pop("V10M")
    #
    Cum_data_ECMWFgrid = {}
    for var in Data_ECMWFgrid:
        if var in ["wind_x", "wind_y", "T2M"]:
            var_cum = np.full((lead_time_max, len(Data_ECMWFgrid["lat"]), len(Data_ECMWFgrid["lon"])), np.nan)
            for ts in range(0, lead_time_max):
                var_cum[ts,:,:] = np.nanmean(Data_ECMWFgrid[var][0:ts+1,:,:], axis = 0)
            #
            Cum_data_ECMWFgrid[var + "_cum"] = np.copy(var_cum)
            Cum_data_ECMWFgrid[var + "_cum"][np.isnan(var_cum) == True] = -32767
    #
    for var in Cum_data_ECMWFgrid:
        Data_AMSR2grid[var] = nearest_neighbor_interp(xx_ECMWF_AMSR2proj, yy_ECMWF_AMSR2proj, domain["x"], domain["y"], Cum_data_ECMWFgrid[var], fill_value = -32767)
    #
    return(Data_AMSR2grid)


# # extract_SIC_obs_predictors function
#     date_task: forecast start date (string "YYYYMMDD")
#     domain: domain data (Datasets["domain"]) 
#     trend_period: Number of days to take into account for calculating the trend
#     proj: dictionary of proj4 strings
#     crs: crs defined in "Constants"

# In[81]:


def extract_SIC_obs_predictors(date_task, domain, proj = proj, crs = crs):
    Data_AMSR2 = {}
    #
    previous_date = datetime.datetime.strptime(date_task, "%Y%m%d") - datetime.timedelta(days = 1)
    previous_date_str = previous_date.strftime("%Y%m%d")
    filename_SIC = paths["AMSR2"] + previous_date_str[0:4] + "/" + previous_date_str[4:6] + "/" + "sic_cosi-5km_" + previous_date_str + "0000-" + date_task + "0000.nc"  
    if os.path.isfile(filename_SIC):
        if previous_date_str in Dates_AMSR2_missing_data:
            pass
        else:
            domain_lat = np.expand_dims(domain["lat"], axis = 0)
            SIC_data = read_netCDF(filename_SIC, variables = ["ice_conc"])
            SIC_data["ice_conc"][domain_lat > 89.1] = -32767
            xx_SIC, yy_SIC = np.meshgrid(SIC_data["x"], SIC_data["y"])
            SIC = nearest_neighbor_interp(xx_SIC, yy_SIC, domain["x"], domain["y"], SIC_data["ice_conc"], fill_value = -32767)
            LSM = np.expand_dims(domain["LSM"], axis = 0)
            SIC[LSM == 0] = 0            
            Data_AMSR2["ice_conc"] = np.squeeze(SIC)
    #
    return(Data_AMSR2)     


# # extract_targets
#     date_task: forecast start date (string "YYYYMMDD")
#     ndays: Number of days to take into account for calculating the trend
#     domain: domain data (Datasets["domain"])
#     proj: dictionary of proj4 strings
#     variables: list of variables to extract 
#     crs: crs defined in "Constants"

# In[82]:


def extract_targets(date_task, ndays, domain, paths = paths):
    Data_AMSR2 = {}
    LSM_extend = np.expand_dims(domain["LSM"], axis = 0)
    #
    for lt in range(0, ndays):
        date_str = (datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt)).strftime("%Y%m%d")
        date_day_after_str = (datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt + 1)).strftime("%Y%m%d%H%M")
        filename_SIC = paths["AMSR2"] + date_str[0:4] + "/" + date_str[4:6] + "/" + "sic_cosi-5km_" + date_str + "0000-" + date_day_after_str + ".nc"
        #
        SIC_obs = read_netCDF(filename_SIC, variables = ["ice_conc", "total_standard_uncertainty"])
        if lt == 0:
            xx_SIC, yy_SIC = np.meshgrid(SIC_obs["x"], SIC_obs["y"])
        #
        domain_lat = np.expand_dims(domain["lat"], axis = 0)
        SIC_obs["ice_conc"] = SIC_obs["ice_conc"]
        SIC_obs["total_standard_uncertainty"] = SIC_obs["total_standard_uncertainty"]
        SIC_obs["ice_conc"][domain_lat > 89.1] = -32767
        SIC_obs["total_standard_uncertainty"][domain_lat > 89.1] = -32767
        SIC = nearest_neighbor_interp(xx_SIC, yy_SIC, domain["x"], domain["y"], SIC_obs["ice_conc"], fill_value = -32767)
        total_uncertainty = nearest_neighbor_interp(xx_SIC, yy_SIC, domain["x"], domain["y"], SIC_obs["total_standard_uncertainty"], fill_value = -32767)
        #
        SIC[LSM_extend == 0] = 0
        total_uncertainty[LSM_extend == 0] = 0
        #
        if lt == 0:
            Data_AMSR2["SIC"] = np.copy(SIC)
            Data_AMSR2["SIC_total_standard_uncertainty"] = np.copy(total_uncertainty)
        else:
            Data_AMSR2["SIC"] = np.concatenate((Data_AMSR2["SIC"], SIC), axis = 0)
            Data_AMSR2["SIC_total_standard_uncertainty"] = np.concatenate((Data_AMSR2["SIC_total_standard_uncertainty"], total_uncertainty), axis = 0)
    #
    return(Data_AMSR2)


# # write_netCDF function
#     date_task: forecast start date (string "YYYYMMDD")
#     Datasets: Dictionary containing all variables that we want to extract
#     paths: paths defined in the Constants section
#     trend_period: Number of days to take into account for calculating the trend

# In[83]:


def write_netCDF(date_task, Datasets, paths):
    Outputs = vars()
    #
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/"
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "Dataset_" + date_task + ".nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    output_netcdf = netCDF4.Dataset(output_filename, 'w', format = 'NETCDF4')
    #
    dimensions = ["time", "x", "y"]
    for di in dimensions:
        if di == "time":
            Outputs[di] = output_netcdf.createDimension(di, lead_time_max)
        else:
            Outputs[di] = output_netcdf.createDimension(di, len(Datasets["domain"][di]))
    #
    dim_variables = dimensions + ["lat", "lon"]
    for dv in dim_variables:
        if dv == "time":
            time_vect = []
            for lt in range(0, lead_time_max):
                time_vect.append((datetime.datetime.strptime(date_task, "%Y%m%d") + datetime.timedelta(days = lt)).strftime("%Y%m%d"))
            Outputs[dv] = output_netcdf.createVariable(dv, "d", (dv))
            Outputs[dv][:] = time_vect
            Outputs[dv].standard_name = "forecast time"
            Outputs[dv].units = "forecasted_date"
        else: 
            if Datasets["domain"][dv].ndim == 1:
                    Outputs[dv] = output_netcdf.createVariable(dv, "d", (dv))
                    Outputs[dv][:] = Datasets["domain"][dv]
                    if dv == "x" or dv == "y":
                        Outputs[dv].standard_name = "projection_" + dv + "_coordinate"
                        Outputs[dv].units = "m"
            elif Datasets["domain"][dv].ndim == 2:
                Outputs[dv] = output_netcdf.createVariable(dv, "d", ("y", "x"))
                Outputs[dv][:,:] = Datasets["domain"][dv]
                if dv == "lat":
                    Outputs[dv].standard_name = "latitude"
                elif dv == "lon":
                    Outputs[dv].standard_name = "longitude"
                Outputs[dv].units = "degrees"
    #
    SIC_variables = ["ice_conc", "fice", "SIC"]
    for ds in Datasets:
        for var in Datasets[ds]:
            if (var in dim_variables) == False:
                if var == "LSM":
                    var_name = "LSM"
                elif var in SIC_variables:
                    var_name = ds + "_SIC"
                else:
                    var_name = ds + "_" + var
                #
                if Datasets[ds][var].ndim == 2:
                    Outputs[var_name] = output_netcdf.createVariable(var_name, "d", ("y", "x"))
                    Outputs[var_name][:,:] = np.round(Datasets[ds][var], 3)
                elif Datasets[ds][var].ndim == 3:
                    Outputs[var_name] = output_netcdf.createVariable(var_name, "d", ("time", "y", "x"))
                    Outputs[var_name][:,:,:] = np.round(Datasets[ds][var], 3)
                #
                if var in SIC_variables:
                    if ds == "TARGET_AMSR2":
                        Outputs[var_name].standard_name = "AMSR2 sea ice concentration"
                    elif ds == "SICobs_AMSR2":
                        Outputs[var_name].standard_name = "Sea ice concentration from AMSR2 during the day preceding the forecast start date"
                    else:
                        Outputs[var_name].standard_name = ds + " sea ice concentration"
                    Outputs[var_name].units = "%"
                if var == "SIC_total_standard_uncertainty":
                    Outputs[var_name].standard_name = "Total uncertainty (one standard deviation) of concentration of sea ice"
                    Outputs[var_name].units = "%"
                elif var == "LSM":
                    Outputs[var_name].standard_name = "Land sea mask"
                    Outputs[var_name].units = "1: ocean, 0: land"
                elif ds + "_" + var == "ECMWF_T2M_cum":
                    Outputs[var_name].standard_name = "ECMWF 2 metre temperature"
                    Outputs[var_name].units = "K"
                elif ds + "_" + var == "ECMWF_wind_x_cum":
                    Outputs[var_name].standard_name = "Mean ECMWF wind in the x direction since the forecast start date"
                    Outputs[var_name].units = "m/s"
                elif ds + "_" + var == "ECMWF_wind_y_cum":
                    Outputs[var_name].standard_name = "Mean ECMWF wind in the y direction since the forecast start date"
                    Outputs[var_name].units = "m/s"
    output_netcdf.close()    


# # Data processing 

# In[84]:


t0 = time.time()
#
date_task = task_date(date_min, date_max, task_ID = $SGE_TASK_ID)
previous_day = (datetime.datetime.strptime(date_task, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
print("date_task", date_task)
if previous_day in Dates_AMSR2_missing_data:
    print("Missing AMSR2 observations during the day preceding the forecast start date")
else:
    filename_ECMWF = paths["ECMWF"] + date_task[0:4] + "/" + date_task[4:6] + "/ECMWF_operational_forecasts_daily_time_steps_T2m_10mwind_" + date_task + ".nc"
    #
    Datasets = {}
    #
    Datasets["domain"] = extract_domain_and_LSM(paths)
    #
    Datasets["ECMWF"] = extract_ECMWF_data(filename = filename_ECMWF, 
                                          ndays = lead_time_max, 
                                          domain = Datasets["domain"], 
                                          variables = variables["ECMWF"], 
                                          crs = crs)
    #
    Datasets["SICobs_AMSR2"] = extract_SIC_obs_predictors(date_task = date_task, 
                                                          domain = Datasets["domain"],
                                                          proj = proj, 
                                                          crs = crs)    
    #
    Datasets["TARGET_AMSR2"] = extract_targets(date_task = date_task, 
                                               ndays = lead_time_max, 
                                               domain = Datasets["domain"], 
                                               paths = paths
                                               )
    #
    write_netCDF(date_task = date_task, 
                 Datasets = Datasets, 
                 paths = paths, 
                 )    
    #
    tf = time.time() - t0
    print("Computing time: ", tf)
###################################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/Training_data_AICE_""$SGE_TASK_ID"".py"
