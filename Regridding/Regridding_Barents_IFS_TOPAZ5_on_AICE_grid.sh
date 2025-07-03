#$ -N Models_on_AICE_grid
#$ -l h_rt=00:05:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=2G,mem_free=2G,h_data=2G
#$ -q research-r8.q
#$ -t 1-1188
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y


source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-08-2024

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/Models_on_AICE_grid_""$SGE_TASK_ID"".py" << EOF
###################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[45]:


import time
import os
import netCDF4
import scipy
import pyproj
import datetime
import numpy as np
import matplotlib.pyplot as plt


# # Constants

# In[46]:
#
date_min = "20220101"
date_max = "20250331"
#
paths = {}
paths["AICE_op_forecasts"] = "/lustre/storeB/project/fou/hi/oper/aice/archive/"
paths["AICE_reforecasts"] = "/lustre/storeB/project/copernicus/cosi/AICE/Predictions/AICE_v1_reforecasts/"
paths["IFS"] = "/lustre/storeB/project/copernicus/cosi/AICE/Data/ECMWF_daily_time_steps/"
paths["TOPAZ5"] = "/lustre/storeB/project/copernicus/sea/metnotopaz5/arctic/mersea-class1/"
paths["Barents"] = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/surface/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/AICE/Data/Models_on_AICE_grid/"


# # List dates

# In[47]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return list_dates


# # Load datasets

# In[ ]:


class read_datasets():
    def __init__(self, date_task, paths):
        self.date_task = date_task
        self.paths = paths

    def read_AICE(self):
        Dataset = {}
        if datetime.datetime.strptime(self.date_task, "%Y%m%d") >= datetime.datetime.strptime("20240401", "%Y%m%d"):
            filename = self.paths["AICE_op_forecasts"] + "AICE_forecasts_" + self.date_task + "T000000Z.nc"
        else:
            filename = self.paths["AICE_reforecasts"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + "AICE_forecasts_" + self.date_task + "T000000Z.nc"
        if os.path.isfile(filename) == True:
            with netCDF4.Dataset(filename, "r") as nc:
                for var in nc.variables:
                    if var == "Lambert_Azimuthal_Grid":
                        Dataset["proj4"] = nc.variables[var].proj4_string
                    else:
                        Dataset[var] = nc.variables[var][:]
            Dataset["sea_mask"] = np.ones(np.shape(Dataset["lat"]))
            Dataset["sea_mask"][np.isnan(Dataset["SIC"][0,:,:]) == True] = 0
        return Dataset
    
    def read_IFS(self):
        Dataset = {}
        filename = self.paths["IFS"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + "ECMWF_operational_forecasts_daily_time_steps_SIC_" + self.date_task + ".nc"
        if os.path.isfile(filename) == True:
            with netCDF4.Dataset(filename, "r") as nc:
                for var in nc.variables:
                    if var == "CI":
                        Dataset["SIC"] = nc.variables[var][:] * 100
                    else:
                        Dataset[var] = nc.variables[var][:]
            Dataset["proj4"] = "+proj=latlon"
            
            filename_land_sea_mask = self.paths["IFS"] + "ECMWF_operational_forecasts_Land_Sea_Mask.nc"
            with netCDF4.Dataset(filename_land_sea_mask, "r") as nc:
                LSM = np.squeeze(nc.variables["LSM"][:])
                Dataset["sea_mask"] = np.zeros(np.shape(LSM))
                Dataset["sea_mask"][LSM == 0] = 1

        return Dataset

    def read_TOPAZ5(self):
        Dataset = {}
        max_lead_time = 10
        if datetime.datetime.strptime(self.date_task, "%Y%m%d") >= datetime.datetime.strptime("20230901", "%Y%m%d"):
            for lt in range(0, max_lead_time):
                forecast_date = (datetime.datetime.strptime(self.date_task, "%Y%m%d") + datetime.timedelta(days = lt)).strftime("%Y%m%d")
                filename = self.paths["TOPAZ5"] + forecast_date[0:4] + "/" + forecast_date[4:6] + "/" + forecast_date + "_dm-metno-MODEL-topaz5-ARC-b" + self.date_task + "-fv02.0.nc"
                if os.path.isfile(filename) == True:
                    with netCDF4.Dataset(filename, "r") as nc:
                        if lt == 0:
                            Dataset["proj4"] = nc.variables["stereographic"].proj4
                            Dataset["x"] = nc.variables["x"][:] * 100 * 1000
                            Dataset["y"] = nc.variables["y"][:] * 100 * 1000
                            Dataset["lat"] = nc.variables["latitude"][:,:]
                            Dataset["lon"] = nc.variables["longitude"][:,:]
                            Dataset["SIC"] = nc.variables["siconc"][:,:,:] * 100
                            Dataset["sea_mask"] = np.ones(np.shape(Dataset["lat"]))
                            Dataset["sea_mask"][np.squeeze(Dataset["SIC"].mask) == True] = 0
                            Dataset["SIC"][np.expand_dims(Dataset["sea_mask"] == 0, axis = 0)] = np.nan
                        else:
                            SIC = nc.variables["siconc"][:,:,:] * 100
                            SIC[np.expand_dims(Dataset["sea_mask"] == 0, axis = 0)] = np.nan
                            Dataset["SIC"] = np.concatenate((Dataset["SIC"], SIC), axis = 0)
                else:
                    if lt > 0:
                        SIC_nan = np.expand_dims(np.full(np.shape(Dataset["lat"]), np.nan), axis = 0)
                        Dataset["SIC"] = np.concatenate((Dataset["SIC"], SIC_nan), axis = 0)
                    else:
                        return Dataset
        return Dataset
                
    def read_Barents(self):
        Dataset = {}
        prod_time = "T00Z"
        N_Barents_members = 6

        if datetime.datetime.strptime(self.date_task, "%Y%m%d") >= datetime.datetime.strptime("20231206", "%Y%m%d"):
            path_data = self.paths["Barents"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + self.date_task[6:8] + "/" + prod_time + "/"
            for em in range(0, N_Barents_members):
                member = "{:02d}".format(em)
                filename = path_data + "barents_sfc_" + self.date_task + prod_time + "m" + member + ".nc"
                if os.path.isfile(filename) == True:
                    with netCDF4.Dataset(filename, "r") as nc:
                        if member == "00":
                            Dataset["proj4"] = nc.variables["projection_lambert"].proj4
                            Dataset["x"] = nc.variables["X"][:]    
                            Dataset["y"] = nc.variables["Y"][:]  
                            Dataset["sea_mask"] = nc.variables["sea_mask"][:]

                        Dataset["Member" + member + "_SIC"] = np.full((4, len(Dataset["y"]), len(Dataset["x"])), np.nan) 
                        for ts in range(0, 4):
                            ts_start = ts * 24
                            ts_end = ts_start + 24
                            Dataset["Member" + member + "_SIC"][ts,:,:] = np.nanmean(nc.variables["ice_concentration"][ts_start:ts_end,:,:], axis = 0) * 100
        return Dataset
                    
    def __call__(self):
        All_datasets = {}
        All_datasets["AICE"] = self.read_AICE()
        All_datasets["IFS"] = self.read_IFS()
        All_datasets["TOPAZ5"] = self.read_TOPAZ5()
        All_datasets["Barents"] = self.read_Barents()
        return All_datasets


#  # Regridding

# In[49]:


class regridding():
    def __init__(self, date_task, Model_data):
        self.date_task = date_task
        self.Model_data = Model_data
    
    def make_padding(self, x, y, field):
        dx = x[1] - x[0]
        x_extent = np.pad(x, (1, 1), constant_values = np.nan)    
        x_extent[0] = x_extent[1] - dx
        x_extent[-1] = x_extent[-2] + dx
        
        dy = y[1] - y[0]
        y_extent = np.pad(y, (1, 1), constant_values = np.nan)
        y_extent[0] = y_extent[1] - dy
        y_extent[-1] = y_extent[-2] + dy
        
        if field.ndim == 2:
            field_extent = np.pad(field, (1,1), constant_values = np.nan)
        elif field.ndim == 3:
            time_dim = len(field[:,0,0])
            field_extent = np.full((time_dim, len(y_extent), len(x_extent)), np.nan)
            
            for t in range(0, time_dim):
                field_extent[t,:,:] = np.pad(field[t,:,:], (1,1), constant_values = np.nan)
        
        return x_extent, y_extent, field_extent
    
    def nearest_neighbor_indexes(self, x_input, y_input, x_output, y_output):
        x_input = np.expand_dims(x_input, axis = 1)
        y_input = np.expand_dims(y_input, axis = 1)
        x_output = np.expand_dims(x_output, axis = 1)
        y_output = np.expand_dims(y_output, axis = 1)
        
        coord_input = np.concatenate((x_input, y_input), axis = 1)
        coord_output = np.concatenate((x_output, y_output), axis = 1)
        
        tree = scipy.spatial.KDTree(coord_input)
        dist, idx = tree.query(coord_output)
        return idx
    
    def extract_idx(self, lat, lon):
        transform = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4("+proj=latlon"), pyproj.CRS.from_proj4(self.Model_data["AICE"]["proj4"]), always_xy = True)
        xx_input, yy_input = transform.transform(lon, lat)   
        
        xx_input = np.ndarray.flatten(xx_input)
        yy_input = np.ndarray.flatten(yy_input)
        
        xx_output, yy_output = np.meshgrid(self.Model_data["AICE"]["x"], self.Model_data["AICE"]["y"])
        xx_output = np.ndarray.flatten(xx_output)
        yy_output = np.ndarray.flatten(yy_output)
        
        idx = self.nearest_neighbor_indexes(xx_input, yy_input, xx_output, yy_output)
        return idx
    
    def nearest_neighbor_interpolation(self):
        Interpolated_datasets = {}
        
        for model in self.Model_data:
            Interpolated_datasets[model] = {}
            
            if model == "AICE":
                for var in self.Model_data[model]:
                    Interpolated_datasets[model][var] = np.copy(self.Model_data[model][var])
            else:
                if len(self.Model_data[model]) > 0:                   
                    for var in self.Model_data[model]:
                        if (var == "sea_mask") or ("SIC" in var):
                            if "x" in self.Model_data[model]:
                                x_pad, y_pad, var_pad = self.make_padding(self.Model_data[model]["x"], self.Model_data[model]["y"], self.Model_data[model][var])
                                xx_pad, yy_pad = np.meshgrid(x_pad, y_pad)
                                transform = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4(self.Model_data[model]["proj4"]), pyproj.CRS.from_proj4("+proj=latlon"), always_xy = True)
                                lon_pad, lat_pad = transform.transform(xx_pad, yy_pad)
                            else:
                                if np.max(self.Model_data[model]["lat"]) == 90:
                                    self.Model_data[model]["lat"][self.Model_data[model]["lat"] == 90] = 89.999999999  # In order to avoid interpolating nan (see 2 lines below)
                                lon_1D_pad, lat_1D_pad, var_pad = self.make_padding(self.Model_data[model]["lon"], self.Model_data[model]["lat"], self.Model_data[model][var])
                                lat_1D_pad[lat_1D_pad > 90] = 90  # In order to avoid latitudes > 90 after padding
                                lon_pad, lat_pad = np.meshgrid(lon_1D_pad, lat_1D_pad)
                            idx = self.extract_idx(lat_pad, lon_pad)

                            if var == "sea_mask":
                                field_flat = np.ndarray.flatten(var_pad)
                                field_interp = field_flat[idx]
                                field_regrid = np.reshape(field_interp, (len(self.Model_data["AICE"]["y"]), len(self.Model_data["AICE"]["x"])), order = "C")
                            elif "SIC" in var:
                                time_dim = len(self.Model_data[model][var][:,0,0])
                                field_regrid = np.full((time_dim, len(self.Model_data["AICE"]["y"]), len(self.Model_data["AICE"]["x"])), np.nan)
                                for t in range(0, time_dim):
                                    field_flat = np.ndarray.flatten(var_pad[t,:,:])
                                    field_interp = field_flat[idx]
                                    field_regrid[t,:,:] = np.reshape(field_interp, (len(self.Model_data["AICE"]["y"]), len(self.Model_data["AICE"]["x"])), order = "C")
                            
                            Interpolated_datasets[model][var] = np.copy(field_regrid)
        return Interpolated_datasets
    #
    def __call__(self):
        Interpolated_datasets = self.nearest_neighbor_interpolation()
        return Interpolated_datasets


# # Write netCDF output

# In[50]:


class write_netCDF():
    def __init__(self, Interpolated_datasets, date_task, paths):
        self.Interpolated_datasets = Interpolated_datasets
        self.date_task = date_task
        self.paths = paths
    
    def make_common_sea_mask(self):
        Common_sea_mask = np.ones(np.shape(self.Interpolated_datasets["AICE"]["sea_mask"]))
        Common_domain_mask = np.ones(np.shape(self.Interpolated_datasets["AICE"]["sea_mask"]))
        
        for model in self.Interpolated_datasets:
            if len(self.Interpolated_datasets[model]) > 0:
                sea_mask = self.Interpolated_datasets[model]["sea_mask"]
                Common_sea_mask[sea_mask < 1] = 0
                
                if "SIC" in self.Interpolated_datasets[model]:
                    SIC_t0 = self.Interpolated_datasets[model]["SIC"][0,:,:]
                    Common_domain_mask[np.isnan(SIC_t0) == True] = 0
                if "Member00_SIC" in self.Interpolated_datasets[model]:
                    SIC_t0 = self.Interpolated_datasets[model]["Member00_SIC"][0,:,:]
                    Common_domain_mask[np.isnan(SIC_t0) == True] = 0    
        
        Common_domain_mask[Common_sea_mask == 0] = 0
        return Common_sea_mask, Common_domain_mask
    
    def expand_grid(self, model, Sea_mask, var):
        time_dim_model = np.shape(self.Interpolated_datasets[model][var])[0]
        time_dim = len(self.Interpolated_datasets["AICE"]["time"])
        y_dim = len(self.Interpolated_datasets["AICE"]["y"])
        x_dim = len(self.Interpolated_datasets["AICE"]["x"])
        Sea_mask_extend = np.repeat(np.expand_dims(Sea_mask, axis = 0), time_dim, axis = 0)
        SIC = np.full((time_dim, y_dim, x_dim), np.nan)
        SIC[0:time_dim_model,:,:] = self.Interpolated_datasets[model][var]
        SIC[Sea_mask_extend < 1] = np.nan
        return SIC

    def write_output_file(self, Sea_mask, Domain_mask):
        path_output_date = self.paths["output"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/"
        if os.path.exists(path_output_date) == False:
            os.system("mkdir -p " + path_output_date)
        output_filename = path_output_date + "Models_SIC_" + self.date_task + "_on_AICE_grid.nc"
        if os.path.isfile(output_filename):
            os.system("rm " + output_filename)
        
        with netCDF4.Dataset(str(output_filename), "w", format = "NETCDF4") as output_netcdf:
            Lambert_Azimuthal_Grid = output_netcdf.createDimension("proj4", 1)
            time = output_netcdf.createDimension("time", len(self.Interpolated_datasets["AICE"]["time"]))
            x = output_netcdf.createDimension("x", len(self.Interpolated_datasets["AICE"]["x"]))
            y = output_netcdf.createDimension("y", len(self.Interpolated_datasets["AICE"]["y"]))
            
            Outputs = vars()
            
            Outputs["Lambert_Azimuthal_Grid"] = output_netcdf.createVariable("Lambert_Azimuthal_Grid", "d")
            Outputs["Lambert_Azimuthal_Grid"].grid_mapping_name = "lambert_azimuthal_equal_area"
            Outputs["Lambert_Azimuthal_Grid"].semi_major_axis = 6378137
            Outputs["Lambert_Azimuthal_Grid"].semi_minor_axis = 6356752.31424518
            Outputs["Lambert_Azimuthal_Grid"].reference_ellipsoid_name = "WGS 84"
            Outputs["Lambert_Azimuthal_Grid"].longitude_of_prime_meridian = "0.0"
            Outputs["Lambert_Azimuthal_Grid"].prime_meridian_name = "Greenwich"
            Outputs["Lambert_Azimuthal_Grid"].geographic_crs_name = "unknown"
            Outputs["Lambert_Azimuthal_Grid"].horizontal_datum_name = "Unknown based on WGS84 ellipsoid"
            Outputs["Lambert_Azimuthal_Grid"].projected_crs_name = "unknown"
            Outputs["Lambert_Azimuthal_Grid"].latitude_of_projection_origin = 90.0
            Outputs["Lambert_Azimuthal_Grid"].longitude_of_projection_origin = 0.0
            Outputs["Lambert_Azimuthal_Grid"].false_easting = 0.0
            Outputs["Lambert_Azimuthal_Grid"].false_northing = 0.0
            Outputs["Lambert_Azimuthal_Grid"].proj4_string = "+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0"
            
            Outputs["time"] = output_netcdf.createVariable("time", "d", ("time"))
            Outputs["time"].units = "seconds since 1970-01-01 00:00:00 +0000"
            Outputs["time"].standard_name = "time"
            Outputs["time"].long_name = "time"
            Outputs["time"][:] = np.copy(self.Interpolated_datasets["AICE"]["time"])
            
            Outputs["x"] = output_netcdf.createVariable("x", "d", ("x"))
            Outputs["x"].units = "m"
            Outputs["x"].standard_name = "projection_x_coordinate"
            Outputs["x"].long_name = "projection_x_coordinate"
            Outputs["x"][:] = np.copy(self.Interpolated_datasets["AICE"]["x"])
            
            Outputs["y"] = output_netcdf.createVariable("y", "d", ("y"))
            Outputs["y"].units = "m"
            Outputs["y"].standard_name = "projection_y_coordinate"
            Outputs["y"].long_name = "projection_y_coordinate"
            Outputs["y"][:] = np.copy(self.Interpolated_datasets["AICE"]["y"])
            
            Outputs["sea_mask"] = output_netcdf.createVariable("sea_mask", "d", ("y", "x"))
            Outputs["sea_mask"].units = "fraction of sea"
            Outputs["sea_mask"].standard_name = "sea mask"
            Outputs["sea_mask"].long_name = "sea land mask (0: land, 1: sea)"
            Outputs["sea_mask"][:,:] = np.copy(Sea_mask)
            
            Outputs["domain_mask"] = output_netcdf.createVariable("domain_mask", "d", ("y", "x"))
            Outputs["domain_mask"].units = "fraction of sea"
            Outputs["domain_mask"].standard_name = "domain mask"
            Outputs["domain_mask"].long_name = "Shared domain between all models (0: outside of the shared domain, 1: shared domain)"
            Outputs["domain_mask"][:,:] = np.copy(Domain_mask)
            
            for model in self.Interpolated_datasets:
                for var in self.Interpolated_datasets[model]:
                    if "SIC" in var:
                        var_output = model + "_" + var
                        member = var.replace("SIC", "").replace("_", "")
                        Outputs[var_output] = output_netcdf.createVariable(var_output, "d", ("time", "y", "x"))
                        Outputs[var_output].units = "%"
                        Outputs[var_output].standard_name = model + " " + member + " sea_ice_area_fraction"
                        Outputs[var_output].long_name = model + " " + member + " sea ice concentration"
                        Outputs[var_output][:,:,:] = self.expand_grid(model, Sea_mask, var)
    #
    def __call__(self):
        Sea_mask, Domain_mask = self.make_common_sea_mask()
        self.write_output_file(Sea_mask, Domain_mask)


# # Main

# In[51]:


t0 = time.time()

list_dates = make_list_dates(date_min = date_min, date_max = date_max)
date_task = list_dates[$SGE_TASK_ID - 1]

Datasets = read_datasets(date_task = date_task, paths = paths)()
Interpolated_datasets = regridding(date_task = date_task, Model_data = Datasets)()
write_netCDF(Interpolated_datasets = Interpolated_datasets, date_task = date_task, paths = paths)()

tf = time.time()
print("Computing time: ", tf - t0)
###################################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/Models_on_AICE_grid_""$SGE_TASK_ID"".py"
