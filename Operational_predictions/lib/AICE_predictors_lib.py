# -*- coding: utf-8 -*-
import os
import sys
import h5py
import glob
import scipy
import pyproj
import netCDF4
import datetime
import numpy as np
import sklearn.linear_model
from scipy.ndimage import distance_transform_cdt

###################################################################################################
# # Regridding functions (nearest_neighbor_indexes and nearest_neighbor_interp)
#     xx_input and yy_input must be 2D arrays
#     x_output and y_output must be vectors  
#     field must be either a 2D array with dimensions (y, x) or a 3D array with dimensions (time, y, x) 
#     invalid_values = fill value to replace by 0. Land is therefore considered as open ocean.


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
#
def nearest_neighbor_interp(xx_input, yy_input, x_output, y_output, field, fill_value = None):
    xx_input_flat = np.ndarray.flatten(xx_input)
    yy_input_flat = np.ndarray.flatten(yy_input)
    #
    if fill_value is not None:
        if field.ndim == 2:
            idx_fill_value = np.ndarray.flatten(field) == fill_value
        elif field.ndim == 3:
            idx_fill_value_2D = np.any(field[:,:,:] == fill_value, axis = 0)
            idx_fill_value = np.ndarray.flatten(idx_fill_value_2D)
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


# # Function create_domain_and_LSM


def extract_domain_and_LSM(paths):
    xmin = 909
    xmax = 1453
    ymin = 1075
    ymax = 1555
    #
    Domain_data = {}
    #
    file_AMSR2 = paths["static"] + "LandOceanLakeMask_cosi-ease2-050.nc"
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
    Domain_data["distance_to_land"] = distance_transform_cdt(Domain_data["LSM"], metric = "taxicab") # Number of grid points to land
    #
    return(Domain_data)


# # Function create_AMSR2_SIC

class create_AMSR2_predictor():
    def __init__(self, Domain_data, paths, crs, today_date):
        self.Domain_data = Domain_data
        self.paths = paths
        self.crs = crs
        self.today_date = today_date
        self.yesterday_date = (datetime.datetime.strptime(self.today_date, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
        self.two_days_ago_date = (datetime.datetime.strptime(self.today_date, "%Y%m%d") - datetime.timedelta(days = 2)).strftime("%Y%m%d")
    #
    def concatenate_SIC(self, date_min, date_max):
        # date_max must be today_date, date_min is generally yesterday_date, except if there are not enough AMSR2 observations. If so, date_min is two_days_ago_date.
        #
        idx_bool = False
        SIC_conc = np.full(np.shape(self.Domain_data["LSM"]), np.nan)
        if date_max == (datetime.datetime.strptime(date_min, "%Y%m%d") + datetime.timedelta(days = 1)).strftime("%Y%m%d"):
            AMSR2_files = sorted(glob.glob(self.paths["AMSR2"] + "multisensor_" + date_min + "*.nc"))
        else:
            yesterday = (datetime.datetime.strptime(date_max, "%Y%m%d") - datetime.timedelta(days = 1)).strftime("%Y%m%d")
            AMSR2_files_2d_ago = glob.glob(self.paths["AMSR2"] + "multisensor_" + date_min + "*.nc")
            AMSR2_files_1d_ago = glob.glob(self.paths["AMSR2"] + "multisensor_" + yesterday + "*.nc")
            AMSR2_files = sorted(AMSR2_files_2d_ago + AMSR2_files_1d_ago)
        #
        for fi in AMSR2_files:
            end_date = fi[-15:-3]
            if datetime.datetime.strptime(end_date, "%Y%m%d%H%M") <= datetime.datetime.strptime(date_max + "0000", "%Y%m%d%H%M"):
                nc = netCDF4.Dataset(fi, "r")
                x_multisensor = nc.variables["xc"][:] * 1000
                y_multisensor = nc.variables["yc"][:] * 1000
                SIC_multisensor = np.ndarray.flatten(nc.variables["ice_conc"][0,:,:])
                nc.close()
                #
                if idx_bool == False:
                    xx_COSI, yy_COSI = np.meshgrid(self.Domain_data["x"], self.Domain_data["y"])
                    xx_COSI_flat = np.ndarray.flatten(xx_COSI)
                    yy_COSI_flat = np.ndarray.flatten(yy_COSI)
                    #
                    transform_multisensor_to_COSI = pyproj.Transformer.from_crs(self.crs["AMSR2_multisensor"], self.crs["AMSR2_COSI"], always_xy = True)
                    xx_multisensor, yy_multisensor = np.meshgrid(x_multisensor, y_multisensor)
                    xx_multisensor_COSIproj, yy_multisensor_COSIproj = transform_multisensor_to_COSI.transform(xx_multisensor, yy_multisensor)
                    xx_multisensor_COSIproj_flat = np.ndarray.flatten(xx_multisensor_COSIproj)
                    yy_multisensor_COSIproj_flat = np.ndarray.flatten(yy_multisensor_COSIproj)
                    #
                    inter_idx = nearest_neighbor_indexes(xx_multisensor_COSIproj_flat, yy_multisensor_COSIproj_flat, xx_COSI_flat, yy_COSI_flat)
                    idx_bool = True
                #
                SIC_interp = SIC_multisensor[inter_idx]
                SIC_regrid = np.reshape(SIC_interp, (len(self.Domain_data["y"]), len(self.Domain_data["x"])), order = "C")
                idx_valid = np.logical_and(SIC_regrid >= 0, SIC_regrid <= 100)
                SIC_conc[idx_valid == True] = SIC_regrid[idx_valid == True]
        #
        idx_pole_hole = np.logical_and(self.Domain_data["lat"] >= 88, np.isnan(SIC_conc) == True)
        SIC_coverage = np.full(np.shape(SIC_conc), np.nan)
        SIC_coverage[np.logical_or(np.isnan(SIC_conc) == False, idx_pole_hole == True)] = 1
        #
        idx_nan = np.logical_or(idx_pole_hole == True, np.isnan(SIC_coverage) == True)
        xx_COSI_cov = np.ndarray.flatten(xx_COSI[idx_nan == False])
        yy_COSI_cov = np.ndarray.flatten(yy_COSI[idx_nan == False])
        SIC_conc_cov = np.ndarray.flatten(SIC_conc[idx_nan == False])
        inter_idx = nearest_neighbor_indexes(xx_COSI_cov, yy_COSI_cov, xx_COSI_flat, yy_COSI_flat)
        SIC_filled_interp = SIC_conc_cov[inter_idx]
        SIC_filled = np.reshape(SIC_filled_interp, (len(self.Domain_data["y"]), len(self.Domain_data["x"])), order = "C")
        SIC_filled[self.Domain_data["LSM"] == 0] = 0
        #
        return(SIC_filled, SIC_coverage)
    #
    def produce_AMSR2_SIC(self):
        LSM_tolerance = 15 # Tolerance of 15 grid points (75 km) from land sea mask
        N_grid_points_without_AMSR2_coverage = 10 # Second tolerance for the number of grid points without AMSR2 coverage
        #
        SIC_AMSR2, SIC_coverage = self.concatenate_SIC(self.yesterday_date, self.today_date)
        SIC_coverage[self.Domain_data["distance_to_land"] <= LSM_tolerance] = 1
        #
        if np.sum(np.isnan(SIC_coverage) == True) > N_grid_points_without_AMSR2_coverage:
            SIC_AMSR2, SIC_coverage = self.concatenate_SIC(self.two_days_ago_date, self.today_date)
            SIC_coverage[self.Domain_data["distance_to_land"] <= LSM_tolerance] = 1
            #
            if np.sum(np.isnan(SIC_coverage) == True) > N_grid_points_without_AMSR2_coverage:
                raise ValueError("AMSR2 coverage is not good enough to produce the AMSR2 sea ice concentration predictor")
        #
        return(SIC_AMSR2)


# # ECMWF_time_steps_to_daily_time_steps function => Compute the daily mean of the variable for each days
# 
#     time_ECMWF: time variable in ECMWF netCDF files
#     field: 2D array variable
#     ndays: number of days (lead time) to compute 


def ECMWF_time_steps_to_daily_time_steps(time_ECMWF, field, ndays):
    lead_time = (time_ECMWF - time_ECMWF[0]) / 3600
    ts_start = np.linspace(0 * 24, (ndays - 1) * 24, ndays)
    ts_end = ts_start + 24
    daily_field = np.full((ndays, field.shape[1], field.shape[2]), np.nan)
    #
    for ts in range(0, ndays):
        lead_time_idx = np.squeeze(np.where(np.logical_and(lead_time >= ts_start[ts], lead_time < ts_end[ts])))
        daily_field[ts,:,:] = np.nanmean(np.ma.squeeze(field[lead_time_idx,:,:]), axis = 0)
    #
    return(daily_field)


# # Rotate_wind function
# 
#     x_wind, y_wind, lats, lons must be numpy arrays


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


# # Function create_ECMWF_predictors


def create_ECMWF_predictors(Domain_data, paths, crs, proj, today_date):
    lead_time_max = 10
    file_ecmwf = paths["ECMWF"] + "ec_atmo_0_1deg_" + today_date + "T000000Z_3h.nc"
    #
    ECMWF = {}
    nc = netCDF4.Dataset(file_ecmwf, "r")
    ECMWF["time"] = nc.variables["time"][:]
    ECMWF["lat"] = nc.variables["latitude"][:]
    ECMWF["lon"] = nc.variables["longitude"][:]
    idx_lat = ECMWF["lat"] > (np.min(Domain_data["lat"]) - 0.3)
    ECMWF["lat"] = ECMWF["lat"][idx_lat == True]
    ECMWF["T2M"] = nc.variables["air_temperature_2m"][:,0,idx_lat,:]
    ECMWF["U10M"] = nc.variables["x_wind_10m"][:,0,idx_lat,:]
    ECMWF["V10M"] = nc.variables["y_wind_10m"][:,0,idx_lat,:]
    nc.close()
    #
    transform_ECMWF_to_COSI = pyproj.Transformer.from_crs(crs["ECMWF"], crs["AMSR2_COSI"], always_xy = True)
    lons_ecmwf, lats_ecmwf = np.meshgrid(ECMWF["lon"], ECMWF["lat"])
    xx_ECMWF_COSIproj, yy_ECMWF_COSIproj = transform_ECMWF_to_COSI.transform(lons_ecmwf, lats_ecmwf)
    #
    Data_ECMWFgrid = {}
    for var in ["T2M", "U10M", "V10M"]:
        Data_ECMWFgrid[var] = ECMWF_time_steps_to_daily_time_steps(ECMWF["time"], ECMWF[var], lead_time_max)
    #
    x_wind = np.full((lead_time_max, len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
    y_wind = np.full((lead_time_max, len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
    #
    for ts in range(0, lead_time_max):
        x_wind_rot, y_wind_rot = rotate_wind(np.ndarray.flatten(Data_ECMWFgrid["U10M"][ts,:,:]), 
                                             np.ndarray.flatten(Data_ECMWFgrid["V10M"][ts,:,:]),
                                             np.ndarray.flatten(lats_ecmwf), 
                                             np.ndarray.flatten(lons_ecmwf), 
                                             proj["ECMWF"], 
                                             proj["AMSR2_COSI"]
                                             )
        #
        x_wind[ts,:,:] = np.reshape(x_wind_rot, (len(ECMWF["lat"]), len(ECMWF["lon"])), order = "C")
        y_wind[ts,:,:] = np.reshape(y_wind_rot, (len(ECMWF["lat"]), len(ECMWF["lon"])), order = "C")    
    #
    Data_ECMWFgrid["wind_x"] = np.copy(x_wind)
    Data_ECMWFgrid["wind_y"] = np.copy(y_wind)
    Data_ECMWFgrid.pop("U10M")
    Data_ECMWFgrid.pop("V10M")
    #
    Cum_data_ECMWFgrid = {}
    for var in Data_ECMWFgrid:
        var_cum = np.full((lead_time_max, len(ECMWF["lat"]), len(ECMWF["lon"])), np.nan)
        for ts in range(0, lead_time_max):
             var_cum[ts,:,:] = np.nanmean(Data_ECMWFgrid[var][0:ts+1,:,:], axis = 0)
        #
        Cum_data_ECMWFgrid[var + "_cum"] = np.copy(var_cum)
        Cum_data_ECMWFgrid[var + "_cum"][np.isnan(var_cum) == True] = -9999
        Cum_data_ECMWFgrid[var + "_cum"][np.isinf(var_cum) == True] = -9999
    #
    Data_COSIgrid = {}
    for var in Cum_data_ECMWFgrid:
        Data_COSIgrid[var] = nearest_neighbor_interp(xx_ECMWF_COSIproj, yy_ECMWF_COSIproj, Domain_data["x"], Domain_data["y"], Cum_data_ECMWFgrid[var], fill_value = -9999)
    #
    return(Data_COSIgrid)


# # Function write_hdf5_predictors


def write_hdf5_predictors(Datasets, paths, today_date):
    timestamps = []
    #
    for lt in range(0, 10):
        timestamps.append((datetime.datetime.strptime(today_date, "%Y%m%d") + datetime.timedelta(days = lt)).timestamp())
    #
    path_output = paths["predictors"] 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "AICE_predictors_" + today_date + "T000000Z.h5"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    #
    hf = h5py.File(output_filename, 'w')
    hf.create_dataset("time", data = timestamps)
    hf.create_dataset("x", data = Datasets["Domain_data"]["x"])
    hf.create_dataset("y", data = Datasets["Domain_data"]["y"])
    hf.create_dataset("lat", data = Datasets["Domain_data"]["lat"])
    hf.create_dataset("lon", data = Datasets["Domain_data"]["lon"])
    hf.create_dataset("LSM", data = Datasets["Domain_data"]["LSM"])
    hf.create_dataset("SICobs_AMSR2", data = Datasets["SICobs_AMSR2_SIC"])
    hf.create_dataset("ECMWF_T2M", data = Datasets["ECMWF"]["T2M_cum"])
    hf.create_dataset("ECMWF_x_wind", data = Datasets["ECMWF"]["wind_x_cum"])
    hf.create_dataset("ECMWF_y_wind", data = Datasets["ECMWF"]["wind_y_cum"])
    hf.close()
