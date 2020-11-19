# import libraries
import os, os.path
import numpy as np
import pandas as pd
# import geopandas as gpd
import sys
from IPython.display import Image
# from shapely.geometry import Point, Polygon
from math import factorial
import scipy
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression
from patsy import cr

from datetime import date
import datetime
import time

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sb



################################################################
#####
#####                   Function definitions
#####
################################################################

def create_calendar_table(SF_year = 2017):
    start = str(SF_year) + "-01-01"
    end = str(SF_year) + "-12-31"
    
    df = pd.DataFrame({"Date": pd.date_range(start, end)})
    df["SF_year"] = df.Date.dt.year
    
    # add day of year
    df["doy"] = 1 + np.arange(len(df))
    # df['Weekday'] = df['Date'].dt.day_name()
    
    return df

########################################################################

def Null_SOS_EOS_by_DoYDiff(pd_TS, min_season_length=40):
    """
    input: pd_TS is a pandas dataframe
           it includes a column SOS and a column EOS

    output: create a vector that measures distance between DoY 
    of an SOS and corresponding EOS.

    It is possible that the number of one of the SOS and EOS is
    different from the other. (perhaps just by 1)

    So, we need to keep that in mind.
    """
    pd_TS_DoYDiff = pd_TS.copy()

    # find indexes of SOS and EOS
    SOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff['SOS'] != 0].tolist()
    EOS_indexes = pd_TS_DoYDiff.index[pd_TS_DoYDiff['EOS'] != 0].tolist()

    """
    It seems it is possible to only have 1 SOS with no EOS. (or vice versa).
    In this case we can consider we only have 1 season!
    """
    if len(SOS_indexes) == 0 or len(EOS_indexes) == 0:
        return pd_TS_DoYDiff

    """
    First we need to fix the prolems such as having 2 SOS and only 1 EOS, or,
                                                    2 EOS and only 1 SOS, or,
    it is possible that number of SOSs and number of EOSs are identical,
    but the plot starts with EOS and ends with SOS.

    It is also possible that first EOS is ealier than first SOS.

    """
    #
    # Check if first EOS is less than first SOS
    #
    SOS_pointer = SOS_indexes[0]
    EOS_pointer = EOS_indexes[0]
    if (pd_TS_DoYDiff.loc[EOS_pointer, 'Date'] < pd_TS_DoYDiff.loc[SOS_pointer, 'Date']):
        
        # Remove the false EOS from dataFrame
        pd_TS_DoYDiff.loc[EOS_pointer, 'EOS'] = 0
        
        # remove the first element of EOS indexes
        EOS_indexes.pop(0)

    #
    # Check if last SOS is greater than last EOS
    #
    if len(EOS_indexes)==0 or len(EOS_indexes)==0:
        return pd_TS_DoYDiff

    SOS_pointer = SOS_indexes[-1]
    EOS_pointer = EOS_indexes[-1]
    if (pd_TS_DoYDiff.loc[EOS_pointer, 'Date'] < pd_TS_DoYDiff.loc[SOS_pointer, 'Date']):
        
        # Remove the false EOS from dataFrame
        pd_TS_DoYDiff.loc[SOS_pointer, 'SOS'] = 0
        
        # remove the first element of EOS indexes
        SOS_indexes.pop()
    
    if len(SOS_indexes) != len(EOS_indexes):
        #
        # in this case we have an extra SOS (at the end) or EOS (at the beginning)
        #
        print ("Error occured at: ")
        print (pd_TS.ID.unique()[0])
        print (pd_TS.image_year.unique()[0])
        raise ValueError("SOS and EOS are not of the same length.")

    """
    Go through seasons and invalidate them in their length is too short
    """
    for ii in np.arange(len(SOS_indexes)):
        SOS_pointer = SOS_indexes[ii]
        EOS_pointer = EOS_indexes[ii]
        
        current_growing_season_Length = (pd_TS_DoYDiff.loc[EOS_pointer, 'Date'] - \
                                         pd_TS_DoYDiff.loc[SOS_pointer, 'Date']).days


        #
        #  Kill/invalidate the SOS and EOS if growing season length is too short
        #
        if current_growing_season_Length < min_season_length:
            pd_TS_DoYDiff.loc[SOS_pointer, 'SOS'] = 0
            pd_TS_DoYDiff.loc[EOS_pointer, 'EOS'] = 0
        
    return(pd_TS_DoYDiff)

########################################################################

def addToDF_SOS_EOS_White(pd_TS, VegIdx = "EVI", onset_thresh=0.15, offset_thresh=0.15):
    """
    In this methods the NDVI_Ratio = (NDVI - NDVI_min) / (NDVI_Max - NDVI_min)
    is computed.
    
    SOS or onset is when NDVI_ratio exceeds onset-threshold
    and EOS is when NDVI_ratio drops below off-set-threshold.
    """
    pandaFrame = pd_TS.copy()
    
    VegIdx_min = pandaFrame[VegIdx].min()
    VegIdx_max = pandaFrame[VegIdx].max()
    VegRange = VegIdx_max - VegIdx_min + sys.float_info.epsilon
    
    colName = VegIdx + "_ratio"
    pandaFrame[colName] = (pandaFrame[VegIdx] - VegIdx_min) / VegRange

    # if (onset_thresh == offset_thresh):
    #     SOS_EOS_candidates = pandaFrame[colName] - onset_thresh
    #     sign_change = find_signChange_locs_EqualOnOffset(SOS_EOS_candidates.values)
    # else:
    #     SOS_candidates = pandaFrame[colName] - onset_thresh
    #     EOS_candidates = offset_thresh - pandaFrame[colName]
    #     sign_change = find_signChange_locs_DifferentOnOffset(SOS_candidates.values, EOS_candidates.values)
    # pandaFrame['SOS_EOS'] = sign_change * pandaFrame[VegIdx]
    
    SOS_candidates = pandaFrame[colName] - onset_thresh
    EOS_candidates = offset_thresh - pandaFrame[colName]

    BOS, EOS = find_signChange_locs_DifferentOnOffset(SOS_candidates, EOS_candidates)
    pandaFrame['SOS'] = BOS * pandaFrame[VegIdx]
    pandaFrame['EOS'] = EOS * pandaFrame[VegIdx]

    return(pandaFrame)

########################################################################

def find_signChange_locs_DifferentOnOffset(SOS_candids, EOS_candids):
    if type(SOS_candids) != np.ndarray:
        SOS_candids = SOS_candids.values

    if type(EOS_candids) != np.ndarray:
        EOS_candids = EOS_candids.values

    SOS_sign_change = np.zeros(len(SOS_candids))
    EOS_sign_change = np.zeros(len(EOS_candids))

    pointer = 0
    for pointer in np.arange(0, len(SOS_candids)-1):
        if SOS_candids[pointer] < 0:
            if SOS_candids[pointer+1] > 0:
                SOS_sign_change[pointer+1] = 1

        if EOS_candids[pointer] < 0:
            if EOS_candids[pointer+1] > 0:
                EOS_sign_change[pointer+1] = 1

    # sign_change = SOS_sign_change + EOS_sign_change
    return (SOS_sign_change, EOS_sign_change) # sign_change

########################################################################



########################################################################

def correct_big_jumps_1DaySeries(dataTMS_jumpie, give_col, maxjump_perDay = 0.015):
    """
    in the function correct_big_jumps_preDefinedJumpDays(.) we have
    to define the jump_amount and the no_days_between_points.
    For example if we have a jump more than 0.4 in less than 20 dats, then
    that is an outlier detected.
    
    Here we modify the approach to be flexible in the following sense:
    if the amount of increase in NDVI is more than #_of_Days * 0.02 then 
    an outlier is detected and we need interpolation.
    
    0.015 came from the SG based paper that used 0.4 jump in NDVI for 20 days.
    That translates into 0.02 = 0.4 / 20 per day.
    But we did choose 0.015 as default
    """
    dataTMS = dataTMS_jumpie.copy()
    dataTMS = initial_clean(df = dataTMS, column_to_be_cleaned = give_col)

    dataTMS.sort_values(by=['image_year', 'doy'], inplace=True)
    dataTMS.reset_index(drop=True, inplace=True)
    dataTMS['system_start_time'] = dataTMS['system_start_time'] / 1000

    thyme_vec = dataTMS['system_start_time'].values.copy()
    Veg_indks = dataTMS[give_col].values.copy()

    time_diff = thyme_vec[1:] - thyme_vec[0:len(thyme_vec)-1]
    time_diff_in_days = time_diff / 86400
    time_diff_in_days = time_diff_in_days.astype(int)

    Veg_indks_diff = Veg_indks[1:] - Veg_indks[0:len(thyme_vec)-1]
    jump_indexes = np.where(Veg_indks_diff > maxjump_perDay)
    jump_indexes = jump_indexes[0]

    jump_indexes = jump_indexes.tolist()

    # It is possible that the very first one has a big jump in it.
    # we cannot interpolate this. so, lets just skip it.
    if len(jump_indexes) > 0: 
        if jump_indexes[0] == 0:
            jump_indexes.pop(0)
    
    if len(jump_indexes) > 0:    
        for jp_idx in jump_indexes:
            if  Veg_indks_diff[jp_idx] >= (time_diff_in_days[jp_idx] * maxjump_perDay):
                #
                # form a line using the adjacent points of the big jump:
                #
                x1, y1 = thyme_vec[jp_idx-1], Veg_indks[jp_idx-1]
                x2, y2 = thyme_vec[jp_idx+1], Veg_indks[jp_idx+1]
                # print (x1)
                # print (x2)
                m = np.float(y2 - y1) / np.float(x2 - x1) # slope
                b = y2 - (m*x2)           # intercept

                # replace the big jump with linear interpolation
                Veg_indks[jp_idx] = m * thyme_vec[jp_idx] + b

    dataTMS[give_col] = Veg_indks
    return(dataTMS)

########################################################################

def correct_big_jumps_preDefinedJumpDays(dataTS_jumpy, given_col, jump_amount = 0.4, no_days_between_points=20):
    dataTS = dataTS_jumpy.copy()
    dataTS = initial_clean(df = dataTS, column_to_be_cleaned = given_col)

    dataTS.sort_values(by=['image_year', 'doy'], inplace=True)
    dataTS.reset_index(drop=True, inplace=True)
    dataTS['system_start_time'] = dataTS['system_start_time'] / 1000

    thyme_vec = dataTS['system_start_time'].values.copy()
    Veg_indks = dataTS[given_col].values.copy()

    time_diff = thyme_vec[1:] - thyme_vec[0:len(thyme_vec)-1]
    time_diff_in_days = time_diff / 86400
    time_diff_in_days = time_diff_in_days.astype(int)

    Veg_indks_diff = Veg_indks[1:] - Veg_indks[0:len(thyme_vec)-1]
    jump_indexes = np.where(Veg_indks_diff > 0.4)
    jump_indexes = jump_indexes[0]

    # It is possible that the very first one has a big jump in it.
    # we cannot interpolate this. so, lets just skip it.
    if jump_indexes[0] == 0:
        jump_indexes.pop(0)

    if len(jump_indexes) > 0:    
        for jp_idx in jump_indexes:
            if time_diff_in_days[jp_idx] >= 20:
                #
                # form a line using the adjacent points of the big jump:
                #
                x1, y1 = thyme_vec[jp_idx-1], Veg_indks[jp_idx-1]
                x2, y2 = thyme_vec[jp_idx+1], Veg_indks[jp_idx+1]
                m = np.float(y2 - y1) / np.float(x2 - x1) # slope
                b = y2 - (m*x2)           # intercept

                # replace the big jump with linear interpolation
                Veg_indks[jp_idx] = m * thyme_vec[jp_idx] + b

    dataTS[given_col] = Veg_indks

    return(dataTS)

########################################################################

def interpolate_outliers_EVI_NDVI(outlier_input, given_col):
    """
    outliers are those that are beyond boundaries. For example and EVI value of 2.
    Big jump in the other function means we have a big jump but we are still
    within the region of EVI values. If in 20 days we have a jump of 0.3 then that is noise.

    in 2017 data I did not see outlier in NDVI. It only happened in EVI.
    """
    outlier_df = outlier_input.copy()
    outlier_df = initial_clean(df = outlier_df, column_to_be_cleaned = given_col)

    outlier_df.sort_values(by=['image_year', 'doy'], inplace=True)
    outlier_df.reset_index(drop=True, inplace=True)
    
    # 1st block
    time_vec = outlier_df['system_start_time'].values.copy()
    vec = outlier_df[given_col].values.copy()
    
    # find out where are outliers
    high_outlier_inds = np.where(vec > 1)[0]
    low_outlier_inds = np.where(vec < -1)[0]
    
    all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
    all_outliers_idx = np.sort(all_outliers_idx)
    non_outiers = np.arange(len(vec))[~ np.in1d(np.arange(len(vec)) , all_outliers_idx)]
    
    # 2nd block
    if len(all_outliers_idx) == 0:
        return outlier_df
    
    """
    it is possible that for a field we only have x=2 data points
    where all the EVI/NDVI is outlier. Then, there is nothing to 
    use for interpolation. So, we return an empty datatable
    """
    if len(all_outliers_idx) == len(outlier_df):
        outlier_df = initial_clean(df=outlier_df, column_to_be_cleaned=given_col)
        outlier_df = outlier_df[outlier_df[given_col] < 1.5]
        outlier_df = outlier_df[outlier_df[given_col] > - 1.5]
        return outlier_df

    # 3rd block

    # Get rid of outliers that are at the beginning of the time series
    # if len(non_outiers) > 0 :
    if non_outiers[0] > 0 :
        vec[0:non_outiers[0]] = vec[non_outiers[0]]
        
        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[~ np.in1d(np.arange(len(vec)) , all_outliers_idx)]
        if len(all_outliers_idx) == 0:
            outlier_df[given_col] = vec
            return outlier_df

    # 4th block
    # Get rid of outliers that are at the end of the time series
    if non_outiers[-1] < (len(vec)-1) :
        vec[non_outiers[-1]:] = vec[non_outiers[-1]]
        
        # find out where are outliers
        high_outlier_inds = np.where(vec > 1)[0]
        low_outlier_inds = np.where(vec < -1)[0]

        all_outliers_idx = np.concatenate((high_outlier_inds, low_outlier_inds))
        all_outliers_idx = np.sort(all_outliers_idx)
        non_outiers = np.arange(len(vec))[~ np.in1d(np.arange(len(vec)) , all_outliers_idx)]
        if len(all_outliers_idx) == 0:
            outlier_df[given_col] = vec
            return outlier_df
    """
    At this point outliers are in the middle of the vector
    and beginning and the end of the vector are clear.
    """
    for out_idx in all_outliers_idx:
        """
         Right here at the beginning we should check
         if vec[out_idx] is outlier or not. The reason is that
         there might be consecutive outliers at position m and m+1
         and we fix the one at m+1 when we are fixing m ...
        """
        # if ~(vec[out_idx] <= 1 and vec[out_idx] >= -1):
        if (vec[out_idx] >= 1 or vec[out_idx] <= -1):
            left_pointer = out_idx - 1
            right_pointer = out_idx + 1
            while ~(vec[right_pointer] <= 1 and vec[right_pointer] >= -1):
                right_pointer += 1

            # form the line and fill in the outlier valies
            x1, y1 = time_vec[left_pointer], vec[left_pointer]
            x2, y2 = time_vec[right_pointer], vec[right_pointer]

            time_diff = x2 - x1
            y_diff = y2 - y1
            slope = y_diff / time_diff
            intercept = y2 - (slope * x2)
            vec[left_pointer+1:right_pointer] = slope * time_vec[left_pointer+1:right_pointer] + intercept
    
    outlier_df[given_col] = vec
    return (outlier_df)

########################################################################

def initial_clean_NDVI(df):
    dt_copy = df.copy()
    # remove the useles system:index column
    if ("system:index" in list(dt_copy.columns)):
        dt_copy = dt_copy.drop(columns=['system:index'])
    
    # Drop rows whith NA in NDVI column.
    dt_copy = dt_copy[dt_copy['NDVI'].notna()]

    # replace values beyond 1 and -1 with 1.5 and -1.5
    dt_copy.loc[dt_copy['NDVI'] > 1, "NDVI"] = 1.5
    dt_copy.loc[dt_copy['NDVI'] < -1, "NDVI"] = -1.5

    if ("image_year" in list(dt_copy.columns)):
        dt_copy.image_year = dt_copy.image_year.astype(int)
    return (dt_copy)

########################################################################

def initial_clean_EVI(df):
    dt_copy = df.copy()
    # remove the useles system:index column
    if ("system:index" in list(dt_copy.columns)):
        dt_copy = dt_copy.drop(columns=['system:index'])
    
    # Drop rows whith NA in EVI column.
    dt_copy = dt_copy[dt_copy['EVI'].notna()]

    # replace values beyond 1 and -1 with 1.5 and -1.5
    dt_copy.loc[dt_copy['EVI'] > 1, "EVI"] = 1.5
    dt_copy.loc[dt_copy['EVI'] < -1, "EVI"] = -1.5

    if ("image_year" in list(dt_copy.columns)):
        dt_copy.image_year = dt_copy.image_year.astype(int)
    
    return (dt_copy)

########################################################################

def initial_clean(df, column_to_be_cleaned):
    dt_copy = df.copy()
    # remove the useles system:index column
    if ("system:index" in list(dt_copy.columns)):
        dt_copy = dt_copy.drop(columns=['system:index'])
    
    # Drop rows whith NA in column_to_be_cleaned column.
    dt_copy = dt_copy[dt_copy[column_to_be_cleaned].notna()]

    if (column_to_be_cleaned in ["NDVI", "EVI"]):
        dt_copy.loc[dt_copy[column_to_be_cleaned] > 1, column_to_be_cleaned] = 1.5
        dt_copy.loc[dt_copy[column_to_be_cleaned] < -1, column_to_be_cleaned] = -1.5

    return (dt_copy)

########################################################################

def convert_human_system_start_time_to_systemStart_time(humantimeDF):
    epoch_vec = pd.to_datetime(humantimeDF['human_system_start_time']).values.astype(np.int64) // 10 ** 6

    # add 83000000 mili sec. since system_start_time is 1 day ahead of image_taken_time
    # that is recorded in human_system_start_time column.
    epoch_vec = epoch_vec + 83000000
    humantimeDF['system_start_time'] = epoch_vec
    """
    not returning anything does the operation in place.
    so, you have to use this function like
    convert_human_system_start_time_to_systemStart_time(humantimeDF)

    If you do:
    humantimeDF = convert_human_system_start_time_to_systemStart_time(humantimeDF)
    Then humantimeDF will be nothing, since we are not returning anything.
    """
########################################################################

def add_human_start_time_by_YearDoY(a_Reg_DF):
    """
    This function is written for regularized data 
    where we miss the Epoch time and therefore, cannot convert it to
    human_start_time using add_human_start_time() function

    Learn:
    x = pd.to_datetime(datetime.datetime(2016, 1, 1) + datetime.timedelta(213 - 1))
    x

    year = 2020
    DoY = 185
    x = str(date.fromordinal(date(year, 1, 1).toordinal() + DoY - 1))
    x

    datetime.datetime(2016, 1, 1) + datetime.timedelta(213 - 1)
    """
    DF_C = a_Reg_DF.copy()

    # DF_C.image_year = DF_C.image_year.astype(float)
    DF_C.doy = DF_C.doy.astype(int)
    DF_C['human_system_start_time'] = pd.to_datetime(DF_C['image_year'].astype(int) * 1000 + DF_C['doy'], format='%Y%j')

    # DF_C.reset_index(drop=True, inplace=True)
    # DF_C['human_system_start_time'] = "1"

    # for row_no in np.arange(0, len(DF_C)):
    #     year = DF_C.loc[row_no, 'image_year']
    #     DoY = DF_C.loc[row_no, 'doy']
    #     x = str(date.fromordinal(date(year, 1, 1).toordinal() + DoY - 1))
    #     DF_C.loc[row_no, 'human_system_start_time'] = x

    return(DF_C)


########################################################################
########################################################################
########################################################################

#
# Kirti look here
#
# detect passing the threshod
def find_signChange_locs_EqualOnOffset(a_vec):
    asign = np.sign(a_vec) # we can drop .values here.
    sign_change = ((np.roll(asign, 1) - asign) != 0).astype(int)

    """
    np.sign considers 0 to have it's own sign, 
    different from either positive or negative values.
    So: 
    """
    sz = asign == 0
    while sz.any():
        asign[sz] = np.roll(asign, 1)[sz]
        sz = asign == 0

    """
    numpy.roll does a circular shift, so if the last 
    element has different sign than the first, 
    the first element in the sign_change array will be 1.
    """
    sign_change[0] = 0

    """
    # Another solution for sign change:
    np.where(np.diff(np.sign(Vector)))[0]
    np.where(np.diff(np.sign(Vector)))[0]
    """

    return(sign_change)

def regularize_movingWindow_windowSteps_2Yrs(one_field_df, SF_yr=2017, veg_idxs, window_size=10):
    #
    #  This function almost returns a data frame with data
    #  that are window_size away from each other. i.e. regular space in time.
    #  **** For **** 5 months + 12 months.
    #

    a_field_df = one_field_df.copy()
    # initialize output dataframe
    regular_cols = ['ID', 'Acres', 'county', 'CropGrp', 'CropTyp',
                    'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                    'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'image_year', 
                    'SF_year', 'doy', veg_idxs]
    #
    # for a good measure we start at 213 (214 does not matter either)
    # and the first 
    #
    first_year_steps = list(range(213, 365, 10))
    first_year_steps[-1] = 366

    full_year_steps = list(range(1, 365, 10))
    full_year_steps[-1] = 366

    DoYs = first_year_steps + full_year_steps

    #
    # There are 5 months first and then a full year
    # (31+30+30+30+31) + 365 = 517 days. If we do every 10 days 
    # then we have 51 data points
    #
    no_days = 517
    no_steps = int(no_days/window_size)

    regular_df = pd.DataFrame(data = None, 
                              index = np.arange(no_steps), 
                              columns = regular_cols)

    regular_df['ID'] = a_field_df.ID.unique()[0]
    regular_df['Acres'] = a_field_df.Acres.unique()[0]
    regular_df['county'] = a_field_df.county.unique()[0]
    regular_df['CropGrp'] = a_field_df.CropGrp.unique()[0]

    regular_df['CropTyp'] = a_field_df.CropTyp.unique()[0]
    regular_df['DataSrc'] = a_field_df.DataSrc.unique()[0]
    regular_df['ExctAcr'] = a_field_df.ExctAcr.unique()[0]
    regular_df['IntlSrD'] = a_field_df.IntlSrD.unique()[0]
    regular_df['Irrigtn'] = a_field_df.Irrigtn.unique()[0]

    regular_df['LstSrvD'] = a_field_df.LstSrvD.unique()[0]
    regular_df['Notes'] = str(a_field_df.Notes.unique()[0])
    regular_df['RtCrpTy'] = str(a_field_df.RtCrpTy.unique()[0])
    regular_df['Shap_Ar'] = a_field_df.Shap_Ar.unique()[0]
    regular_df['Shp_Lng'] = a_field_df.Shp_Lng.unique()[0]
    regular_df['TRS'] = a_field_df.TRS.unique()[0]

    regular_df['SF_year'] = a_field_df.SF_year.unique()[0]

    # I will write this in 3 for-loops.
    # perhaps we can do it in a cleaner way like using zip or sth.
    #
    #####################################################
    #
    #  First year (last 5 months of previous year)
    #
    #
    #####################################################
    for row_or_count in np.arange(len(first_year_steps)-1):
        curr_year = SF_yr - 1
        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()
        curr_time_window = curr_time_window[curr_time_window.doy >= first_year_steps[row_or_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < first_year_steps[row_or_count+1]]

        """
        In each time window peak the maximum of present values

        If in a window (e.g. 10 days) we have no value observed by Sentinel, 
        then use -1.5 as an indicator. That will be a gap to be filled. (function fill_theGap_linearLine).
        """
        if len(curr_time_window)==0: 
            regular_df.loc[row_or_count, veg_idxs] = -1.5
        else:
            regular_df.loc[row_or_count, veg_idxs] = max(curr_time_window[veg_idxs])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = first_year_steps[row_or_count]

    #############################################
    #
    #  Full year (main year, 12 months)
    #
    #############################################
    row_count_start = len(first_year_steps) - 1
    row_count_end = row_count_start + len(full_year_steps) - 1

    for row_or_count in np.arange(row_count_start, row_count_end):
        curr_year = SF_yr
        curr_count = row_or_count - row_count_start

        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()
        curr_time_window = curr_time_window[curr_time_window.doy >= full_year_steps[curr_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < full_year_steps[curr_count+1]]

        """
          In each time window peak the maximum of present values

          If in a window (e.g. 10 days) we have no value observed by Sentinel, 
          then use -1.5 as an indicator. That will be a gap to be filled (function fill_theGap_linearLine).
        """
        if len(curr_time_window)==0: 
            regular_df.loc[row_or_count, veg_idxs] = -1.5
        else:
            regular_df.loc[row_or_count, veg_idxs] = max(curr_time_window[veg_idxs])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = full_year_steps[curr_count]
    return(regular_df)



def extract_XValues_of_2Yrs_TS(regularized_TS, SF_yr):
    # old name extract_XValues_of_RegularizedTS_2Yrs().
    # I do not know why I had Regularized in it.
    # new name extract_XValues_of_2Yrs_TS
    """
    Jul 1.
    This function is being written since Kirti said
    we do not need to have parts of the next year. i.e. 
    if we are looking at what is going on in a field in 2017,
    we only need data since Aug. 2016 till the end of 2017.
    We do not need anything in 2018.
    """

    X_values_prev_year = regularized_TS[regularized_TS.image_year == (SF_yr - 1)]['doy'].copy().values
    X_values_full_year = regularized_TS[regularized_TS.image_year == (SF_yr)]['doy'].copy().values

    if check_leap_year(SF_yr - 1):
        X_values_full_year = X_values_full_year + 366
    else:
        X_values_full_year = X_values_full_year + 365
    return (np.concatenate([X_values_prev_year, X_values_full_year]))

def regularize_movingWindow_windowSteps_12Months(one_field_df, SF_yr=2017, V_idxs="NDVI", window_size=10):
    #
    #  This function almost returns a data frame with data
    #  that are window_size away from each other. i.e. regular space in time.
    
    # copy the field input into the new variale.
    a_field_df = one_field_df.copy()

    # initialize output dataframe
    regular_cols = ['ID', 'Acres', 'county', 'CropGrp', 'CropTyp',
                    'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                    'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'image_year', 
                    'SF_year', 'doy', V_idxs]

    full_year_steps = list(range(1, 365, 10)) # [1, 10, 20, 30, ..., 360]
    full_year_steps[-1] = 366 # save the last extra 5 (or 6) days.
    DoYs = full_year_steps


    no_days = 366 # number of days in a year
    no_steps = int(no_days/window_size) # 

    regular_df = pd.DataFrame(data = None, 
                              index = np.arange(no_steps), 
                              columns = regular_cols)

    regular_df['ID'] = a_field_df.ID.unique()[0]
    regular_df['Acres'] = a_field_df.Acres.unique()[0]
    regular_df['county'] = a_field_df.county.unique()[0]
    regular_df['CropGrp'] = a_field_df.CropGrp.unique()[0]

    regular_df['CropTyp'] = a_field_df.CropTyp.unique()[0]
    regular_df['DataSrc'] = a_field_df.DataSrc.unique()[0]
    regular_df['ExctAcr'] = a_field_df.ExctAcr.unique()[0]
    regular_df['IntlSrD'] = a_field_df.IntlSrD.unique()[0]
    regular_df['Irrigtn'] = a_field_df.Irrigtn.unique()[0]

    regular_df['LstSrvD'] = a_field_df.LstSrvD.unique()[0]
    regular_df['Notes'] = str(a_field_df.Notes.unique()[0])
    regular_df['RtCrpTy'] = str(a_field_df.RtCrpTy.unique()[0])
    regular_df['Shap_Ar'] = a_field_df.Shap_Ar.unique()[0]
    regular_df['Shp_Lng'] = a_field_df.Shp_Lng.unique()[0]
    regular_df['TRS'] = a_field_df.TRS.unique()[0]

    regular_df['SF_year'] = a_field_df.SF_year.unique()[0]
    
    # I will write this in 3 for-loops.
    # perhaps we can do it in a cleaner way like using zip or sth.
    #    
    
    for row_or_count in np.arange(len(full_year_steps)-1):
        curr_year = SF_yr
        
        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()

        # [1, 10, 20, 30, ..., 350, 366]
        curr_time_window = curr_time_window[curr_time_window.doy >= full_year_steps[row_or_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < full_year_steps[row_or_count+1]]

        if len(curr_time_window)==0: # this means in that time window there is no NDVI value
            regular_df.loc[row_or_count, V_idxs] = -1.5 # indicator for missing value
        else:
            regular_df.loc[row_or_count, V_idxs] = max(curr_time_window[V_idxs])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = full_year_steps[row_or_count]
        
    return (regular_df)

def fill_theGap_linearLine(regular_TS, V_idx, SF_year):

    # regular_TS: is output of function (regularize_movingWindow_windowSteps_12Months)

    a_regularized_TS = regular_TS.copy()

    if (len(a_regularized_TS.image_year.unique()) == 2):
        x_axis = extract_XValues_of_2Yrs_TS(regularized_TS = a_regularized_TS, SF_yr = SF_year)
    elif (len(a_regularized_TS.image_year.unique()) == 3):
        x_axis = extract_XValues_of_3Yrs_TS(regularized_TS = a_regularized_TS, SF_yr = SF_year)
    elif (len(a_regularized_TS.image_year.unique()) == 1):
        x_axis = a_regularized_TS["doy"].values.copy()

    TS_array = a_regularized_TS[V_idx].copy().values

    """
    TS_array[0] = -1.5
    TS_array[51] = -1.5
    TS_array[52] = -1.5
    TS_array[53] = -1.5
    TS_array.shape
    """

    """
    -1.5 is an indicator of missing values by Sentinel, i.e. a gap.
    The -1.5 was used as indicator in the function regularize_movingWindow_windowSteps_2Yrs()
    """
    missing_indicies = np.where(TS_array == -1.5)[0]
    Notmissing_indicies = np.where(TS_array != -1.5)[0]

    #
    #    Check if the first or last k values are missing
    #    if so, replace them with proper number and shorten the task
    #
    left_pointer = Notmissing_indicies[0]
    right_pointer = Notmissing_indicies[-1]

    if left_pointer > 0:
        TS_array[:left_pointer] = TS_array[left_pointer]

    if right_pointer < (len(TS_array) - 1):
        TS_array[right_pointer:] = TS_array[right_pointer]
    #    
    # update indexes.
    #
    missing_indicies = np.where(TS_array == -1.5)[0]
    Notmissing_indicies = np.where(TS_array != -1.5)[0]

    # left_pointer = Notmissing_indicies[0]
    stop = right_pointer
    right_pointer = left_pointer + 1

    missing_indicies = np.where(TS_array == -1.5)[0]

    while len(missing_indicies) > 0:
        left_pointer = missing_indicies[0] - 1
        left_value = TS_array[left_pointer]
        right_pointer = missing_indicies[0]
        
        while TS_array[right_pointer] == -1.5:
            right_pointer += 1
        
        right_value = TS_array[right_pointer]
        
        if (right_pointer - left_pointer) == 2:
            # if there is a single gap, then we have just average of the
            # values
            # Avoid extra computation!
            #
            TS_array[left_pointer + 1] = 0.5 * (TS_array[left_pointer] + TS_array[right_pointer])
        else:
            # form y= ax + b
            slope = (right_value - left_value) / (x_axis[right_pointer] - x_axis[left_pointer]) # a
            b = right_value - (slope * x_axis[right_pointer])
            TS_array[left_pointer+1 : right_pointer] = slope * x_axis[left_pointer+1 : right_pointer] + b
            missing_indicies = np.where(TS_array == -1.5)[0]
            
        
    a_regularized_TS[V_idx] = TS_array
    return (a_regularized_TS)

########################################################################
########################################################################
########################################################################

#
#   These will not give what we want. It is a 10-days window
#   The days are actual days. i.e. between each 2 entry of our
#   time series there is already some gap.
#

def add_human_start_time(HDF):
    HDF.system_start_time = HDF.system_start_time / 1000
    time_array = HDF["system_start_time"].values.copy()
    human_time_array = [time.strftime('%Y-%m-%d', time.localtime(x)) for x in time_array]
    HDF["human_system_start_time"] = human_time_array
    return(HDF)

########################################################################

def extract_XValues_of_3Yrs_TS(regularized_TS, SF_yr):
    # old name extract_XValues_of_RegularizedTS_3Yrs().
    # I do not know why I had Regularized in it.
    # new name extract_XValues_of_3Yrs_TS
    """
    Jul 1.
    This function is written for inluding data from 3 years.
    e.g.:
    3 months in 2016, full year 2017, and 3 months in 2018

    """

    X_values_prev_year = regularized_TS[regularized_TS.image_year == (SF_yr - 1)]['doy'].copy().values
    X_values_full_year = regularized_TS[regularized_TS.image_year == (SF_yr)]['doy'].copy().values
    X_values_next_year = regularized_TS[regularized_TS.image_year == (SF_yr + 1)]['doy'].copy().values

    if check_leap_year(SF_yr-1):
        X_values_full_year = X_values_full_year + 366
        X_values_next_year = X_values_next_year + 366
    else:
        X_values_full_year = X_values_full_year + 365
        X_values_next_year = X_values_next_year + 365

    if check_leap_year(SF_yr):
        X_values_next_year = X_values_next_year + 366
    else:
        X_values_next_year = X_values_next_year + 365
    
    return (np.concatenate([X_values_prev_year, X_values_full_year, X_values_next_year]))

########################################################################

def check_leap_year(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return (True)
            else:
                return (False)
        else:
            return (True)
    else:
        return (False)

########################################################################

def regularize_movingWindow_windowSteps_18Months(one_field_df, SF_yr, V_idks, window_size=10):
    #
    #  This function almost returns a data frame with data
    #  that are window_size away from each other. i.e. regular space in time.
    #  **** For **** 3 months + 12 months + 3 months data.
    #
    
    a_field_df = one_field_df.copy()
    # initialize output dataframe
    regular_cols = ['ID', 'Acres', 'county', 'CropGrp', 'CropTyp',
                    'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                    'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'image_year', 
                    'SF_year', 'doy', V_idks]
    #
    # for a good measure we start at 273 (274 does not matter either)
    # and the first 
    #
    first_year_steps = list(range(274, 365, 10))
    first_year_steps[-1] = 366

    full_year_steps = list(range(1, 365, 10))
    full_year_steps[-1] = 366

    last_year_steps = list(range(1, 91, 10))
    last_year_steps = last_year_steps + [91]

    DoYs = first_year_steps + full_year_steps + last_year_steps

    #
    # There are 3 months before, a full year, and 3 months after
    # (30+30+31) + 365 + (31+28+31) = 546 days. If we do every 10 days 
    # then we have 54 data points
    #
    no_days = 546
    no_steps = int(no_days/window_size)

    regular_df = pd.DataFrame(data = None, 
                              index = np.arange(no_steps), 
                              columns = regular_cols)

    regular_df['ID'] = a_field_df.ID.unique()[0]
    regular_df['Acres'] = a_field_df.Acres.unique()[0]
    regular_df['county'] = a_field_df.county.unique()[0]
    regular_df['CropGrp'] = a_field_df.CropGrp.unique()[0]

    regular_df['CropTyp'] = a_field_df.CropTyp.unique()[0]
    regular_df['DataSrc'] = a_field_df.DataSrc.unique()[0]
    regular_df['ExctAcr'] = a_field_df.ExctAcr.unique()[0]
    regular_df['IntlSrD'] = a_field_df.IntlSrD.unique()[0]
    regular_df['Irrigtn'] = a_field_df.Irrigtn.unique()[0]

    regular_df['LstSrvD'] = a_field_df.LstSrvD.unique()[0]
    regular_df['Notes'] = str(a_field_df.Notes.unique()[0])
    regular_df['RtCrpTy'] = str(a_field_df.RtCrpTy.unique()[0])
    regular_df['Shap_Ar'] = a_field_df.Shap_Ar.unique()[0]
    regular_df['Shp_Lng'] = a_field_df.Shp_Lng.unique()[0]
    regular_df['TRS'] = a_field_df.TRS.unique()[0]

    regular_df['SF_year'] = a_field_df.SF_year.unique()[0]
    
    # I will write this in 3 for-loops.
    # perhaps we can do it in a cleaner way like using zip or sth.
    #
    #############################################
    #
    #  First year (last 3 months of previous year)
    #
    #############################################
    for row_or_count in np.arange(len(first_year_steps)-1):
        curr_year = SF_yr - 1
        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()
        curr_time_window = curr_time_window[curr_time_window.doy >= first_year_steps[row_or_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < first_year_steps[row_or_count+1]]

        if len(curr_time_window)==0: 
            regular_df.loc[row_or_count, V_idks] = -1.5
        else:
            regular_df.loc[row_or_count, V_idks] = max(curr_time_window[V_idks])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = first_year_steps[row_or_count]

    #############################################
    #
    #  Full year (main year, 12 months)
    #
    #############################################
    row_count_start = len(first_year_steps) - 1
    row_count_end = row_count_start + len(full_year_steps) - 1

    for row_or_count in np.arange(row_count_start, row_count_end):
        curr_year = SF_yr
        curr_count = row_or_count - row_count_start

        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()
        curr_time_window = curr_time_window[curr_time_window.doy >= full_year_steps[curr_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < full_year_steps[curr_count+1]]

        if len(curr_time_window)==0: 
            regular_df.loc[row_or_count, V_idks] = -1.5
        else:
            regular_df.loc[row_or_count, V_idks] = max(curr_time_window[V_idks])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = full_year_steps[curr_count]

    #############################################
    #
    #  Last year (First 3 months of Next year)
    #
    #############################################
    row_count_start = len(first_year_steps) - 1 + len(full_year_steps) - 1
    row_count_end = row_count_start + len(last_year_steps) - 1

    for row_or_count in np.arange(row_count_start, row_count_end):
        curr_year = SF_yr + 1
        curr_count = row_or_count - row_count_start

        curr_time_window = a_field_df[a_field_df.image_year == curr_year].copy()
        curr_time_window = curr_time_window[curr_time_window.doy >= last_year_steps[curr_count]]
        curr_time_window = curr_time_window[curr_time_window.doy < last_year_steps[curr_count+1]]

        if len(curr_time_window)==0: 
            regular_df.loc[row_or_count, V_idks] = -1.5
        else:
            regular_df.loc[row_or_count, V_idks] = max(curr_time_window[V_idks])

        regular_df.loc[row_or_count, 'image_year'] = curr_year
        regular_df.loc[row_or_count, 'doy'] = last_year_steps[curr_count]
        
    return (regular_df)

########################################################################

def max_movingWindow_windowSteps(VI_TS_npArray, window_size):
    """
    This function moves the window by a step size of window_size.
    i.e. window 1 is from 1-10, window 2 is from 11-20...
    """
    # replace NAs with -1.5
    VI_TS_npArray = np.where(np.isnan(VI_TS_npArray), -1.5, VI_TS_npArray)

    # form the output vector
    output_len = int(np.floor(len(VI_TS_npArray) / window_size))
    output = np.ones(output_len) * (-666)

    for count in range(output_len):
        window_start = count * window_size
        window_end = window_start + window_size

        if (count == output_len-1):
            window_end = len(VI_TS_npArray)

        curr_window = VI_TS_npArray[window_start : window_end]
        output[count] = max(curr_window)
    return(output)

########################################################################

def max_movingWindow_1Steps(VI_TS_npArray, window_size):
    """
    This function moves the window by a step size of 1.
    i.e. window 1 is from 1-10, window 2 is from 2-11, window 3 is 3-12 ...
    """
    # replace NAs with -1.5
    VI_TS_npArray = np.where(np.isnan(VI_TS_npArray), -1.5, VI_TS_npArray)

    # form the output vector
    output_len = int(len(VI_TS_npArray) - window_size)
    output = np.ones(output_len) * (-666)

    for count in range(output_len):
        window_start = count
        window_end = window_start + window_size
        output[count] = max(VI_TS_npArray[window_start : window_end])
    return(output)

########################################################################

def find_difference_date_by_systemStartTime(earlier_day_epoch, later_day_epoch):
    #
    #  Given two epoch time, find the difference between them in number of days
    #

    early = datetime.datetime.fromtimestamp(earlier_day_epoch)
    late =  datetime.datetime.fromtimestamp(later_day_epoch)
    diff = ( late - early).days
    return (diff)

########################################################################

def correct_timeColumns_dataTypes(dtf):
    dtf.system_start_time = dtf.system_start_time/1000
    dtf = dtf.astype({'doy': 'int', 'image_year': 'int'})
    return(dtf)

########################################################################

def divide_double_nonDouble_peaks(dt_dt):
    
    # subset the double-peaked
    double_peaked = dt_dt[dt_dt["peak_count"] == 2.0].copy()

    # subset the not double-peaked
    not_double_peaked = dt_dt[dt_dt["peak_count"] != 2.0 ].copy()

    return (double_peaked, not_double_peaked)

def divide_double_nonDouble_by_notes(dt_d):
    dt_copy = dt_d.copy()

    # convert NaN and NAs to string so we can subset/index 
    dt_copy[["Notes"]] = dt_copy[["Notes"]].astype(str)

    # convert to lower case
    dt_copy["Notes"] = dt_copy["Notes"].str.lower()

    # replace dbl with double
    dt_copy.replace(to_replace="dbl", value="double", inplace=True)
    
    # subset the notes with double in it.
    double_cropped = dt_copy[dt_copy["Notes"].str.contains("double")]

    # subset the notes without double in it.
    not_double_cropped = dt_copy[~dt_copy["Notes"].str.contains("double")]

    return (double_cropped, not_double_cropped)

def filter_double_by_Notes(dt_d):
    dt_copu = dt_d.copy()
    # convert NaN and NAs to string so we can subset/index 
    dt_copu[["Notes"]] = dt_copu[["Notes"]].astype(str)

    # convert to lower case
    dt_copu["Notes"] = dt_copu["Notes"].str.lower()

    # replace dbl with double
    dt_copu.replace(to_replace="dbl", value="double", inplace=True)
    
    # subset the notes with double in it.
    double_cropped = dt_copu[dt_copu["Notes"].str.contains("double")]

    return (double_cropped)

def form_xs_ys_from_peakdetect(max_peak_list, doy_vect):
    dd = np.array(doy_vect)
    xs = np.zeros(len(max_peak_list))
    ys = np.zeros(len(max_peak_list))
    for ii in range(len(max_peak_list)):  
        xs[ii] = dd[int(max_peak_list[ii][0])]
        ys[ii] = max_peak_list[ii][1]
    return (xs, ys)

def keep_WSDA_columns(dt_dt):
    needed_columns = ['ID', 'Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year']
    """
    # Using DataFrame.drop
    df.drop(df.columns[[1, 2]], axis=1, inplace=True)

    # drop by Name
    df1 = df1.drop(['B', 'C'], axis=1)
    """
    dt_dt = dt_dt[needed_columns]
    return dt_dt

def convert_TS_to_a_row(a_dt):
    a_dt = keep_WSDA_columns(a_dt)
    a_dt = a_dt.drop_duplicates()
    return(a_dt)

def save_matlab_matrix(filename, matDict):
    """
    Write a MATLAB-formatted matrix file given a dictionary of
    variables.
    """
    try:
        sio.savemat(filename, matDict)
    except:
        print("ERROR: could not write matrix file " + filename)

def separate_x_and_y(m_list):
    #
    #  input is a list whose elements are arrays of size 2: (DoY, peak)
    #  
    #  output: two vectors DoY = [d1, d2, ..., dn] and peaks[p1, p2, ..., pn]
    #
    DoY_vec = np.zeros(len(m_list))
    peaks_vec = np.zeros(len(m_list))
    counter = 0
    for entry in m_list:  
        DoY_vec[counter] = int(entry[0])
        peaks_vec[counter] = entry[1]
        counter += 1
    return (DoY_vec, peaks_vec)

def generate_peak_df(an_EE_TS):
    
    """
    input an_EE_TS is a file with several polygon 
          where for each polygon it includes the time series of NDVI.

    output: a dataframe that includes only the peak values and their corresponding
            DoY per field. It also includes the WSDA information.
    """
    an_EE_TS = initial_clean(an_EE_TS)

    ### List of unique polygons
    polygon_list = an_EE_TS['ID'].unique()

    output_columns = ['ID', 'Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year',
                      'peak_Doy', 'peak_value']
    # all_polygons_and_their_peaks = pd.DataFrame(data=None, 
    #                                             columns=output_columns)

    #
    # for each polygon assume there will be 3 peaks.
    # for memory allocation and speed up
    #
    all_polygons_and_their_peaks = pd.DataFrame(data=None, 
                                                index=np.arange(3*len(an_EE_TS)), 
                                                columns=output_columns)

    double_columns = ['ID', 'Acres', 'CovrCrp', 'CropGrp', 'CropTyp',
                      'DataSrc', 'ExctAcr', 'IntlSrD', 'Irrigtn', 'LstSrvD', 'Notes',
                      'RtCrpTy', 'Shap_Ar', 'Shp_Lng', 'TRS', 'county', 'year']

    double_polygons = pd.DataFrame(data=None, 
                                   index=np.arange(2*len(an_EE_TS)), 
                                   columns=double_columns)


    pointer = 0
    double_pointer = 0
    for a_poly in polygon_list:
        curr_field = an_EE_TS[an_EE_TS['ID']==a_poly]

        year = int(curr_field['year'].unique())
        plant = curr_field['CropTyp'].unique()[0]
        county = curr_field['county'].unique()[0]
        TRS = curr_field['TRS'].unique()[0]

        ### 
        ###  There is a chance that a polygon is repeated twice?
        ###

        X = curr_field['doy']
        y = curr_field['NDVI']
        freedom_df = 7
        #############################################
        ###
        ###             Smoothen
        ###
        #############################################

        # Generate spline basis with "freedom_df" degrees of freedom
        x_basis = cr(X, df=freedom_df, constraints='center')

        # Fit model to the data
        model = LinearRegression().fit(x_basis, y)

        # Get estimates
        y_hat = model.predict(x_basis)

        #############################################
        ###
        ###             find peaks
        ###
        #############################################
        # peaks_LWLS_1 = peakdetect(LWLS_1[:, 1], lookahead = 10, delta=0)
        # max_peaks = peaks_LWLS_1[0]
        # peaks_LWLS_1 = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

        peaks_spline = peakdetect(y_hat, lookahead = 10, delta=0)
        max_peaks =  peaks_spline[0]
        peaks_spline = form_xs_ys_from_peakdetect(max_peak_list = max_peaks, doy_vect=X)

        DoYs_series = pd.Series(peaks_spline[0])
        peaks_series = pd.Series(peaks_spline[1])

        peak_df = pd.DataFrame({ 
                           'peak_Doy': DoYs_series,
                           'peak_value': peaks_series
                          }) 


        WSDA_df = keep_WSDA_columns(curr_field)
        WSDA_df = WSDA_df.drop_duplicates()
        
        if (len(peak_df)>0):
            WSDA_df = pd.concat([WSDA_df]*peak_df.shape[0]).reset_index()
            # WSDA_df = pd.concat([WSDA_df, peak_df], axis=1, ignore_index=True)
            WSDA_df = WSDA_df.join(peak_df)
            if ("index" in WSDA_df.columns):
                WSDA_df = WSDA_df.drop(columns=['index'])

            # all_polygons_and_their_peaks = all_polygons_and_their_peaks.append(WSDA_df, sort=False)

            """
            copy the .values. Otherwise the index inconsistency between
            WSDA_df and all_poly... will prevent the copying.
            """
            all_polygons_and_their_peaks.iloc[pointer:(pointer + len(WSDA_df))] = WSDA_df.values

            if (len(WSDA_df) == 2):
                WSDA_df = WSDA_df.drop(columns=['peak_Doy', 'peak_value'])
                WSDA_df = WSDA_df.drop_duplicates()
                double_polygons.iloc[double_pointer:(double_pointer + len(WSDA_df))] = WSDA_df.values
                double_pointer += len(WSDA_df)

            pointer += len(WSDA_df)

            # to make sure the reference by address thing 
            # will not cause any problem.
        del(WSDA_df)


        """
        # first I decided to add all DoY and peaks in one row to avoid
        # multiple rows per (field, year)
        # However, in this way, each pair of (field, year)
        # can have different column sizes.
        # So, we cannot have one dataframe to include everything in it.
        # so, we will have to do dictionary to save out puts.
        # Lets just do replicates... easier to handle perhaps down the road.
        #
        DoY_colNames = [i + j for i, j in zip(\
                                              ["DoY_"]*(len(DoYs_series)+1), \
                                              [str(i) for i in range(1, len(DoYs_series)+1)] )] 

        peak_colNames = [i + j for i, j in zip(\
                                               ["peak_"]*(len(peaks_series)+1), \
                                               [str(i) for i in range(1, len(peaks_series)+1)] )]

        WSDA_df[DoY_colNames] = pd.DataFrame([DoYs_series], index=WSDA_df.index)
        WSDA_df[peak_colNames] = pd.DataFrame([peaks_series], index=WSDA_df.index)
        """

    """
    Instead of the following two we can do drop_duplicates()
    """
    all_polygons_and_their_peaks = all_polygons_and_their_peaks[0:(pointer+1)]
    double_polygons = double_polygons[0:(double_pointer+1)]
    return(all_polygons_and_their_peaks, double_polygons)
