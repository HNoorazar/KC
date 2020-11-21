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
    
    SOS_candidates = pandaFrame[colName] - onset_thresh
    EOS_candidates = offset_thresh - pandaFrame[colName]

    BOS, EOS = find_signChange_locs_DifferentOnOffset(SOS_candidates, EOS_candidates)
    pandaFrame['SOS'] = BOS * pandaFrame[VegIdx]
    pandaFrame['EOS'] = EOS * pandaFrame[VegIdx]

    return(pandaFrame)

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

def initial_clean(df, column_to_be_cleaned):
    dt_copy = df.copy()
    # remove the useles system:index column
    if ("system:index" in list(dt_copy.columns)):
        dt_copy = dt_copy.drop(columns=['system:index'])
    
    # Drop rows whith NA in column_to_be_cleaned column.
    dt_copy = dt_copy[dt_copy[column_to_be_cleaned].notna()]

    if (column_to_be_cleaned in ["NDVI", "EVI"]):
        #
        # 1.5 and -1.5 are just indicators for values that have violated the boundaries.
        #
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
          In each time window pick the maximum of present values

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


