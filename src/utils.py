#%%
import pandas as pd
import numpy as np
# from . import s3_utils.pandas_from_csv_s3
from .s3_utils import *

def zScore(X, dim=0):
    """
    z-score normalize time series. This method works even when nan values are included

    Inputs
    ------
    X : a time series stored as a numpy array or pandas dataframe
        X is a time series, typically of shape (timesteps, variables).
    dim : int
        Denotes which variable is the time dimension. If the input series are of shape (timesteps, variables), dim should be 0.
        If the shape is (variables, timesteps) it should be 1. The same applies for more dimensions: If the same is (timesteps,
        participants, variables), for example, the dim should be 0.

    Returns
    -------
    This function returns a new time series of the same shape as the input, but for which the mean of each variable is 0 and
    its standard deviation is 1.
    """

    # --- if X is a pandas dataframe, change it to a numpy array ---
    if isinstance(X, pd.core.frame.DataFrame):
        X = X.to_numpy()

    # --- subtract mean and divide by standard deviation ---
    # Add tiny value to denominator to avoid dividing by 0
    return (X - np.nanmean(X, dim, keepdims=True)) / (np.nanstd(X, dim, keepdims=True) + 1e-7)

def moving_average(X, w):
    """
    Smooth an input time series by replacing values with the average of their neighbors

    Inputs
    ------
    X : a time series stored as a numpy array
        X is a time series of shape (timesteps, variables).
    w : int
        The window size for timesteps surrounding each value to compute the average
    """
    return np.convolve(X, np.ones(w), 'valid') / w

def data_load(data_keys={'oura_sleep', 'birth'}, wave=4):
    """
    Load data from bump s3 bucket

    Inputs
    ------
    data_key : set
               data_key is a set of datasets that need to be loaded
    wave:      int
               wave determines which wave of data that needs to be loaded. (e.g. 1, 2, 3, 4 etc.) 
               Max = 4
    
    return 
    ------
    dfs:       dict
               This function returns a dictionary of requested dataset where the key is the name
               of the key, and the value is pandas DataFrame.
    """

    #List s3 keys
    # get_matching_s3_keys(bucket, prefix='bump/') #Remove prefix to view non-BUMP data
    bucket = 'fouryouandme-study-data' # for 4YouandMe, Bump, CamCog or Bodyport project

    # prefix = {study_name} or {study_name}/{source}
    # sources: app_activities, bodyport, camcog, garmin, oura, redcap, rescuetime
    # note camcog not accessible to bodyport (and vice a versa)
    key = f'bump/redcap/wave_{wave}/study_ids.csv.gz'
    df_studyID = pandas_from_csv_s3(bucket, key=key, compression='gzip')

    # put all data in this merged dfs list
    dfs = {}

    # Some dataframes use 'record_id' instead of 'user_id'. 
    # You'll need to match it up with df_studyID where'evidation_id' is 'user_id'
    # NOTE: Very few examples of this. Birthing data is the important one

    # Birthing Data
    if 'birth' in data_keys:
        key = f'bump/redcap/wave_{wave}/birthing_data_cohort_2_only.csv.gz'
        df_birth = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_birth['date'] = pd.to_datetime(df_birth.birth_date).dt.date
        df_birth = pd.merge(df_birth, df_studyID, on='record_id')
        df_birth['user_id'] = df_birth.evidation_id

        # There is a missing value in the birth_date or birth_scheduled, remove them
        df_birth = df_birth.dropna(subset=['birth_date', 'birth_scheduled']).reset_index(drop=True)
        dfs['birth'] = df_birth

    # Bodyport
    if 'bodyport' in data_keys:
        key = f'bump/bodyport/wave_{wave}/bodyport.csv.gz'
        df_bodyport = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        # OPTIONAL: Convert date format
        df_bodyport['date'] = pd.to_datetime(df_bodyport.event_date).dt.date 
        dfs['bodyport'] = df_bodyport

    # OURA Data
    if 'oura_sleep' in data_keys:
        key = f'bump/oura/wave_{wave}/oura_sleep.csv.gz'
        df_sleep = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_sleep['date'] = pd.to_datetime(df_sleep.event_date).dt.date
        dfs['oura_sleep'] = df_sleep
    
    if 'oura_activity' in data_keys:
        key = f'bump/oura/wave_{wave}/oura_activity.csv.gz'
        df_activity = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_activity['date'] = pd.to_datetime(df_activity.event_date).dt.date
        dfs['oura_activity'] = df_activity

    if 'oura_readiness' in data_keys:
        key = f'bump/oura/wave_{wave}/oura_readiness.csv.gz'
        df_readiness = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_readiness['date'] = pd.to_datetime(df_readiness.event_date).dt.date
        dfs['oura_readiness'] = df_readiness

    # Cambridge Cognition Data
    if 'camcog_EBT' in data_keys:
        key = f'bump/camcog/wave_{wave}/EBT.csv.gz'
        df_EBT = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_EBT['date'] = pd.to_datetime(df_EBT.event_date).dt.date
        dfs['camcog_EBT'] = df_EBT

    if 'camcog_NBX' in data_keys:
        key = f'bump/camcog/wave_{wave}/NBX.csv.gz'
        df_NBX = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        # df_NBX['date'] = pd.to_datetime(df_NBX.event_date).dt.date
        dfs['camcog_NBX'] = df_NBX

    if 'camcog_PVT' in data_keys:
        key = f'bump/camcog/wave_{wave}/PVT.csv.gz'
        df_PVT = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        # df_PVT['date'] = pd.to_datetime(df_PVT.event_date).dt.date
        dfs['camcog_PVT'] = df_PVT
    
    # Garmin Data
    if 'garmin_activities' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_activities.csv.gz'
        df_garmin_activities = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_activities['date'] = pd.to_datetime(df_garmin_activities.event_date).dt.date
        dfs['garmin_activities'] = df_garmin_activities
    
    if 'garmin_activity_details' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_activity_details.csv.gz'
        df_garmin_activity_details = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_activity_details['date'] = pd.to_datetime(df_garmin_activity_details.event_date).dt.date
        dfs['garmin_activity_details'] = df_garmin_activity_details
    
    if 'garmin_dailies' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_dailies.csv.gz'
        df_garmin_dailies = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_dailies['date'] = pd.to_datetime(df_garmin_dailies.event_date).dt.date
        dfs['garmin_dailies'] = df_garmin_dailies
    
    if 'garmin_epochs' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_epochs.csv.gz'
        df_garmin_epochs = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_epochs['date'] = pd.to_datetime(df_garmin_epochs.event_date).dt.date
        dfs['garmin_epochs'] = df_garmin_epochs
    
    if 'garmin_moveiq' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_moveiq.csv.gz'
        df_garmin_moveiq = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_moveiq['date'] = pd.to_datetime(df_garmin_moveiq.event_date).dt.date
        dfs['garmin_moveiq'] = df_garmin_moveiq
    
    if 'garmin_pulse_ox' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_pulse_ox.csv.gz'
        df_garmin_pulse_ox = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_pulse_ox['date'] = pd.to_datetime(df_garmin_pulse_ox.event_date).dt.date
        dfs['garmin_pulse_ox'] = df_garmin_pulse_ox

    if 'garmin_respiration' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_respiration.csv.gz'
        df_garmin_respiration = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_respiration['date'] = pd.to_datetime(df_garmin_respiration.event_date).dt.date
        dfs['garmin_respiration'] = df_garmin_respiration
    
    if 'garmin_sleep' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_sleep.csv.gz'
        df_garmin_sleep = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_sleep['date'] = pd.to_datetime(df_garmin_sleep.event_date).dt.date
        dfs['garmin_sleep'] = df_garmin_sleep

    if 'garmin_stress_details' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_stress_details.csv.gz'
        df_garmin_stress_details = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_stress_details['date'] = pd.to_datetime(df_garmin_stress_details.event_date).dt.date
        dfs['garmin_stress_details'] = df_garmin_stress_details

    if 'garmin_user_metrics' in data_keys:
        key = f'bump/garmin/wave_{wave}/garmin_user_metrics.csv.gz'
        df_garmin_user_metrics = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_garmin_user_metrics['date'] = pd.to_datetime(df_garmin_user_metrics.event_date).dt.date
        dfs['garmin_user_metrics'] = df_garmin_user_metrics

    # Survey
    if 'surveys' in data_keys:
        key = f'bump/app_activities/wave_{wave}/surveys.csv.gz'
        df_survey = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_survey['date'] = pd.to_datetime(df_survey.updated_at).dt.date
        dfs['surveys'] = df_survey

    # SAM survey
    if 'quick_activities' in data_keys:
        key = f'bump/app_activities/wave_{wave}/quick_activities.csv.gz'
        df_sam = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        df_sam['date'] = pd.to_datetime(df_sam.event_date).dt.date
        dfs['quick_activities'] = df_sam

    # Others
    if 'ace' in data_keys:
        key = f'bump/redcap/wave_{wave}/ace.csv.gz'
        df_ace = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['ace'] = df_ace
    if 'adverse_event_report' in data_keys:
        key = f'bump/redcap/wave_{wave}/adverse_event_report.csv.gz'
        df_adverse_event_report = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['adverse_event_report'] = df_adverse_event_report
    if 'birthing_data_cohort_2_only' in data_keys:
        key = f'bump/redcap/wave_{wave}/birthing_data_cohort_2_only.csv.gz'
        df_birthing_data_cohort_2_only = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['birthing_data_cohort_2_only'] = df_birthing_data_cohort_2_only
    if 'check_in_2_addendum' in data_keys:
        key = f'bump/redcap/wave_{wave}/check_in_2_addendum.csv.gz'
        df_check_in_2_addendum = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['check_in_2_addendum'] = df_check_in_2_addendum
    if 'check_in_adherence_log' in data_keys:
        key = f'bump/redcap/wave_{wave}/check_in_adherence_log.csv.gz'
        df_check_in_adherence_log = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['check_in_adherence_log'] = df_check_in_adherence_log
    if 'cssrs' in data_keys:
        key = f'bump/redcap/wave_{wave}/cssrs.csv.gz'
        df_cssrs = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['cssrs'] = df_cssrs
    if 'enrollment_status' in data_keys:
        key = f'bump/redcap/wave_{wave}/enrollment_status.csv.gz'
        df_enrollment_status = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['enrollment_status'] = df_enrollment_status
    if 'generalized_anxiety_disorder_scale_gad7' in data_keys:
        key = f'bump/redcap/wave_{wave}/generalized_anxiety_disorder_scale_gad7.csv.gz'
        df_generalized_anxiety_disorder_scale_gad7 = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['generalized_anxiety_disorder_scale_gad7'] = df_generalized_anxiety_disorder_scale_gad7
    if 'medications' in data_keys:
        key = f'bump/redcap/wave_{wave}/medications.csv.gz'
        df_medications = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['medications'] = df_medications
    if 'metadata' in data_keys:
        key = f'bump/redcap/wave_{wave}/metadata.csv.gz'
        df_metadata = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['metadata'] = df_metadata
    if 'monthly_points_log' in data_keys:
        key = f'bump/redcap/wave_{wave}/monthly_points_log.csv.gz'
        df_monthly_points_log = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['monthly_points_log'] = df_monthly_points_log
    if 'personal_characteristics' in data_keys:
        key = f'bump/redcap/wave_{wave}/personal_characteristics.csv.gz'
        df_personal_characteristics = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['personal_characteristics'] = df_personal_characteristics
    if 'phq9' in data_keys:
        key = f'bump/redcap/wave_{wave}/phq9.csv.gz'
        df_phq9 = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['phq9'] = df_phq9
    if 'pitting_edema_cohort_2_only' in data_keys:
        key = f'bump/redcap/wave_{wave}/pitting_edema_cohort_2_only.csv.gz'
        df_pitting_edema_cohort_2_only = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['pitting_edema_cohort_2_only'] = df_pitting_edema_cohort_2_only
    if 'risk_alert_trigger' in data_keys:
        key = f'bump/redcap/wave_{wave}/risk_alert_trigger.csv.gz'
        df_risk_alert_trigger = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['risk_alert_trigger'] = df_risk_alert_trigger
    if 'study_completion_form' in data_keys:
        key = f'bump/redcap/wave_{wave}/study_completion_form.csv.gz'
        df_study_completion_form = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['study_completion_form'] = df_study_completion_form
    if 'study_dates_cohort_info' in data_keys:
        key = f'bump/redcap/wave_{wave}/study_dates_cohort_info.csv.gz'
        df_study_dates_cohort_info = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['study_dates_cohort_info'] = df_study_dates_cohort_info
    if 'study_ids' in data_keys:
        key = f'bump/redcap/wave_{wave}/study_ids.csv.gz'
        df_study_ids = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['study_ids'] = df_study_ids
    if 'suicide_risk_response_completed_by_study_doctor' in data_keys:
        key = f'bump/redcap/wave_{wave}/suicide_risk_response_completed_by_study_doctor.csv.gz'
        df_suicide_risk_response_completed_by_study_doctor = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['suicide_risk_response_completed_by_study_doctor'] = df_suicide_risk_response_completed_by_study_doctor
    if 'symptom_tracking_permission' in data_keys:
        key = f'bump/redcap/wave_{wave}/symptom_tracking_permission.csv.gz'
        df_symptom_tracking_permission = pandas_from_csv_s3(bucket, key=key, compression='gzip')
        dfs['symptom_tracking_permission'] = df_symptom_tracking_permission
    
    # make sure all requested datasets are stored in dfs
    assert (len(dfs) == len(data_keys))
    return dfs

# %%
