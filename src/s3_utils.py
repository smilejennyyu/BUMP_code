import numpy as np
import pandas as pd
import boto3
import json
import os
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date

def get_session():
    return boto3.session.Session()

# Get matching s3 keys
def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    session = get_session()
    s3_client = session.client('s3', region_name='us-west-2')
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    keys = []
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        if 'Contents' not in resp:
            break
        for obj in resp['Contents']:
            key = obj['Key']
            if key.endswith(suffix):
                keys.append(key)

        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break
    return keys
    
# Download locally
def download_from_s3(bucket, key, localpath):
    session = get_session()
    s3_resource = session.resource('s3', region_name='us-west-2')
    s3_resource.Bucket(bucket).download_file(key, localpath)

# Read s3 csv into pandas
def pandas_from_csv_s3(bucket, key, compression=None, credentials=None):
    session = get_session()
    s3_client = session.client('s3', region_name='us-west-2')
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    if compression:
        df = pd.read_csv(io.BytesIO(obj['Body'].read()), compression=compression)
    else:
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df