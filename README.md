# Towards faster seeding for k-means++ via lower bound and triangle inequality
This repository is the official implementation of [Towards faster seeding for k-means++ via lower bound and triangle inequality], submitted to Neurocomputing （Under Review））

## Usage

The specific algorithm is implemented in the kmeans_plus_plus.py file.

To evaluate the efficiency of the proposed method, run 'main.py'

The Parameter Sensitivity Analysis can be verified by changing the parameters used in 'parameter_sensivity.py'.

## The interface designed for the UCR time series.

download and read the UCR dataset(https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

In the main file, just set the file_name to the time series file name you want to test.

## General Example
Two datasets 'coil100'， 'Wafer' are provided. Just run 'main.py'

**Requirements**: NumPy, scipy, matplotlib, math, cv2, pandas,PIL



