# calculate_highB.py
#   Created 4/26/22 by Katie Merriman
#   Searches through designated folder for DICOM files of all patients on csv list
#   Converts DICOMs to Nifti, determines DICOM type, and saves to designated folder
#   Resamples high B based on T2

# Requirements
#   pydicom
#   SimpleITK

# Inputs (changed in class definition)
#   1: csv file, including path.
#           Must include column of MRNs labeled as "MRN"
#                MRNs can be alone or in MRN_DateOfMRI format
#   2: path to folder containing patient folders with NIfTI files
#   example usage:
#   python calculate_highB.py

# Outputs
#   Nifti files saved as T2, ADC, or highB in folders created for each patient within designated save folder


import os
#import sys
import numpy
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
#import csv
import glob
import pandas as pd
#import math
import numpy as np


class calculate_highb():
    def __init__(self):

        ### personal directory mapping
        self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\Patient_list_directories_short2.csv'
        self.patient_folder = r'T:\MIP\Katie_Merriman\Project1Data\PatientNifti_data'

        ### lambda desktop directory mapping
        #self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Patient_list_directories_short2.csv'
        #self.patient_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNifti_data'

        ### choose high b value
        self.highBvalue = 1500


    def calculateAll(self):
        patient = []

        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['MRN']))

            patientPath = glob.glob(os.path.join(self.patient_folder, p + '*'))
            if patientPath:
                patientPath = patientPath[0]
                patientID = os.path.basename(patientPath)
                patientPath = os.path.join(patientPath, 'NIfTI')
                patient.append([patientID, patientPath])

        for i in range(0, len(patient)):
            self.calculateB(patient[i])


    def calculateB(self, patient):
        patientB = os.path.join(patient[1], 'b0_Resampled.nii')
        patientADC = os.path.join(patient[1], 'ADC.nii')

        print("reading high b for", patient[0])
        b0_img = sitk.ReadImage(patientB)
        b0 = sitk.GetArrayFromImage(b0_img)
        b0 = b0.astype(float)
        ADC_img = sitk.ReadImage(patientADC)
        ADC = sitk.GetArrayFromImage(ADC_img)
        ADC = ADC.astype(float)
        ADC = ADC*1e-6
        highB = np.multiply(b0, np.exp(-1 * self.highBvalue * ADC))
        highBimg = sitk.GetImageFromArray(highB)


        # make sure all header info matches
        highBimg.CopyInformation(b0_img)
        for meta_elem in b0_img.GetMetaDataKeys():
            highBimg.SetMetaData(meta_elem, b0_img.GetMetaData(meta_elem))

        sitk.WriteImage(highBimg,os.path.join(patient[1],'highB_resampled_calculated.nii.gz'))


if __name__ == '__main__':
    c = calculate_highb()
    c.calculateAll()

