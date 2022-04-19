# Dicom_searchConvert
#   Created 4/18/22 by Katie Merriman
#   Searches through designated folder for DICOM files of all patients on csv list
#   Converts DICOMs to Nifti, determines DICOM type, and saves to designated folder
#   Resamples high B based on T2

# Requirements
#   pydicom
#   SimpleITK

# Inputs
#   1: csv file, including path.
#           Must include column of MRNs labeled as "MRN" and
#               column with filepath to patient folder containing DICOMS labeled "topSourceFolder".
#           MRNs can be alone or in MRN_DateOfMRI format
#   2: path to folder containing patient folders with DICOM files
#   3: path to folder where converted files and csv reports should be saved
#   example usage:
#   python Dicom_searchConvert.py "/path/to/cvsFileName.csv" "/path/to/folder/of/patient/folders" "/path/to/save/folder"

# Outputs
#   Nifti files saved as T2, ADC, or highB in folders created for each patient within designated save folder


import os
#import sys
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk
#import csv
#import glob
import pandas as pd


class resample_highb():
    def __init__(self):
        #self.dicom_folder = r'T:\MIP\Katie_Merriman\surgery_cases_samplesTEMP\VOI_test'
        #self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\Patient_list_directories.csv'
        #self.save_folder = r'T:\MIP\Katie_Merriman\surgery_cases_samplesTEMP\ConvertedVOIs'

        self.csv_file = r"T:\MIP\Katie_Merriman\Project2Data\Patient_list_directories_short.csv"
        #self.sourceFolder = r"T:\MRIClinical\surgery_cases"
        self.patientFolder = r"T:\MIP\Katie_Merriman\Project1Data\PatientNifti_data"


    def resampleAll(self):
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        patient = []
        endFolder = []
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['topSourceFolder']))
            p2 = os.path.join(self.patientFolder, os.path.basename(p), "NIfTI")
            patient.append([p,p2,"."])

        for i in range(0,len(patient)):
            patient[i][2] = self.readDicoms(patient[i])

    def readDicoms(self, p):
        success = 0
        dicomFoldersList = []
        dicomSeriesList = []
        dicomProtocolList = []

        for root, dirs, files in os.walk(p[0]):
            for name in files:
                filePath = os.path.join(root, name)
                try:
                    ds = dcmread(filePath)
                except IOError:
                    # print(f'No such file')
                    continue
                except InvalidDicomError:
                    # print(f'Invalid Dicom file')
                    continue
                if name != ("DICOMDIR" or "LOCKFILE" or "VERSION"):
                    dicomString = filePath[:-(len(name) + 1)]
                    if dicomString not in dicomFoldersList:
                        if 'delete' in dicomString:
                             break
                        dicomFoldersList.append(dicomString)
                        dicomSeries = ds.ProtocolName.replace('/', '-')
                        dicomSeries = dicomSeries.replace(" ", "_")
                        # print(f'DICOM found...')
                        ADClist = ['Apparent Diffusion Coefficient', 'adc', 'ADC', 'dWIP', 'dSSh', 'dReg']
                        if ('T2' in dicomSeries or 't2' in dicomSeries):
                            dicomSeriesType = 'T2'
                        elif (any([substring in dicomString for substring in ADClist])) or (
                        any([substring in dicomSeries for substring in ADClist])):
                            dicomSeriesType = 'ADC'
                        else:
                            series_descript = ds.SeriesDescription
                            if any([substring in series_descript for substring in ADClist]) or (
                                    dicomString.endswith('ADC') or dicomString.endswith('adc')):
                                dicomSeriesType = 'ADC'
                            elif ('T2' in series_descript or 't2' in series_descript) or (
                                    dicomString.endswith('T2') or dicomString.endswith('t2')):
                                dicomSeriesType = 'T2'
                            elif (dicomString.endswith('DCE') or dicomString.endswith('dce')):
                                dicomSeriesType = 'DCE'
                            else:
                                dicomSeriesType = 'highB'
                        dicomSeriesList.append(dicomSeries)
                        dicomProtocolList.append(dicomSeriesType)

        if dicomFoldersList:
            print("Resampling HighB")
            success = self.resample(dicomFoldersList, dicomSeriesList, dicomProtocolList, p)

        return success

    def resample(self, dicomFoldersList, dicomSeriesList, dicomProtocolList, p):
        success = 0
        highB_img = []
        for refPath, DCMseries, dicomProtocol in zip(dicomFoldersList, dicomSeriesList, dicomProtocolList):

            # read original dicom
            reader = sitk.ImageSeriesReader()
            # print(f'Reading images...')
            dicom_names = reader.GetGDCMSeriesFileNames(refPath)

            # convert to NIfTI
            reader.SetFileNames(dicom_names)
            # print(f'Converting files...')
            if dicomProtocol == 'highB':
                highB_img = reader.Execute()
                niihighB = os.path.join(p[1], "highB_Resampled.nii.gz")
                # print(f'Converting ADC image')
            else:
                if dicomProtocol == 'T2':
                    t2_img = reader.Execute()


        if highB_img:
            # Resample highB
            # set the resample filter based on T2 header info
            Filter = sitk.ResampleImageFilter()
            Filter.SetReferenceImage(t2_img)
            Filter.SetOutputDirection(t2_img.GetDirection())
            Filter.SetOutputOrigin(t2_img.GetOrigin())
            Filter.SetOutputSpacing(t2_img.GetSpacing())

            # execute
            highB_resamp = Filter.Execute(highB_img)
            highB_resamp.CopyInformation(t2_img)

            # make sure all header info matches
            for meta_elem in t2_img.GetMetaDataKeys():
                highB_resamp.SetMetaData(meta_elem, t2_img.GetMetaData(meta_elem))

            sitk.WriteImage(highB_resamp, niihighB)
            success = 1
        return success


if __name__ == '__main__':
    c = resample_highb()
    c.resampleAll()

