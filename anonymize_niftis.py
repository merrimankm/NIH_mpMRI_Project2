# anonymize_niftis.py
#   Created 5/4/22 by Katie Merriman
#   Searches through designated folder for NIfTI files for T2, ADC, and highB with specific names
#   Saves files to new folder with anonymized naming convention
#   Creates .csv of MRNs corresponding to anonymized names
#   Creates .csv doc of any patients in list for whom the code couldn't save all 3 file types



# Requirements
#   pydicom
#   Pandas

# Inputs
#   1: csv file, including path.
#           Must include column of MRNs labeled as "MRN"  (MRNs must be in MRN_DateOfMRI format)
#   2: path to directory containing patient subfolders with NIfTIs
#           Csv files will also be saved to this directory
#   3. path to directory where anonymized subfolders will be saved
#   example usage:
#   python anonymize_nifti.py "/path/to/cvsFileName.csv" "/path/to/patient/folder"

# Outputs
#   Nifti files saved into anonymized subfolders
#   .csv file of MRNs corresponding to anonymized subfolders
#   .csv file of error list


import os
import os.path
from os import path
import pandas as pd
import SimpleITK as sitk
import csv


class anonymizer():
    def __init__(self):
        #self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\Patient_list2.csv'
        #self.patientFolder = r'T:\MIP\Katie_Merriman\Project1Data\PatientNifti_data'
        #self.saveFolder = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs'

       ### lambda desktop directory mapping
        self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Patient_list.csv'
        self.patientFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNifti_data'
        self.saveFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Anonymized_NIfTIs'


    def anonymize(self):
        anonymized = []
        errors = []

        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        patient = []
        endFolder = []
        for rows, file_i in df_csv.iterrows():
            p = (str(file_i['MRN with date']))
            p2 = os.path.join(self.patientFolder, p, "NIfTI")
            patient.append([p,p2])

        for i in range(0,len(patient)):
            print('Anonymizing', patient[i][0])
            anonName = str(i+1)
            if len(anonName)<2:
                anonName = '0'+anonName
            if len(anonName)<3:
                anonName = '0'+anonName
            anonName = 'SURG-'+anonName
            anonFolder = os.path.join(self.saveFolder, anonName)
            os.makedirs(anonFolder)

            T2 = os.path.join(patient[i][1], 'T2.nii')
            ADC = os.path.join(patient[i][1], 'ADC.nii')
            B_ext = os.path.join(patient[i][1], 'highB_resampled.nii')
            B_calc = os.path.join(patient[i][1], 'highB_resampled_calculated.nii')
            B_calc_old = os.path.join(patient[i][1], 'highB_calculated_resampled.nii')

            try:
                T2_img = sitk.ReadImage(T2)
                sitk.WriteImage(T2_img, os.path.join(anonFolder, 'T2.nii.gz'))
                try:
                    ADC_img = sitk.ReadImage(ADC)
                    sitk.WriteImage(ADC_img, os.path.join(anonFolder, 'ADC.nii.gz'))
                    try:
                        highB_img = sitk.ReadImage(B_calc)
                        sitk.WriteImage(highB_img, os.path.join(anonFolder, 'highB.nii.gz'))
                        anonymized.append([patient[i][0], anonFolder])
                    except RuntimeError:
                        try:
                            highB_img = sitk.ReadImage(B_calc_old)
                            errors.append([patient[i][0], 'highB not calculated correctly'])
                        except RuntimeError:
                            try:
                                highB_img = sitk.ReadImage(B_ext)
                                sitk.WriteImage(highB_img, os.path.join(anonFolder, 'highB.nii.gz'))
                                anonymized.append([patient[i][0], anonFolder])
                            except RuntimeError:
                                errors.append([patient[i][0], 'highB not found'])
                except RuntimeError:
                    errors.append([patient[i][0],'ADC not found'])
            except RuntimeError:
                errors.append([patient[i][0], 'T2 not found'])


        #### write .csv files ####

        anonymized_cvsFileName = os.path.join(self.patientFolder, 'MRNs_anonymized.csv')
        niftiHeader = ['MRN', 'Anonymized folder']
        with open(anonymized_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(anonymized)

        errors_cvsFileName = os.path.join(self.patientFolder, 'Anonymization_errors.csv')
        niftiHeader = ['MRN']
        with open(errors_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(errors)

if __name__ == '__main__':
    c = anonymizer()
    c.anonymize()