# VOI_convert_w_lesion_features

# Requirements
#   SimpleITK
#   pandas
#   scikit-image
#   numpy
#   pidicom
#   dicom2nifti
#


import SimpleITK as sitk
import pandas as pd
from skimage import draw
import numpy as np
import pydicom
import math
import nibabel
import re
import dicom2nifti
import shutil
import csv
import os
import glob
import radiomics
from radiomics import featureextractor, imageoperations, firstorder
import six
np.set_printoptions(threshold=np.inf)


class VOIdilation():
    def __init__(self):

        local  = 0

        if local:
            self.dicom_folder = r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP\Anonymized_NIfTIs_WP'
            self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\dilation_list.csv'
            self.save_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data'
        else:
            self.mask_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/NVIDIA_output/Anonymized_NIfTIs_WP/Anonymized_NIfTIs_WP'
            self.dicom_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Anonymized_NIfTIs'
            self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/dilation_list.csv'
            self.save_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/DilatedProstate_data'

       

    def checkVariance(self):
        '''
        create masks for all VOIs for all patients, save as .nii files and collect patient radiomics data from masks
        '''
        #a = 2
        #b = 14
        a = 4
        b = 28

        variance = []
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)
        #patients = ['1966157_20111213', '4369518_20080411', '7394433_20150529']
        # steps across patients
        for index, file_i in df_csv.iterrows():
            patient_id = str(file_i['Anon'])

        #for patient_id in patients:

            # if does not exist:
            path = os.path.join(self.save_folder, patient_id)
            if not os.path.exists(path):
                os.mkdir(path)
            prostPath = os.path.join(self.mask_folder, patient_id+'_WP.nii')
            self.dilateProst(patient_id, prostPath, a, b)

            '''
            # create and save images of prostate, with expansion of 'a' pixels in z direction
            # and 'b' pixels in x/y direction
            print(patient_id, '1')
            wp_file1 = os.path.join(self.dicom_folder, patient_id, 'nifti', 'wp_1_bt.nii')
            self.dilateProst(wp_file1, a, b)
            print(patient_id, '2')
            wp_file2 = os.path.join(self.dicom_folder, patient_id, 'nifti', 'wp_2_bt.nii')
            self.dilateProst(wp_file2,  a, b)
            print(patient_id, '3')
            wp_file3 = os.path.join(self.dicom_folder, patient_id, 'nifti', 'wp_3_bt.nii')
            self.dilateProst(wp_file3,  a, b)
            print(patient_id, '4')
            wp_file4 = os.path.join(self.dicom_folder, patient_id, 'nifti', 'wp_4_bt.nii')
            self.dilateProst(wp_file4,  a, b)
            print(patient_id, '5')
            wp_file5 = os.path.join(self.dicom_folder, patient_id, 'nifti', 'wp_5_bt.nii')
            self.dilateProst(wp_file5,  a, b)
            '''

        return


    def dilateProst(self, patient_id, imgpath, z, xy):
        prost = sitk.ReadImage(imgpath)
        prostArr = sitk.GetArrayFromImage(prost)
        dilatedArr = sitk.GetArrayFromImage(prost) # makes copy with original data
        prostNZ = prostArr.nonzero() # saved as tuple in z,y,x order

        arr_size = prost.GetSize()
        sizeX = arr_size[0]
        sizeY = arr_size[1]
        sizeZ = arr_size[2]

        prostEdge = []
        # find array of x,y,z tuples corresponding to voxels of prostNZ that are on edge of prostate array
        # and also adjacent to lesion voxels outside of prostate
        for prostVoxel in range(len(prostNZ[0])):
            for slice in range(z):
                # changes voxels in slices above and below selected voxel from 0 to 1, z times
                    # skips continuing dilation if index is out of bounds
                if not prostNZ[0][prostVoxel]-slice < 0:
                        dilatedArr[prostNZ[0][prostVoxel]-slice, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                if prostNZ[0][prostVoxel]+slice < sizeZ: # less than, in order to account for 0 index
                    dilatedArr[prostNZ[0][prostVoxel]+slice, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            for xyvox in range(xy):
                # changes voxel left/right of and superior/inferior to selected voxel from 0 to 1, xy times
                if not prostNZ[1][prostVoxel] -xyvox < 0:
                    dilatedArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]-xyvox, prostNZ[2][prostVoxel]] = 1
                if prostNZ[1][prostVoxel] + xyvox < sizeY:
                    dilatedArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]+xyvox, prostNZ[2][prostVoxel]] = 1
                if not prostNZ[2][prostVoxel] - xyvox < 0:
                    dilatedArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]-xyvox] = 1
                if prostNZ[2][prostVoxel] + xyvox < sizeX:
                    dilatedArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]+xyvox] = 1


        #newname = imgpath[:-4] + '_dilated.nii.gz'
        #newname2 = imgpath[:-4] + '_dilated.mhd'
        newname = os.path.join(self.save_folder, patient_id, 'wp_bt.nii.gz')
        dilatedMask = sitk.GetImageFromArray(dilatedArr)
        dilatedMask.CopyInformation(prost)
        sitk.WriteImage(dilatedMask, newname)
        #sitk.WriteImage(dilatedMask, newname2)

        T2path = os.path.join(self.dicom_folder, patient_id, 'T2.nii')
        T2img = sitk.ReadImage(T2path)
        T2name = os.path.join(self.save_folder, patient_id, 't2.mhd')
        sitk.WriteImage(T2img, T2name)

        ADCpath = os.path.join(self.dicom_folder, patient_id, 'ADC.nii')
        ADCimg = sitk.ReadImage(ADCpath)
        ADCname = os.path.join(self.save_folder, patient_id, 'adc.mhd')
        sitk.WriteImage(ADCimg, ADCname)

        highBpath = os.path.join(self.dicom_folder, patient_id, 'highB.nii')
        highBimg = sitk.ReadImage(highBpath)
        highBname = os.path.join(self.save_folder, patient_id, 'highb.mhd')
        sitk.WriteImage(highBimg, highBname)


        return



if __name__ == '__main__':
    c = VOIdilation()
    c.checkVariance()
#    c.create_csv_files()
    print('Check successful')
