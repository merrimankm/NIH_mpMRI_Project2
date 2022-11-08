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


class EPE_detector():
    def __init__(self):
        self.dicom_folder = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs'
        self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs\patient_list.csv'
        self.save_folder = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs'
        #self.dicom_folder = 'Mdrive_mount/MRIClinical/surgery_cases'
        #self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNifti_data/DCM_patients_list_short2.csv'
        #self.save_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project1Data/PatientNifti_data'
        self.patient_data = []
        self.lesion_data = []
        self.error_data = []
        self.wp_mask = np.empty([2,2])
        self.lesion_mask = np.empty([2,2])
        self.wp_array1 = []
        self.wp_array2 = []

    def checkEPE(self):
        '''
        create masks for all VOIs for all patients, save as .nii files and collect patient radiomics data from masks
        '''

        variance = []
        df_csv = pd.read_csv(self.csv_file, sep=',', header=0)

        # steps across patients
        for index, file_i in df_csv.iterrows():
            patient_id = str(file_i['MRN'])

            # check AI lesions for EPE
            wp_file = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP\Anonymized_NIfTIs_WP', patient_id + '_WP.nii')
            lesion_file = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_PI-RADS\Anonymized_NIfTIs_PI-RADS', patient_id + '_PI-RADS.nii')
            #wp_file1 = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\Contour_variance2', patient_id, 'nifti', 'wp_1_bt.nii')
            #wp_file2 = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\Contour_variance2', patient_id, 'nifti', 'wp_2_bt.nii')
            #wp_file3 = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\Contour_variance2', patient_id, 'nifti', 'wp_3_bt.nii')
            #wp_file4 = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\Contour_variance2', patient_id, 'nifti', 'wp_4_bt.nii')
            #wp_file5 = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\Contour_variance2', patient_id, 'nifti', 'wp_5_bt.nii')

            wp_img = sitk.ReadImage(wp_file)
            lesion_img = sitk.ReadImage(lesion_file)
            #wp_img1 = sitk.ReadImage(wp_file1)
            #wp_img2 = sitk.ReadImage(wp_file2)
            #wp_img3 = sitk.ReadImage(wp_file3)
            #wp_img4 = sitk.ReadImage(wp_file4)
            #wp_img5 = sitk.ReadImage(wp_file5)

            # 1 vs 2
            self.wp_array1 = sitk.GetArrayFromImage(lesion_img)
            self.wp_array2 = sitk.GetArrayFromImage(wp_img)
            [lesion_size, outside_voxels, max_distance] = self.findMaxDist()
            variance.append([patient_id, lesion_size, outside_voxels, max_distance])
            print(patient_id, lesion_size, outside_voxels, max_distance)



    def findMaxDist(self):
        wp1 = self.wp_array1.nonzero()
        a=len(wp1[0])
        outside = []

        b = 0
        for i in range(len(wp1[0])):
            if self.wp_array2[wp1[0][i], wp1[1][i], wp1[2][i]] == 0:
                outside.append([wp1[0][i], wp1[1][i], wp1[2][i]])
                b = b + 1
        epe = 0
        max_error = -1
        epe_list = []
        # for each pixel outside of wp:
        for i in range(len(outside)):
            min_dist = 0
            wp_points = []
            wp = self.wp_array2
            point1 = np.array((outside[i][0], outside[i][1], outside[i][2]))
            x_cent = outside[i][0]
            y_cent = outside[i][1]
            z_cent = outside[i][2]
            for z_off in range(-10, 11):
                for y_off in range(-10, 11):
                    for x_off in range(-1, 2):
                        x = x_cent + x_off
                        y = y_cent + y_off
                        z = z_cent + z_off
                        if x>-1:
                            if x<28:
                                if y>-1:
                                    if y<512:
                                        if z>-1:
                                            if z<512:
                                                if self.wp_array2[x][y][z]:
                                                    wp_points.append([x, y, z])
            if wp_points:
                #for j in range(len(wp_points)):
                #    point2 = wp_points[j]
                #    dist = np.linalg.norm(point1 - point2)
                #    if j==0:
                #        min_dist=dist
                #    if dist<min_dist:
                #        min_dist = dist
                #if min_dist>max_error:
                #    max_error = min_dist
                epe_list.append(0)
            else:
                epe_list.append(1)
                epe = 1
        variance_info = [a, b, epe]
        return(variance_info)


            # check manual lesions for EPE
            #wp_man_file = os.path.join(r'T:\MIP\Katie_Merriman\Project1Data\PatientNifti_data', MRN ,'NIfTI','T2.nii')
            #lesion_man_file = os.path.join(r'T:\MIP\Katie_Merriman\Project1Data\PatientNifti_data', MRN, 'NIfTI', '2_midline_left_apex_PZ_PIRADS_4_3_4_bt.nii')

            #wp_man_img = sitk.ReadImage(wp_man_file)
            #wp_man_array = sitk.GetArrayFromImage(wp_man_img)

            #lesion_man_img = sitk.ReadImage(lesion_man_file)
            #lesion_man_array = sitk.GetArrayFromImage(lesion_man_img)
            #lesion_man=lesion_man_array.nonzero()

            #b = 0
            #for i in range(len(lesion_man[0])):
            #    if wp_man_array[lesion_man[0][i], lesion_man[1][i], lesion_man[2][i]] == 0:
            #        b = b + 1






    def create_nifti_mask(self,pt_num='',patient_id='',wp_file='', lesion_file = ''):
        '''
        use simple itk to read in t2 nifti, create a mask, write with same properties
        '''

        #load t2 and adc niftis, create arrays
        t2_img_path = os.path.join('T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs', patient_id, 'T2.nii')
        t2_img = sitk.ReadImage(t2_img_path)
        img_array = sitk.GetArrayFromImage(t2_img)
        img_array = np.swapaxes(img_array, 2, 0)

        #create wp mask
        self.wp_mask = np.empty(img_array.shape)
        wp_dict = self.mask_coord_dict(patient_id=patient_id, file=wp_file, img_shape = (img_array.shape[0], img_array.shape[1]))
        if wp_dict:
            for key in wp_dict.keys():
                self.wp_mask[:,:,int(key)]=wp_dict[key]

        # create lesion mask
        self.lesion_mask = np.empty(img_array.shape)
        lesion_dict = self.mask_coord_dict(patient_id=patient_id, file=lesion_file,
                                       img_shape=(img_array.shape[0], img_array.shape[1]))
        if lesion_dict:
            for key in lesion_dict.keys():
                self.lesion_mask[:, :, int(key)] = lesion_dict[key]

        return

    def mask_coord_dict(self,patient_id='',file='',img_shape=()):
        '''
        creates a dictionary where keys are slice number and values are a mask (value 1) for area
        contained within .voi polygon segmentation
        :param patient_dir: root for directory to each patient
        :param type: types of file (wp,tz,urethra,PIRADS)
        :return: dictionary where keys are slice number, values are mask
        '''

        # define path to voi file
        #voi_path=os.path.join(self.dicom_folder, patient_id, file)
        voi_path = file

        #read in .voi file as pandas df
        pd_df = pd.read_fwf(voi_path)

        # use get_ROI_slice_loc to find location of each segment
        dict=self.get_ROI_slice_loc(voi_path)

        output_dict={}
        if dict:
            for slice in dict.keys():
                values=dict[slice]
                select_val=list(range(values[1],values[2]))
                specific_part=pd_df.iloc[select_val,:]
                split_df = specific_part.join(specific_part['MIPAV VOI FILE'].str.split(' ', 1, expand=True).rename(columns={0: "X", 1: "Y"})).drop(['MIPAV VOI FILE'], axis=1)
                X_coord=np.array(split_df['X'].tolist(),dtype=float).astype(int)
                Y_coord=np.array(split_df['Y'].tolist(),dtype=float).astype(int)
                mask=self.poly2mask(vertex_row_coords=X_coord, vertex_col_coords=Y_coord, shape=img_shape)
                output_dict[slice]=mask

        return(output_dict)

    def get_ROI_slice_loc(self,path):
        '''
        selects each slice number and the location of starting coord and end coord
        :return: dict of {slice number:(tuple of start location, end location)}

        '''

        pd_df=pd.read_fwf(path)

        #get the name of the file
        filename=path.split(os.sep)[-1].split('.')[0]

        #initialize empty list and empty dictionary
        slice_num_list=[]
        last_line=[]
        loc_dict={}

        #find the location of the last line -->
        for line in range(len(pd_df)):
            line_specific=pd_df.iloc[line,:]
            as_list=line_specific.str.split(r"\t")[0]
            if "# slice number" in as_list: #find location of all #slice numbers
                slice_num_list.append(line)
            if '# unique ID of the VOI' in as_list:
                last_line.append(line)

        if len(slice_num_list) < 1:
            return None
        else:
            for i in range(len(slice_num_list)):
                # for all values except the last value
                if i<(len(slice_num_list)-1):
                    loc=slice_num_list[i]
                    line_specific=pd_df.iloc[loc,:]
                    slice_num=line_specific.str.split(r"\t")[0][0]
                    start=slice_num_list[i]+3
                    end=slice_num_list[i+1]-1
                    loc_dict.update({slice_num:(filename,start,end)})

                #for the last value
                if i == (len(slice_num_list) - 1):
                    loc = slice_num_list[i]
                    line_specific=pd_df.iloc[loc,:]
                    slice_num=line_specific.str.split(r"\t")[0][0]
                    start=slice_num_list[i]+3
                    end=(last_line[0]-1)
                loc_dict.update({slice_num: (filename, start, end)})

        return (loc_dict)

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        ''''''
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=int)
        mask[fill_row_coords, fill_col_coords] = 1
        return mask

    def create_csv_files(self):
        nifti_cvsFileName = os.path.join(self.save_folder, 'PatientDataListTemporary.csv')
        niftiHeader = ['MRN', 'Number of Lesions', 'Min Mean ADC', 'Lesion with Min Mean ADC', 'Min Median ADC',
                       'Lesion with Min Median ADC', 'Min 10th Percentile ADC', 'Lesion with Min 10th Percentile ADC',
                       'Min ADC Entropy', 'Lesion with Min ADC Entropy', 'Max ADC Entropy',
                       'Lesion with Max ADC Entropy', 'Min T2 Entropy', 'Lesion with Min T2 Entropy', 'Max T2 Entropy',
                       'Lesion with Max T2 Entropy', 'Whole Prostate ADC Entropy', 'Whole Prostate T2 Entropy',
                       'Prostate Volume', 'PIRADS 5 Relative Volume', 'PIRADS 4 Relative Volume',
                       'PIRADS <=3 Relative Volume', 'Relative total volume']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.patient_data)

        nifti_cvsFileName = os.path.join(self.save_folder, 'VOI_ErrorsListTemporary.csv')
        niftiHeader = ['MRN', 'Error type']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.error_data)

        nifti_cvsFileName = os.path.join(self.save_folder, 'VOI_LesionsListTemporary.csv')
        niftiHeader = ['MRN', 'Lesion', 'ADC Mean', 'ADC Median', 'ADC 10th Percentile', 'ADC Entropy', 'T2 Entropy', 'Lesion Volume', 'Surface to Volume Ratio', 'Sphericity', 'Maximum 3D Diameter', 'Elongation', 'Flatness']
        with open(nifti_cvsFileName, 'w', newline="") as file2write:
            csvwriter = csv.writer(file2write)
            csvwriter.writerow(niftiHeader)
            csvwriter.writerows(self.lesion_data)
if __name__ == '__main__':
    c = EPE_detector()
    c.checkEPE()
#    c.create_csv_files()
    print('Check successful - 3')
