import SimpleITK as sitk
import numpy as np
import csv
import os
from scipy.ndimage import label


class EPEdetector:
    def __init__(self):

        local = 1
        self.threshold = 0.35
        self.variance = 6 #maximum 3D distance observed as variance: 6.15379664144412

        if local:
            self.mask_folder = r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP' \
                               r'\Anonymized_NIfTIs_WP '
            self.dicom_folder = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs'
            self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\dilation_list.csv'
            self.save_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data'
            self.fileName1 = r'T:\MIP\Katie_Merriman\Project2Data\EPEresultsbylesion.csv'
            self.fileName2 = r'T:\MIP\Katie_Merriman\Project2Data\EPEresultsbypatient.csv'
        else:
            self.mask_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/NVIDIA_output/Anonymized_NIfTIs_WP' \
                               '/Anonymized_NIfTIs_WP '
            self.dicom_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Anonymized_NIfTIs'
            self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/dilation_list.csv'
            self.save_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/DilatedProstate_data'
            self.fileName1 = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/EPEresultsbylesion.csv'
            self.fileName2 = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/EPEresultsbypatient.csv'

    def determineEPE(self):
        #file = open(self.fileName, 'a+', newline='')
        #with file:
        #    write = csv.writer(file)
        #    headers = [['patient', 'lesion', 'EPE grade','distance from capsule', 'area outside of capsule']]
        #    write.writerows(headers)
        #file.close()

        EPEdata = []

        for p in range(51, 101):

            # patient name should follow format 'SURG-00X'
            patient = 'SURG-'+str(p+1000)[1:]
            print(patient)

            prost = sitk.ReadImage(os.path.join(
                r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP\Anonymized_NIfTIs_WP',
                patient + '_WP.nii'))
            prostArr = sitk.GetArrayFromImage(prost)
            prostEdge = self.createEdge(patient, prost, prostArr)

            prostVariance = sitk.ReadImage(os.path.join(
                r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data2', patient, 'wp_bt_variance.nii'))
            varArr = sitk.GetArrayFromImage(prostVariance)

            lesionHeatMap = sitk.ReadImage(os.path.join(
                r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data2', patient, r'output\lesion_prob.nii'))
            probArr = sitk.GetArrayFromImage(lesionHeatMap)

            lesionMask = sitk.ReadImage(os.path.join(
                r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data2', patient, r'output\lesion_mask.nii'))
            lesionArr = sitk.GetArrayFromImage(lesionMask)

            [EPElesionData, EPEpatientData] = self.EPEbyLesion(patient, lesionHeatMap, prostArr, prostEdge, varArr, probArr, lesionArr)

            EPEdata.append(EPElesionData)

            file = open(self.fileName1, 'a+', newline='')
            # writing the data into the file
            with file:
                write = csv.writer(file)
                write.writerows(EPElesionData)
            file.close()

            file = open(self.fileName2, 'a+', newline='')
            # writing the data into the file
            with file:
                write = csv.writer(file)
                write.writerows([EPEpatientData])
            file.close()

        return



    def createEdge(self, patient, prost, prostArr):

        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        arr_shape = prostArr.shape
        capsule = np.zeros(arr_shape, dtype=int)



        # find array of x,y,z tuples corresponding to voxels of prostNZ that are on edge of prostate array
        # and also adjacent to lesion voxels outside of prostate
        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            if (prostNZ[0][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel] - 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            elif (prostNZ[0][prostVoxel] + 1) < arr_shape[0]:
                if prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
            elif (prostNZ[1][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] - 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            elif (prostNZ[1][prostVoxel] + 1) < arr_shape[1]:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] + 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel to right or left of current voxel is 0, voxel is on the edge
            elif (prostNZ[2][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] - 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            elif (prostNZ[2][prostVoxel] + 1) < arr_shape[2]:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] + 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1

        #save edge mask to folder:
        newname = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\fullTest', patient + '_capsule.nii.gz')
        edgeMask = sitk.GetImageFromArray(capsule)
        edgeMask.CopyInformation(prost)
        sitk.WriteImage(edgeMask, newname)

        return capsule


    def EPEbyLesion(self, patient, lesionMask, prostArr, edgeArr, varArr, probArr, lesionArr):

        # create binary lesion array using threshold
        #binaryArrNZ = binaryArr.nonzero()
        #arr_size = probArr.shape
        #capsule = np.zeros(arr_size, dtype=int)

        # save binary lesion mask
        binaryArr = np.where(probArr > self.threshold, 1, 0)
        binaryname = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\fullTest', patient + '_allLesions.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionMask)
        sitk.WriteImage(binarymask, binaryname)

        # create labeled array separating individual lesions
        labeled_array, num_features = label(binaryArr)
        labeledname = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\fullTest', patient + '_lesions_labeled.nii.gz')
        labeledmask = sitk.GetImageFromArray(labeled_array)
        labeledmask.CopyInformation(lesionMask)
        sitk.WriteImage(labeledmask, labeledname)

        lesionEPEdata = []
        patientEPEdata = []
        edgeNZ = edgeArr.nonzero()
        spacing = lesionMask.GetSpacing()

        EPEmax = 0
        EPEscore = 0
        for i in range(num_features):
            val = i + 1
            probLesionArr = np.where(labeled_array == val, 1, 0)
            probLesionname = os.path.join(r'T:\MIP\Katie_Merriman\Project2Data\fullTest',
                                          patient + '_lesion' + str(val) + '.nii.gz')
            probLesionmask = sitk.GetImageFromArray(probLesionArr)
            probLesionmask.CopyInformation(lesionMask)
            sitk.WriteImage(probLesionmask, probLesionname)

            # check that lesionMask overlaps with probLesion array
            lesionNZ = probLesionArr.nonzero()
            excludeLesion = 1
            for ind in range(len(lesionNZ[0])):
                if lesionArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 1:
                    excludeLesion = 0
                    break

            # if lesion does match lesion on mask:
            if excludeLesion == 0:
                outsideVar = []
                outsideProst = []
                insideProst = []

                for ind in range(len(lesionNZ[0])):


                    # check if lesion is outside of prostate variance
                    if varArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 0:
                        outsideVar.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])
                    # else check if lesion is outside of prostate
                    elif prostArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 0:
                        outsideProst.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])
                    # else store lesion locations inside of prost
                    else:
                        insideProst.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])

                # if lesion outside of prostate variance:
                if len(outsideVar) >100:
                    distfromCapsule = 'outside variance'
                    EPEscore = 3
                    outsideArea = len(outsideVar)
                    lesionEPEdata.append([patient, 'lesion' + str(i + 1), EPEscore, distfromCapsule, outsideArea])
                elif len(outsideVar) != 0:
                    distfromCapsule = 'outside variance'
                    EPEscore = 2
                    outsideArea = len(outsideVar)
                    lesionEPEdata.append([patient, 'lesion' + str(i+1), EPEscore, distfromCapsule, outsideArea])
                # else if lesion inside variance but outside prostate:
                elif len(outsideProst) != 0:
                    outsideArea = len(outsideProst)
                    print('len outsideProst:', outsideArea)
                    distfromCapsule = 0
                    for vox in range(outsideArea):
                        min_dist = 256
                        if vox % 10 == 0:
                            print('vox', vox)
                        for prostVox in range(len(edgeNZ[0])):
                            dist = np.sqrt((spacing[2]*(outsideProst[vox][0] - edgeNZ[0][prostVox]))**2 +
                                           (spacing[1]*(outsideProst[vox][1] - edgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0]*(outsideProst[vox][2] - edgeNZ[2][prostVox])) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                    # if distance outside capsule at least 1/4 of variance OR if area > 200, EPEscore = 1
                    # if both conditions are true, EPE score = 2
                    if distfromCapsule > (self.variance/2):
                        if outsideArea > 200:
                            EPEscore = 2
                        else:
                            EPEscore = 1
                    elif outsideArea > 200:
                        EPEscore = 1
                    else:
                        EPEscore = 0
                    lesionEPEdata.append([patient, 'lesion' + str(i+1), EPEscore, distfromCapsule, outsideArea])
                # else, check how close prostate-confined lesion is to capsule:
                else:
                    '''
                    print('len insideProst:', len(insideProst))
                    distfromCapsule = 0
                    for vox in range(len(insideProst)):
                        if vox % 10 == 0:
                            print('vox', vox)
                        min_dist = 256
                        for prostVox in range(len(edgeNZ[0])):

                            dist = np.sqrt((spacing[2]*(insideProst[vox][0] - edgeNZ[0][prostVox])) ** 2
                                           + (spacing[1]*(insideProst[vox][1] - edgeNZ[1][prostVox])) ** 2
                                           + (spacing[0]*(insideProst[vox][2] - edgeNZ[2][prostVox])) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                    if distfromCapsule > (self.variance/4):
                        EPEscore = 0
                    else:
                        EPEscore = 1
                    '''
                    EPEscore = 0
                    outsideArea = 0
                    distfromCapsule = 'organ confined'
                    lesionEPEdata.append([patient, 'lesion' + str(i+1), EPEscore, distfromCapsule, outsideArea])



            if EPEscore > EPEmax:
                EPEmax = EPEscore
        lesionEPEdata.append([patient, 'all lesions', EPEmax])
        patientEPEdata = [patient, EPEmax]

        return [lesionEPEdata, patientEPEdata]



if __name__ == '__main__':
    c = EPEdetector()
    c.determineEPE()
#    c.create_csv_files()
    print('Check successful')
