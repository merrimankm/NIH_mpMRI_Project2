import SimpleITK as sitk
import numpy as np
import csv
import os
from scipy.ndimage import label


class EPEdetector:
    def __init__(self):

        local = 1
        self.threshold = 0.35 #threshold for lesion mask is 0.6344772701607316
        self.testNum = 'test1'
        self.variance = 6 #maximum 3D distance observed as variance: 6.15379664144412
        self.EPEVarThresh = 100
        self.EPE3AreaThresh = 300
        self.EPE1_2AreaThresh= 150


        if local:
            self.mask_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3'
            self.dicom_folder = r'T:\MIP\Katie_Merriman\Project2Data\Anonymized_NIfTIs'
            self.csv_file = r'T:\MIP\Katie_Merriman\Project2Data\MONAIdilation_list.csv'
            self.save_folder = r'T:\MIP\Katie_Merriman\Project2Data\MONAIDilatedProstate_data'
            self.fileName1 = r'T:\MIP\Katie_Merriman\Project2Data\EPEresultsbylesion_MONAI_wDist.csv'
            self.fileName2 = r'T:\MIP\Katie_Merriman\Project2Data\EPEresultsbypatient_MONAI_wDist.csv'
            self.prostFolder = r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP' \
                               r'\Anonymized_NIfTIs_WP'
        else:
            self.mask_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/DilatedProstate_data3'
            self.dicom_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/Anonymized_NIfTIs'
            self.csv_file = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/MONAIdilation_list.csv'
            self.save_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/MONAIDilatedProstate_data'
            self.fileName1 = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/EPEresultsbylesion_MONAI_wDist.csv'
            self.fileName2 = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/EPEresultsbypatient_MONAI_wDist.csv'
            self.prostFolder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/NVIDIA_output/Anonymized_NIfTIs_WP' \
                               '/Anonymized_NIfTIs_WP'
    def determineEPE(self):
        #file = open(self.fileName, 'a+', newline='')
        #with file:
        #    write = csv.writer(file)
        #    headers = [['patient', 'lesion', 'EPE grade','distance from capsule', 'area outside of capsule']]
        #    write.writerows(headers)
        #file.close()

        file = open(self.fileName2, 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([["patient", "EPEmax", "altEPEmax", "origOutsideAreaMax", "altOutsideAreaMax",
                              "TotalAreaMax", "asymmetry", "maxDist", "lesionAreaMax", "lesion_wMaxArea",
                              "lesionDistMax", "lesion_wMaxArea"]])

        file.close()

        EPEdata = []
        YML = [1, 9, 18, 19, 26, 29, 30, 33, 34, 35, 38, 42, 53, 56, 60, 75, 76, 77,
               88, 92, 98, 99, 108, 109, 114, 146, 151, 152, 156, 157, 166, 177, 181,
               198, 201, 204, 214, 220, 221, 224, 226, 238, 241, 246, 248, 249, 255,
               264, 266, 269, 273, 274, 279, 281, 303, 304, 320, 328, 330, 336, 345,
               346, 347, 371, 382, 385, 388, 400, 423, 425, 435, 458, 471, 476, 501,
               510, 514, 542, 555]

        noForRemote = [2, 7, 10, 21, 23, 43, 64, 74, 75, 76, 81, 84, 105, 107, 110, 111,
                       113, 121, 125, 131, 133, 146, 147, 149, 152, 158, 164, 170, 183,
                       185, 187, 189, 190, 205, 218, 219, 225, 231, 232, 269, 278, 291,
                       301, 307, 311, 312, 328, 329, 331, 332, 348, 349, 363, 368, 379,
                       385, 394, 413, 420, 465, 472, 481, 485, 508, 512, 516, 550]

        #capsuleTest = [8, 9, 10, 1, 2, 3, 4, 5, 6, 7]

        for p in noForRemote:
        #for p in range(1, 556):
            #skipping YML scored patients:

            #if p == 329:
            #if p in YML:
            #    continue

            # patient name should follow format 'SURG-00X'
            patient = 'SURG-'+str(p+1000)[1:]
            print(patient)

            try:
                prost = sitk.ReadImage(os.path.join(
                    self.prostFolder,
                    patient + '_WP.nii'))
                prostArr = sitk.GetArrayFromImage(prost)
                prostEdge = self.createEdge(patient, prost, prostArr,'_capsule.nii.gz')
                flippedProst = self.MaskFlip(patient, prost, prostArr)
                #flippedEdge = self.createEdge(patient, prost, flippedProst, '_flippedEdge.nii.gz')
                flippedEdge = []

                prostVariance = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'wp_bt_variance.nii'))
                varArr = sitk.GetArrayFromImage(prostVariance)

                lesionHeatMap = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_prob.nii'))
                probArr = sitk.GetArrayFromImage(lesionHeatMap)

                lesionMask = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_mask.nii'))
                lesionArr = sitk.GetArrayFromImage(lesionMask)

                [EPElesionData, EPEpatientData] = self.EPEbyLesion(patient, lesionHeatMap, prostArr, prostEdge, varArr,
                                                                   probArr, lesionArr, flippedProst, flippedEdge)

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
            except RuntimeError:
                print("remote error")
                file = open(self.fileName2, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, "remote error"]])
                file.close()
        return



    def createEdge(self, patient, prost, prostArr, suffix):
        # leaving this as function of EPEdetector to allow easy integration of self.savefolder later
        arr_shape = prostArr.shape
        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        capsule = np.zeros(arr_shape, dtype=int)



        # find array of x,y,z tuples corresponding to voxels of prostNZ that are on edge of prostate array
        # and also adjacent to lesion voxels outside of prostate
        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            if (prostNZ[0][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel] - 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[0][prostVoxel] + 1) < arr_shape[0]:
                if prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
            if (prostNZ[1][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] - 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[1][prostVoxel] + 1) < arr_shape[1]:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] + 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel to right or left of current voxel is 0, voxel is on the edge
            if (prostNZ[2][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] - 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[2][prostVoxel] + 1) < arr_shape[2]:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] + 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1

        #save edge mask to folder:

        EPEmaskfolder = os.path.join(self.mask_folder, patient, self.testNum)
        if not os.path.exists(EPEmaskfolder):
            os.mkdir(EPEmaskfolder)
        newname = os.path.join(EPEmaskfolder, patient + suffix)
        edgeMask = sitk.GetImageFromArray(capsule)
        edgeMask.CopyInformation(prost)
        sitk.WriteImage(edgeMask, newname)

        return capsule

    def MaskFlip(self, patient, prost, prostArr):
        arr_shape = prostArr.shape
        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        flippedProst = np.zeros(arr_shape, dtype=int)
        midline = int(round(sum(prostNZ[2]) / len(prostNZ[2])))
        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            flippedProst[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], (2 * midline - prostNZ[2][prostVoxel])] = 1

        FlippedMaskfolder = os.path.join(self.mask_folder, patient, self.testNum)
        newname = os.path.join(FlippedMaskfolder, patient + '_flippedMask.nii.gz')
        FlippedMask = sitk.GetImageFromArray(flippedProst)
        FlippedMask.CopyInformation(prost)
        sitk.WriteImage(FlippedMask, newname)

        return flippedProst

    def EPEbyLesion(self, patient, lesionMask, prostArr, edgeArr, varArr, probArr, lesionArr, flippedProst, flippedEdge):
        prostNZ = prostArr.nonzero()
        asymmetry = np.sum(flippedProst != prostArr) / len(prostNZ[0])

        # create binary lesion array using threshold
        #binaryArrNZ = binaryArr.nonzero()
        #arr_size = probArr.shape
        #capsule = np.zeros(arr_size, dtype=int)

        print("asymmetry =", asymmetry)
        EPEmaskfolder = os.path.join(self.mask_folder, patient, self.testNum)


        # save binary lesion mask
        binaryArr = np.where(probArr > self.threshold, 1, 0)
        binaryname = os.path.join(EPEmaskfolder, patient + '_allLesions.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionMask)
        sitk.WriteImage(binarymask, binaryname)

        # create labeled array separating individual lesions
        labeled_array, num_features = label(binaryArr)
        labeledname = os.path.join(EPEmaskfolder, patient + '_lesions_labeled.nii.gz')
        labeledmask = sitk.GetImageFromArray(labeled_array)
        labeledmask.CopyInformation(lesionMask)
        sitk.WriteImage(labeledmask, labeledname)

        lesionEPEdata = []
        patientEPEdata = []
        edgeNZ = edgeArr.nonzero()
        #flippedNZ = flippedEdge.nonzero()
        spacing = lesionMask.GetSpacing()

        EPEmax = 0
        EPEscore = 0
        altEPEmax = -1
        origOutsideAreaMax = 0
        altOutsideAreaMax = 0
        TotalAreaMax = 0
        origOutsideAreaMax = 0
        altOutsideAreaMax = 0
        maxDist = 0
        lesionDistMax = 0
        lesion_wMaxDist = 0
        lesionAreaMax = 0
        lesion_wMaxArea = 0

        for i in range(num_features):
            val = i + 1
            probLesionArr = np.where(labeled_array == val, 1, 0)
            probLesionname = os.path.join(EPEmaskfolder, patient + '_lesion' + str(val) + '.nii.gz')
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
                outsideFlipped = []
                altEPEscore = -1

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

                    if flippedProst[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 0:
                        outsideFlipped.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])

                if (len(outsideVar) + len(outsideProst)) > len(insideProst):
                #if (len(outsideVar)+len(outsideProst)) > 1.5*len(insideProst):
                    continue

                if len(outsideProst) != 0:
                    print('len outsideProst:', len(outsideProst))
                    # if lesion outside of prostate variance:
                    if len(outsideVar) != 0:
                        distfromCapsule = 0
                        print('len outsideVar:', len(outsideVar))
                        for vox in range(len(outsideVar)):
                            min_dist = 256
                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(edgeNZ[0])):
                                dist = np.sqrt((spacing[2] * (outsideVar[vox][0] - edgeNZ[0][prostVox])) ** 2 +
                                               (spacing[1] * (outsideVar[vox][1] - edgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0] * (outsideVar[vox][2] - edgeNZ[2][prostVox])) ** 2)
                                if dist < min_dist:
                                    min_dist = dist
                            if min_dist > distfromCapsule:
                                distfromCapsule = min_dist
                        #distfromCapsule = 'outside variance' + str(distfromCapsule)
                    # else if lesion inside variance but outside prostate:
                    else:
                        distfromCapsule = 0
                        for vox in range(len(outsideProst)):
                            min_dist = 256
                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(edgeNZ[0])):
                                dist = np.sqrt((spacing[2]*(outsideProst[vox][0] - edgeNZ[0][prostVox]))**2 +
                                               (spacing[1]*(outsideProst[vox][1] - edgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0]*(outsideProst[vox][2] - edgeNZ[2][prostVox])) ** 2)
                                if dist < min_dist:
                                    min_dist = dist
                            if min_dist > distfromCapsule:
                                distfromCapsule = min_dist

                    outsideArea = len(outsideProst) + len(outsideVar)



                    # if distance outside capsule at least 1/4 of variance OR if area > 200, EPEscore = 1
                    # if both conditions are true, EPE score = 2
                    if len(outsideVar) > self.EPEVarThresh:
                        EPEscore = 3
                    elif (len(outsideVar) < self.EPEVarThresh) & (len(outsideVar)!=0):
                        EPEscore = 2
                    elif outsideArea > self.EPE3AreaThresh:
                        EPEscore = 3
                    elif distfromCapsule > self.variance:
                        EPEscore = 3
                    elif distfromCapsule > (self.variance/4):
                        if outsideArea > self.EPE1_2AreaThresh:
                            EPEscore = 2
                        else:
                            EPEscore = 1
                    elif outsideArea > self.EPE1_2AreaThresh:
                        EPEscore = 1
                    else:
                        EPEscore = 0

                # else, check how close prostate-confined lesion is to capsule:
                else:
                    EPEscore = 0
                    outsideArea = 0
                    distfromCapsule = 'organ confined'

                # if the amount of the lesion outside of the flipped prostate has changed by more than 1/2 of the
                # amount outside of the original prostate mask, use alternate EPE score
                origOutsideArea = len(outsideVar)+len(outsideProst)
                altOutsideArea = len(outsideFlipped)
                TotalArea = len(insideProst) + origOutsideArea
                if TotalArea > 0:
                    percAltOut = altOutsideArea/TotalArea
                else:
                    percAltOut = 0
                diff = altOutsideArea-origOutsideArea
                ave = (altOutsideArea+origOutsideArea)/2

                if origOutsideArea < 100 and altOutsideArea < 10:
                    altEPEscore = 0

                elif EPEscore == 0:
                    if ave > 250:
                        if origOutsideArea > 100:
                            altEPEscore = 3
                        else:
                            altEPEscore = 2
                    elif ave > self.EPE1_2AreaThresh:
                        altEPEscore = 1
                    else:
                        altEPEscore = 0

                    ##    if (altOutsideArea > 75 or ave > 75):
                     ##       altEPEscore = 3
                     ##   else:
                     ##       altEPEscore = 0
                    ##elif origOutsideArea > 5:
                    ##    if diff > 500:
                    ##        altEPEscore = 2
                    ##    elif asymmetry > 0.18:
                    ##        altEPEscore = 1
                    ##    else:
                    ##        altEPEscore = 0



                elif EPEscore < 3:

                    if ave > 900 and altOutsideArea > 900:
                        altEPEscore = 3
                    else:
                        altEPEscore = EPEscore

                else:
                    if asymmetry>0.13 and diff < -500:
                        altEPEscore = 1
                    elif (origOutsideArea < 400) & (ave < 400):
                        altEPEscore = 2
                    else:
                        altEPEscore = EPEscore


                if altEPEscore == 3:
                    if ave < 300 & altOutsideArea < origOutsideArea:
                        altEPEscore = EPEscore


                lesionEPEdata.append([patient, 'lesion' + str(i + 1), EPEscore, altEPEscore,
                                       outsideArea, altOutsideArea, distfromCapsule, len(outsideVar), len(outsideProst),
                                      len(insideProst)+origOutsideArea, asymmetry])
                if EPEscore > EPEmax:
                    EPEmax = EPEscore
                    maxDist = distfromCapsule
                if altEPEscore > altEPEmax:
                        altEPEmax = altEPEscore
                        origOutsideAreaMax = origOutsideArea
                        altOutsideAreaMax = altOutsideArea
                        TotalAreaMax = TotalArea
                if TotalArea > lesionAreaMax:
                    lesionAreaMax = TotalArea
                    lesion_wMaxArea = val
                if not distfromCapsule == 'organ confined':
                    if distfromCapsule > lesionDistMax:
                        lesionDistMax = distfromCapsule
                        lesion_wMaxDist = val

        lesionEPEdata.append([patient, 'all lesions', EPEmax, altEPEmax, origOutsideAreaMax, altOutsideAreaMax])
        patientEPEdata = [patient, EPEmax, altEPEmax, origOutsideAreaMax, altOutsideAreaMax, TotalAreaMax, asymmetry,
                          maxDist, lesionAreaMax, lesion_wMaxArea, lesionDistMax, lesion_wMaxDist]

        return [lesionEPEdata, patientEPEdata]



if __name__ == '__main__':
    c = EPEdetector()
    c.determineEPE()
#    c.create_csv_files()
    print('Check successful')
