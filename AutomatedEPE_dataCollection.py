import SimpleITK as sitk
import numpy as np
import csv
import os
from scipy.ndimage import label


class EPEdetector:
    def __init__(self):

        local = 1
        self.threshold1 = 0.35 #threshold for lesion mask is 0.6344772701607316 , previously tried .35
        self.threshold2 = 0.45
        self.threshold3 = 0.55
        self.threshold4 = 0.65
        self.testNum = 'test4'

        if local:
            self.mask_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData.csv")
        else:
            self.mask_folder = 'Mdrive_mount/MIP/Katie_Merriman/Project2Data/DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData_remote.csv")


    def getEPEdata(self):


        file = open(self.fileName1, 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([['patient', 'version', 'threshold_num', 'lesion', 'insideCapsule', 'outsideCapsule',
                                      'outsideVariance', 'prostCoords', 'distfromVar', 'varCoords']])

        file.close()


        for p in range(1, 556):

            # patient name should follow format 'SURG-00X'
            patient = 'SURG-'+str(p+1000)[1:]
            print(patient)

            try:
                prost = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'wp_bt_undilated.nii.gz'))
                prostArr = sitk.GetArrayFromImage(prost)
                prostEdge = self.createEdge(patient, prost, prostArr,'_capsuleProst.nii.gz')
                prostVariance = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'wp_bt_variance.nii'))
                varArr = sitk.GetArrayFromImage(prostVariance)
                varEdge = self.createEdge(patient, prostVariance, varArr, '_capsuleVar.nii.gz')

                flippedProst = self.MaskFlip(patient, prost, prostArr, '_flippedProst.nii.gz')
                flippedProstArr = sitk.GetArrayFromImage(flippedProst)
                flippedProstEdge = self.createEdge(patient, flippedProst, flippedProstArr, '_flippedCapsuleProst.nii.gz')
                flippedVar = self.MaskFlip(patient, prostVariance, varArr, '_flippedVar.nii.gz')
                flippedVarArr = sitk.GetArrayFromImage(flippedVar)
                flippedVarEdge = self.createEdge(patient, flippedVar, flippedVarArr, "_flippedCapsuleVar.nii.gz")

                asymmetry = np.sum(flippedProstArr != prostArr) / len(prostArr.nonzero()[0])
                print("asymmetry:", asymmetry)

                self.createBinaryLesions(patient)
                print("binaries created")

                lesionMask = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_mask.nii'))
                lesionArr = sitk.GetArrayFromImage(lesionMask)

                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, asymmetry]])
                file.close()


                lesiondata1 = []
                flippeddata1 = []
                lesiondata2 = []
                flippeddata2 = []
                lesiondata3 = []
                flippeddata3 = []
                lesiondata4 = []
                flippeddata4 = []

                print("beginning data calculation")
                self.lesionData(patient, prostArr, varArr, prostEdge, varEdge, lesionArr, "1", "orig")
                self.lesionData(patient, flippedProstArr, flippedVarArr,
                                flippedProstEdge, flippedVarEdge, lesionArr, "1", "flipped")
                self.lesionData(patient, prostArr, varArr, prostEdge, varEdge, lesionArr, "2", "orig")
                self.lesionData(patient, flippedProstArr, flippedVarArr,
                                flippedProstEdge, flippedVarEdge, lesionArr, "2", "flipped")
                self.lesionData(patient, prostArr, varArr, prostEdge, varEdge, lesionArr, "3", "orig")
                self.lesionData(patient, flippedProstArr, flippedVarArr,
                                flippedProstEdge, flippedVarEdge, lesionArr, "3", "flipped")
                self.lesionData(patient, prostArr, varArr, prostEdge, varEdge, lesionArr, "4", "orig")
                self.lesionData(patient, flippedProstArr, flippedVarArr,
                                flippedProstEdge, flippedVarEdge, lesionArr, "4", "flipped")





            except RuntimeError:
                print("remote error")
                file = open(self.fileName1, 'a+', newline='')
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
            if (prostNZ[0][prostVoxel]) < (arr_shape[0] - 1):
                if prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
            if (prostNZ[1][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] - 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[1][prostVoxel]) < (arr_shape[1] - 1):
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] + 1, prostNZ[2][prostVoxel]] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                # if voxel to right or left of current voxel is 0, voxel is on the edge
            if (prostNZ[2][prostVoxel] - 1) > -1:
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] - 1] == 0:
                    capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[2][prostVoxel]) < (arr_shape[2] - 1):
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

    def MaskFlip(self, patient, prost, prostArr, suffix):
        arr_shape = prostArr.shape
        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        flippedProst = np.zeros(arr_shape, dtype=int)
        midline = int(round(sum(prostNZ[2]) / len(prostNZ[2])))
        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            flippedProst[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], (2 * midline - prostNZ[2][prostVoxel])] = 1

        FlippedMaskfolder = os.path.join(self.mask_folder, patient, self.testNum)
        newname = os.path.join(FlippedMaskfolder, patient + suffix)
        FlippedMask = sitk.GetImageFromArray(flippedProst)
        FlippedMask.CopyInformation(prost)
        sitk.WriteImage(FlippedMask, newname)

        return FlippedMask

    def createBinaryLesions(self, patient):
        saveFolder = os.path.join(self.mask_folder, patient, self.testNum)

        lesionHeatMap = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_prob.nii'))
        probArr = sitk.GetArrayFromImage(lesionHeatMap)

        # save binary lesion mask for threshold 1
        binaryArr = np.where(probArr > self.threshold1, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions1.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)


        # save binary lesion mask for threshold 2
        binaryArr = np.where(probArr > self.threshold2, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions2.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)


        # save binary lesion mask for threshold 3
        binaryArr = np.where(probArr > self.threshold3, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions3.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)


        # save binary lesion mask for threshold 4
        binaryArr = np.where(probArr > self.threshold4, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions4.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        return


    def lesionData(self, patient, prostArr, varArr, prostEdge, varEdge, lesionArr, num, version):
        saveFolder = os.path.join(self.mask_folder, patient, self.testNum)
        allLesions = sitk.ReadImage(os.path.join(saveFolder, patient + '_allLesions_thresh' +num+ '.nii.gz'))
        binaryArr = sitk.GetArrayFromImage(allLesions)
        # create labeled array separating individual lesions
        labeled_array, num_features = label(binaryArr)
        labeledname = os.path.join(saveFolder, patient + '_lesions_labeled_thresh' +num+ '.nii.gz')
        labeledmask = sitk.GetImageFromArray(labeled_array)
        labeledmask.CopyInformation(allLesions)
        sitk.WriteImage(labeledmask, labeledname)

        spacing = allLesions.GetSpacing()

        for i in range(num_features):
            val = i + 1
            print(patient + " threshold num: " + num + " lesion: " + str(val))

            distfromCapsule = 0
            distfromVar = 0
            outsideCapsule = 0
            outsideVar = 0
            probLesionArr = np.where(labeled_array == val, 1, 0)
            probLesionname = os.path.join(saveFolder, patient + '_lesion' + str(val) + '_thresh' + num+ '.nii.gz')
            probLesionmask = sitk.GetImageFromArray(probLesionArr)
            probLesionmask.CopyInformation(allLesions)
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
                prostCoords = []
                varCoords = []



                outsideVarArr = np.where(varArr == 0, probLesionArr, 0)
                outsideProstArr = np.where(prostArr == 0,probLesionArr,0)
                insideProstArr = np.where(prostArr == 1, probLesionArr, 0)


                for ind in range(len(lesionNZ[0])):
                    # check if lesion is outside of prostate variance
                    if varArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 0:
                        outsideVar.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])
                    # else check if lesion is outside of prostate
                    if prostArr[lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]] == 0:
                        outsideProst.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])
                    # else store lesion locations inside of prost
                    else:
                        insideProst.append([lesionNZ[0][ind], lesionNZ[1][ind], lesionNZ[2][ind]])


                if len(outsideProst) > len(insideProst):
                #if more than half of the lesion is outside of the organ (likely false call):
                    print("more outside than inside")
                    continue

                outsideCapsule = len(outsideProst)
                print("outsideCapsule old: ", outsideCapsule, " outsideCapsule new: ", len(outsideProstArr.nonzero()[0]))
                outsideVariance = len(outsideVar)
                print("outsideVariance old: ", outsideVariance, " outsideVariance new: ", len(outsideVarArr.nonzero()[0]))
                insideCapsule = len(insideProst)
                print("insodeCapsule old: ", insideCapsule, " insideCapsule new: ", len(insideProstArr.nonzero()[0]))


                ## Create portion of capsule that lesion passes through to check distance against:
                if outsideCapsule !=0:
                    checkEdge = np.where(probLesionArr==1,prostEdge,0)
                    checkEdgeImg = sitk.GetImageFromArray(checkEdge)
                    checkEdgeImg.CopyInformation(allLesions)
                    sitk.WriteImage(checkEdgeImg, os.path.join(saveFolder, "CheckEdgeOutsideCapsule"
                                                               + version + num + "_" + str(val) + ".nii.gz"))

                if outsideVariance != 0:
                    checkVarEdge = np.where(probLesionArr == 1, varEdge, 0)
                    checkVarEdgeImg = sitk.GetImageFromArray(checkVarEdge)
                    checkVarEdgeImg.CopyInformation(allLesions)
                    sitk.WriteImage(checkVarEdgeImg, os.path.join(saveFolder, "CheckEdgeOutsideVariance"
                                                               + version + num + "_" + str(val) + ".nii.gz"))

                prostEdgeNZ = prostEdge.nonzero()
                varEdgeNZ = varEdge.nonzero()
                prostCoordsTemp = []
                prostCoords = []
                varCoordsTemp = []
                varCoords = []

                ## Find distance away from capsule
                if outsideCapsule != 0:
                    print('len outsideProst:', outsideCapsule)
                    # if lesion outside of prostate variance:
                    for vox in range(len(outsideProst)):
                        min_dist = 256
                        if vox % 50 == 0:
                            print('vox', vox)
                        for prostVox in range(len(prostEdgeNZ[0])):
                            dist = np.sqrt((spacing[2] * (outsideProst[vox][0] - prostEdgeNZ[0][prostVox])) ** 2 +
                                           (spacing[1] * (outsideProst[vox][1] - prostEdgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0] * (outsideProst[vox][2] - prostEdgeNZ[2][prostVox])) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                prostCoordsTemp = [
                                    'lesion:' + str(outsideProst[vox][0]) + ',' + str(outsideProst[vox][1]) + ',' +
                                    str(outsideProst[vox][2]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                    + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                            prostCoords = prostCoordsTemp

                    if outsideVariance != 0:
                        print('len outsideVariance:', outsideVariance)
                        # if lesion outside of prostate variance:
                        for vox in range(outsideVariance):
                            min_dist = 256
                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(varEdgeNZ[0])):
                                dist = np.sqrt(
                                    (spacing[2] * (outsideVar[vox][0] - varEdgeNZ[0][prostVox])) ** 2 +
                                    (spacing[1] * (outsideVar[vox][1] - varEdgeNZ[1][prostVox])) ** 2 +
                                    (spacing[0] * (outsideVar[vox][2] - varEdgeNZ[2][prostVox])) ** 2)
                                if dist < min_dist:
                                    min_dist = dist
                                    varCoordsTemp = [
                                        'lesion:' + str(outsideVar[vox][0]) + ',' + str(outsideVar[vox][1]) + ',' +
                                        str(outsideVar[vox][2]) + ', prostate:' + str(varEdgeNZ[0][prostVox])
                                        + ',' + str(varEdgeNZ[1][prostVox]) + ',' + str(varEdgeNZ[2][prostVox])]
                            if min_dist > distfromVar:
                                distfromVar = min_dist
                                varCoords = varCoordsTemp[0]



                else:
                    distfromCapsule = -256
                    for vox in range(len(lesionNZ[0])):
                        min_dist = -256
                        if vox % 50 == 0:
                            print('vox', vox)
                        for prostVox in range(len(prostEdgeNZ[0])):
                            # distance from capsule on inside measured in negative values
                            # greater magnitude is still greater distance from capsule
                            dist = -1 * np.sqrt(
                                            (spacing[2]*(lesionNZ[0][vox] - prostEdgeNZ[0][prostVox]))**2 +
                                           (spacing[1]*(lesionNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0]*(lesionNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            if dist > min_dist:
                                min_dist = dist
                                prostCoordsTemp = ['lesion:' + str(lesionNZ[0][vox]) + ',' + str(lesionNZ[0][vox]) + ',' +
                                       str(lesionNZ[0][vox]) + ', prostate:' + str(prostEdgeNZ[0][prostVox]) +
                                       ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                            prostCoords = prostCoordsTemp[0]

                print(patient, version, num, 'lesion'+ str(val), insideCapsule, outsideCapsule, outsideVariance,
                      distfromCapsule, prostCoords, distfromVar, varCoords)


                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, version, num, 'lesion'+ str(val), insideCapsule, outsideCapsule,
                                      outsideVariance, distfromCapsule, prostCoords, distfromVar, varCoords]])
                file.close()




        return



if __name__ == '__main__':
    c = EPEdetector()
    c.getEPEdata()
#    c.create_csv_files()
    print('Check successful')
