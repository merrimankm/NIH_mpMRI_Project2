import SimpleITK as sitk
import numpy as np
import csv
import os
from scipy.ndimage import label


class EPEdetector:
    def __init__(self):

        local = 1
        self.threshold1 = 0.35  # threshold for lesion mask is 0.6344772701607316 , previously tried .35
        self.threshold2 = 0.45
        self.threshold3 = 0.55
        self.threshold4 = 0.65
        self.testNum = 'test4'

        if local:
            self.mask_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData_updatedDist.csv")
        else:
            self.mask_folder = '/data/merrimankm/DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData_remote.csv")

    def getEPEdata(self):

        file = open(self.fileName1, 'a+', newline='')
        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows([["patient", "version", "thresholdNum", "lesion",
                              "insideCapsule", "outsideCapsule", "outsideVariance",
                              "distfromCapsule", "distfromCapsule_xy", "distfromCapsule_z", "prostCoords",
                              "xy_distfromCapsule_3D", "xy_distfromCapsule_xy", "xy_distfromCapsule_z", "prostCoords_xy",
                              "z_distfromCapsule_3D", "z_distfromCapsule_xy", "z_distfromCapsule_z", "prostCoords_z",
                              "distfromVar", "distfromVar_xy", "distfromVar_z", "varCoords",
                              "xy_distfromVar_3D", "xy_distfromVar_xy", "xy_distfromVar_z", "varCoords_xy",
                              "z_distfromVar_3D", "z_distfromVar_xy", "z_distfromVar_z", "varCoords_z",
                              "distInsideCapsule", "distInsideCapsule_xy", "distInsideCapsule_z", "insideCoords",
                              "xy_distInsideCapsule_3D", "xy_distInsideCapsule_xy", "xy_distInsideCapsule_z", "insideCoords_xy",
                              "z_distInsideCapsule_3D", "z_distInsideCapsule_xy", "z_distInsideCapsule_z", "insideCoords_z"]])

        file.close()

        for p in range(1, 556):

            # patient name should follow format 'SURG-00X'
            patient = 'SURG-' + str(p + 1000)[1:]
            print(patient)

            try:
                prost = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'wp_bt_undilated.nii.gz'))
                prostArr = sitk.GetArrayFromImage(prost)
                prostEdge = self.createEdge(patient, prost, prostArr, '_capsuleProst.nii.gz')
                prostVariance = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'wp_bt_variance.nii'))
                varArr = sitk.GetArrayFromImage(prostVariance)
                varEdge = self.createEdge(patient, prostVariance, varArr, '_capsuleVar.nii.gz')

                flippedProst = self.MaskFlip(patient, prost, prostArr, '_flippedProst.nii.gz')
                flippedProstArr = sitk.GetArrayFromImage(flippedProst)
                flippedProstEdge = self.createEdge(patient, flippedProst, flippedProstArr,
                                                   '_flippedCapsuleProst.nii.gz')
                flippedVar = self.MaskFlip(patient, prostVariance, varArr, '_flippedVar.nii.gz')
                flippedVarArr = sitk.GetArrayFromImage(flippedVar)
                flippedVarEdge = self.createEdge(patient, flippedVar, flippedVarArr, "_flippedCapsuleVar.nii.gz")

                asymmetry = np.sum(flippedProstArr != prostArr) / len(prostArr.nonzero()[0])
                print("asymmetry:", asymmetry)

                self.createBinaryLesions(patient)
                print("binaries created")

                lesionMask = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_mask.nii'))
                lesionArr = sitk.GetArrayFromImage(lesionMask)

                spacing = prost.GetSpacing()

                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, asymmetry, spacing]])
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

        # save edge mask to folder:

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
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh1.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 2
        binaryArr = np.where(probArr > self.threshold2, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh2.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 3
        binaryArr = np.where(probArr > self.threshold3, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh3.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        # save binary lesion mask for threshold 4
        binaryArr = np.where(probArr > self.threshold4, 1, 0)
        binaryname = os.path.join(saveFolder, patient + '_allLesions_thresh4.nii.gz')
        binarymask = sitk.GetImageFromArray(binaryArr)
        binarymask.CopyInformation(lesionHeatMap)
        sitk.WriteImage(binarymask, binaryname)

        return

    def lesionData(self, patient, prostArr, varArr, prostEdge, varEdge, lesionArr, num, version):
        saveFolder = os.path.join(self.mask_folder, patient, self.testNum)
        allLesions = sitk.ReadImage(os.path.join(saveFolder, patient + '_allLesions_thresh' + num + '.nii.gz'))
        binaryArr = sitk.GetArrayFromImage(allLesions)
        # create labeled array separating individual lesions
        labeled_array, num_features = label(binaryArr)
        labeledname = os.path.join(saveFolder, patient + '_lesions_labeled_thresh' + num + '.nii.gz')
        labeledmask = sitk.GetImageFromArray(labeled_array)
        labeledmask.CopyInformation(allLesions)
        sitk.WriteImage(labeledmask, labeledname)

        spacing = allLesions.GetSpacing()

        for i in range(num_features):
            val = i + 1
            print(patient + " threshold num: " + num + " lesion: " + str(val))

            distfromCapsule = -1
            distfromCapsule_xy = -1
            distfromCapsule_z = -1
            xy_distfromCapsule_3D = -1
            xy_distfromCapsule_xy = -1
            xy_distfromCapsule_z = -1
            z_distfromCapsule_3D = -1
            z_distfromCapsule_xy = -1
            z_distfromCapsule_z = -1
            distfromVar = -1
            distfromVar_xy = -1
            distfromVar_z = -1
            xy_distfromVar_3D = -1
            xy_distfromVar_xy = -1
            xy_distfromVar_z = -1
            z_distfromVar_3D = -1
            z_distfromVar_xy = -1
            z_distfromVar_z = -1
            distInsideCapsule = -256
            distInsideCapsule_xy = -256
            distInsideCapsule_z = -256
            xy_distInsideCapsule_3D = -256
            xy_distInsideCapsule_xy = -256
            xy_distInsideCapsule_z = -256
            z_distInsideCapsule_3D = -256
            z_distInsideCapsule_xy = -256
            z_distInsideCapsule_z = -256
            outsideCapsule = 0
            outsideVar = 0
            probLesionArr = np.where(labeled_array == val, 1, 0)
            probLesionname = os.path.join(saveFolder, patient + '_lesion' + str(val) + '_thresh' + num + '.nii.gz')
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
                prostCoords_xy = []
                prostCoords_z = []
                varCoords = []
                varCoords_xy = []
                varCoords_z = []
                insideCoords = []
                insideCoords_xy = []
                insideCoords_z = []

                outsideVarArr = np.where(varArr == 0, probLesionArr, 0)
                outsideVarEdge = self.createEdge(patient, allLesions, outsideVarArr,
                                                 "_outsideVarEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                     val) + ".nii.gz")
                outsideProstArr = np.where(prostArr == 0, probLesionArr, 0)
                outsideProstEdge = self.createEdge(patient, allLesions, outsideProstArr,
                                                   "_outsideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                       val) + ".nii.gz")
                insideProstArr = np.where(prostArr == 1, probLesionArr, 0)
                insideProstEdge = self.createEdge(patient, allLesions, insideProstArr,
                                                  "_insideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                      val) + ".nii.gz")

                outsideCapsule = len(outsideProstArr.nonzero()[0])
                print("outsideCapsule new: ", outsideCapsule)
                outsideVariance = len(outsideVarArr.nonzero()[0])
                print("outsideVariance new: ", outsideVariance)
                insideCapsule = len(insideProstArr.nonzero()[0])
                print("insideCapsule new: ", insideCapsule)

                if outsideCapsule > 2*insideCapsule:
                    # if more than 2/3 of the lesion is outside of the organ (likely false call):
                    print("more outside than inside")
                    continue

                ## Create portion of capsule that lesion passes through to check distance against:
                if outsideCapsule != 0:
                    checkEdge = np.where(probLesionArr == 1, prostEdge, 0)
                    checkEdgeImg = sitk.GetImageFromArray(checkEdge)
                    checkEdgeImg.CopyInformation(allLesions)
                    sitk.WriteImage(checkEdgeImg, os.path.join(saveFolder, "CheckEdgeOutsideCapsule_"
                                                               + version + num + "_" + str(val) + ".nii.gz"))

                if outsideVariance != 0:
                    checkVarEdge = np.where(probLesionArr == 1, varEdge, 0)
                    checkVarEdgeImg = sitk.GetImageFromArray(checkVarEdge)
                    checkVarEdgeImg.CopyInformation(allLesions)
                    sitk.WriteImage(checkVarEdgeImg, os.path.join(saveFolder, "CheckEdgeOutsideVariance_"
                                                                  + version + num + "_" + str(val) + ".nii.gz"))

                prostEdgeNZ = prostEdge.nonzero()
                varEdgeNZ = varEdge.nonzero()
                outsideProstNZ = outsideProstEdge.nonzero()
                outsideVarNZ = outsideVarEdge.nonzero()
                insideNZ = insideProstEdge.nonzero()



                ## Find distance away from capsule
                if outsideCapsule != 0:
                    print('len outsideProstEdge:', len(outsideProstNZ[0]))
                    # if lesion outside of prostate variance:
                    for vox in range(len(outsideProstNZ[0])):
                        min_dist = 256
                        min_dist_xy = 256
                        min_dist_z = 256
                        min_xy_dist_3D = 256
                        min_xy_dist_xy = 256
                        min_xy_dist_z = 256
                        min_z_dist_3D = 256
                        min_z_dist_xy = 256
                        min_z_dist_z = 256
                        if vox % 50 == 0:
                            print('vox', vox)
                        for prostVox in range(len(prostEdgeNZ[0])):
                            dist = np.sqrt((spacing[2] * (outsideProstNZ[0][vox] - prostEdgeNZ[0][prostVox])) ** 2 +
                                           (spacing[1] * (outsideProstNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0] * (outsideProstNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_xy = np.sqrt((spacing[1] * (outsideProstNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                              (spacing[0] * (outsideProstNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_z = spacing[2] * abs(outsideProstNZ[0][vox] - prostEdgeNZ[0][prostVox])
                            # save full coordinate info for point with minimum 3D distance
                            if dist < min_dist:
                                min_dist = dist
                                min_dist_xy = dist_xy
                                min_dist_z = dist_z
                                prostCoordsTemp = [
                                    'lesion:' + str(outsideProstNZ[0][vox]) + ',' + str(outsideProstNZ[1][vox]) + ',' +
                                    str(outsideProstNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                    + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                            # if on same slice, save full coordinate info for point with minimum xy distance
                            if outsideProstNZ[0][vox] == prostEdgeNZ[0][prostVox]:
                                if dist_xy < min_xy_dist_xy:
                                    min_xy_dist_3D = dist
                                    min_xy_dist_xy = dist_xy
                                    min_xy_dist_z = dist_z
                                    prostCoordsTemp_xy = [
                                        'lesion:' + str(outsideProstNZ[0][vox]) + ',' + str(outsideProstNZ[1][vox]) + ',' +
                                        str(outsideProstNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                        + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                            # save full coordinate info for point with minimum xy distance
                            if (outsideProstNZ[2][vox] == prostEdgeNZ[2][prostVox]) & \
                                    (outsideProstNZ[1][vox] == prostEdgeNZ[1][prostVox]):
                                if dist_z < min_z_dist_z:
                                    min_z_dist_3D = dist
                                    min_z_dist_xy = dist_xy
                                    min_z_dist_z = dist_z
                                    prostCoordsTemp_z = [
                                        'lesion:' + str(outsideProstNZ[0][vox]) + ',' + str(outsideProstNZ[1][vox]) + ',' +
                                        str(outsideProstNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                        + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                        if min_dist > distfromCapsule:
                            distfromCapsule = min_dist
                            distfromCapsule_xy = min_dist_xy
                            distfromCapsule_z = min_dist_z
                            prostCoords = prostCoordsTemp
                        if min_xy_dist_xy > xy_distfromCapsule_xy:
                            xy_distfromCapsule_3D = min_xy_dist_3D
                            xy_distfromCapsule_xy = min_xy_dist_xy
                            xy_distfromCapsule_z = min_xy_dist_z
                            prostCoords_xy = prostCoordsTemp_xy
                        if min_z_dist_z > z_distfromCapsule_z:
                            z_distfromCapsule_3D = min_z_dist_3D
                            z_distfromCapsule_xy = min_z_dist_xy
                            z_distfromCapsule_z = min_z_dist_z
                            prostCoords_z = prostCoordsTemp_z

                    if outsideVariance != 0:
                        print('len outsideVarEdge:', len(outsideVarNZ[0]))
                        # if lesion outside of prostate variance:
                        for vox in range(len(outsideVarNZ[0])):
                            min_dist = 256
                            min_dist_xy = 256
                            min_dist_z = 256
                            min_xy_dist_3D = 256
                            min_xy_dist_xy = 256
                            min_xy_dist_z = 256
                            min_z_dist_3D = 256
                            min_z_dist_xy = 256
                            min_z_dist_z = 256
                            if vox % 50 == 0:
                                print('vox', vox)
                            for prostVox in range(len(prostEdgeNZ[0])):
                                dist = np.sqrt((spacing[2] * (outsideVarNZ[0][vox] - prostEdgeNZ[0][prostVox])) ** 2 +
                                               (spacing[1] * (outsideVarNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0] * (outsideVarNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                                dist_xy = np.sqrt(
                                    (spacing[1] * (outsideVarNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                    (spacing[0] * (outsideVarNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                                dist_z = spacing[2] * abs(outsideVarNZ[0][vox] - prostEdgeNZ[0][prostVox])
                                # save full coordinate info for point with minimum 3D distance
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_xy = dist_xy
                                    min_dist_z = dist_z
                                    varCoordsTemp = [
                                        'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                        str(outsideVarNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                        + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                                # save full coordinate info for point with minimum xy distance
                                if outsideVarNZ[0][vox] == prostEdgeNZ[0][prostVox]:
                                    if dist_xy < min_xy_dist_xy:
                                        min_xy_dist_3D = dist
                                        min_xy_dist_xy = dist_xy
                                        min_xy_dist_z = dist_z
                                        varCoordsTemp_xy = [
                                            'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                            str(outsideVarNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                            + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                                # save full coordinate info for point with minimum xy distance
                                if (outsideVarNZ[2][vox] == prostEdgeNZ[2][prostVox]) & \
                                        (outsideVarNZ[1][vox] == prostEdgeNZ[1][prostVox]):
                                    if dist_z < min_z_dist_z:
                                        min_z_dist_3D = dist
                                        min_z_dist_xy = dist_xy
                                        min_z_dist_z = dist_z
                                        varCoordsTemp_z = [
                                            'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                            str(outsideVarNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                            + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                            if min_dist > distfromVar:
                                distfromVar = min_dist
                                distfromVar_xy = min_dist_xy
                                distfromVar_z = min_dist_z
                                varCoords = varCoordsTemp
                            if min_xy_dist_xy > xy_distfromVar_xy:
                                xy_distfromVar_3D = min_xy_dist_3D
                                xy_distfromVar_xy = min_xy_dist_xy
                                xy_distfromVar_z = min_xy_dist_z
                                varCoords_xy = varCoordsTemp_xy
                            if min_z_dist_z > z_distfromVar_z:
                                z_distfromVar_3D = min_z_dist_3D
                                z_distfromVar_xy = min_z_dist_xy
                                z_distfromVar_z = min_z_dist_z
                                varCoords_z = varCoordsTemp_z

                else:
                    print('len insideProstEdge:', len(insideNZ[0]))
                    # if lesion inside prostate:
                    for vox in range(len(insideNZ[0])):
                        min_dist = -256
                        min_dist_xy = -256
                        min_dist_z = -256
                        min_xy_dist_3D = -256
                        min_xy_dist_xy = -256
                        min_xy_dist_z = -256
                        min_z_dist_3D = -256
                        min_z_dist_xy = -256
                        min_z_dist_z = -256
                        if vox % 50 == 0:
                            print('vox', vox)
                        for prostVox in range(len(prostEdgeNZ[0])):
                            dist = -1.0 * np.sqrt((spacing[2] * (insideNZ[0][vox] - prostEdgeNZ[0][prostVox])) ** 2 +
                                                  (spacing[1] * (insideNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                                  (spacing[0] * (insideNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_xy = -1.0 * np.sqrt(
                                (spacing[1] * (insideNZ[1][vox] - prostEdgeNZ[1][prostVox])) ** 2 +
                                (spacing[0] * (insideNZ[2][vox] - prostEdgeNZ[2][prostVox])) ** 2)
                            dist_z = -1.0 * spacing[2] * abs(insideNZ[0][vox] - prostEdgeNZ[0][prostVox])
                            # save full coordinate info for point with minimum 3D distance
                            if dist > min_dist:
                                min_dist = dist
                                min_dist_xy = dist_xy
                                min_dist_z = dist_z
                                insideCoordsTemp = [
                                    'lesion:' + str(insideNZ[0][vox]) + ',' + str(insideNZ[1][vox]) + ',' +
                                    str(insideNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                    + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                            # save full coordinate info for point with minimum xy distance
                            if (insideNZ[0][vox] == prostEdgeNZ[0][prostVox]):
                                if dist_xy > min_xy_dist_xy:
                                    min_xy_dist_3D = dist
                                    min_xy_dist_xy = dist_xy
                                    min_xy_dist_z = dist_z
                                    insideCoordsTemp_xy = [
                                        'lesion:' + str(insideNZ[0][vox]) + ',' + str(insideNZ[1][vox]) + ',' +
                                        str(insideNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                        + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                            # save full coordinate info for point with minimum xy distance
                            if (insideNZ[1][vox] == prostEdgeNZ[1][prostVox]) & \
                                    (insideNZ[2][vox] == prostEdgeNZ[2][prostVox]):
                                if dist_z > min_z_dist_z:
                                    min_z_dist_3D = dist
                                    min_z_dist_xy = dist_xy
                                    min_z_dist_z = dist_z
                                    insideCoordsTemp_z = [
                                        'lesion:' + str(insideNZ[0][vox]) + ',' + str(insideNZ[1][vox]) + ',' +
                                        str(insideNZ[2][vox]) + ', prostate: ' + str(prostEdgeNZ[0][prostVox])
                                        + ',' + str(prostEdgeNZ[1][prostVox]) + ',' + str(prostEdgeNZ[2][prostVox])]
                        if min_dist > distInsideCapsule:
                            distInsideCapsule = min_dist
                            distInsideCapsule_xy = min_dist_xy
                            distInsideCapsule_z = min_dist_z
                            insideCoords = insideCoordsTemp
                        if min_xy_dist_xy > xy_distInsideCapsule_xy:
                            xy_distInsideCapsule_3D = min_xy_dist_3D
                            xy_distInsideCapsule_xy = min_xy_dist_xy
                            xy_distInsideCapsule_z = min_xy_dist_z
                            insideCoords_xy = insideCoordsTemp_xy
                        if min_z_dist_z > z_distInsideCapsule_z:
                            z_distInsideCapsule_3D = min_z_dist_3D
                            z_distInsideCapsule_xy = min_z_dist_xy
                            z_distInsideCapsule_z = min_z_dist_z
                            insideCoords_z = insideCoordsTemp_xy

                print(patient, version, num, 'lesion' + str(val), insideCapsule, outsideCapsule, outsideVariance,
                      distfromCapsule, distfromCapsule_xy, distfromCapsule_z, prostCoords,
                      xy_distfromCapsule_3D, xy_distfromCapsule_xy, xy_distfromCapsule_z, prostCoords_xy,
                      z_distfromCapsule_3D, z_distfromCapsule_xy, z_distfromCapsule_z, prostCoords_z,
                      distfromVar, distfromVar_xy, distfromVar_z, varCoords,
                      xy_distfromVar_3D, xy_distfromVar_xy, xy_distfromVar_z, varCoords_xy,
                      z_distfromVar_3D, z_distfromVar_xy, z_distfromVar_z, varCoords_z,
                      distInsideCapsule, distInsideCapsule_xy, distInsideCapsule_z, insideCoords,
                      xy_distInsideCapsule_3D, xy_distInsideCapsule_xy, xy_distInsideCapsule_z, insideCoords_xy,
                      z_distInsideCapsule_3D, z_distInsideCapsule_xy, z_distInsideCapsule_z, insideCoords_z)

                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, version, num, 'lesion' + str(val),
                                      insideCapsule, outsideCapsule, outsideVariance,
                                      distfromCapsule, distfromCapsule_xy, distfromCapsule_z, prostCoords,
                                      xy_distfromCapsule_3D, xy_distfromCapsule_xy, xy_distfromCapsule_z,
                                      prostCoords_xy,
                                      z_distfromCapsule_3D, z_distfromCapsule_xy, z_distfromCapsule_z, prostCoords_z,
                                      distfromVar, distfromVar_xy, distfromVar_z, varCoords,
                                      xy_distfromVar_3D, xy_distfromVar_xy, xy_distfromVar_z, varCoords_xy,
                                      z_distfromVar_3D, z_distfromVar_xy, z_distfromVar_z, varCoords_z,
                                      distInsideCapsule, distInsideCapsule_xy, distInsideCapsule_z, insideCoords,
                                      xy_distInsideCapsule_3D, xy_distInsideCapsule_xy, xy_distInsideCapsule_z, insideCoords_xy,
                                      z_distInsideCapsule_3D, z_distInsideCapsule_xy, z_distInsideCapsule_z, insideCoords_z]])
                file.close()

        return


if __name__ == '__main__':
    c = EPEdetector()
    c.getEPEdata()
    #    c.create_csv_files()
    print('Check successful')
