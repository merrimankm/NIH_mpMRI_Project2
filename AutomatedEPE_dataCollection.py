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
        self.testNum = 'test5'

        if local:
            self.mask_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData_quickVersion.csv")
        else:
            self.mask_folder = '/data/merrimankm/DilatedProstate_data3'
            self.fileName1 = os.path.join(os.path.dirname(self.mask_folder), "AllLesionData_remote_quickVersion.csv")

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
                prostPath = os.path.join(self.mask_folder, patient, 'wp_bt_undilated.nii.gz')
                prostImg = sitk.ReadImage(prostPath)
                prostArr = sitk.GetArrayFromImage(prostImg)

                flippedProstPath = os.path.join(self.mask_folder, patient, 'wp_bt_undilated-Flipped.nii.gz')
                flippedProstImg = sitk.ReadImage(flippedProstPath)
                flippedProstArr = sitk.GetArrayFromImage(flippedProstImg)

                asymmetry = np.sum(flippedProstArr != prostArr) / len(prostArr.nonzero()[0])
                print("asymmetry:", asymmetry)
                spacing = prostImg.GetSpacing()

                file = open(self.fileName1, 'a+', newline='')
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerows([[patient, asymmetry, spacing]])
                file.close()

                prostEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_prostEdge.nii.gz"))
                prostEdgeArr = sitk.GetArrayFromImage(prostEdgeImg)
                varImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_fullVar.nii.gz"))
                varArr = sitk.GetArrayFromImage(varImg)
                varEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_outsideVarEdge.nii.gz"))
                varEdgeArr = sitk.GetArrayFromImage(varEdgeImg)
                insideImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_bt_inside.nii.gz"))
                insideArr = sitk.GetArrayFromImage(insideImg)
                insideEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_insideVarEdge.nii.gz"))
                insideEdgeArr = sitk.GetArrayFromImage(insideEdgeImg)
                varZoneImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_bt_fullVarZone.nii.gz"))
                varZoneArr = sitk.GetArrayFromImage(varZoneImg)

                flippedProstEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_prostEdge.nii.gz"))
                flippedProstEdgeArr = sitk.GetArrayFromImage(flippedProstEdgeImg)
                flippedVarImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_fullVar.nii.gz"))
                flippedVarArr = sitk.GetArrayFromImage(flippedVarImg)
                flippedVarEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_outsideVarEdge.nii.gz"))
                flippedVarEdgeArr = sitk.GetArrayFromImage(flippedVarEdgeImg)
                flippedInsideImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_bt_inside.nii.gz"))
                flippedInsideArr = sitk.GetArrayFromImage(flippedInsideImg)
                flippedInsideEdgeImg = sitk.ReadImage(os.path.join(self.mask_folder, patient, "wp_insideVarEdge.nii.gz"))
                flippedInsideEdgeArr = sitk.GetArrayFromImage(flippedInsideEdgeImg)
                flippedVarZoneImg = sitk.ReadImage(os.path.join(self.mask_folder, patient,
                                                                "wp_bt_fullVarZone-Flipped.nii.gz"))
                flippedVarZoneArr = sitk.GetArrayFromImage(flippedVarZoneImg)


                self.createBinaryLesions(patient)
                print("binaries created")

                lesionMask = sitk.ReadImage(os.path.join(self.mask_folder, patient, 'output', 'lesion_mask.nii'))
                lesionArr = sitk.GetArrayFromImage(lesionMask)



                print("beginning data calculation")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "1", "orig")
                self.lesionData(patient, flippedProstArr, flippedProstEdgeArr, flippedVarArr, flippedVarEdgeArr,
                                flippedInsideArr, flippedInsideEdgeArr, lesionArr, flippedVarZoneArr, "1", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "2", "orig")
                self.lesionData(patient, flippedProstArr, flippedProstEdgeArr, flippedVarArr, flippedVarEdgeArr,
                                flippedInsideArr, flippedInsideEdgeArr, lesionArr, flippedVarZoneArr, "2", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "3", "orig")
                self.lesionData(patient, flippedProstArr, flippedProstEdgeArr, flippedVarArr, flippedVarEdgeArr,
                                flippedInsideArr, flippedInsideEdgeArr, lesionArr, flippedVarZoneArr, "3", "orig")
                self.lesionData(patient, prostArr, prostEdgeArr, varArr, varEdgeArr,
                                insideArr, insideEdgeArr, lesionArr, varZoneArr, "4", "orig")
                self.lesionData(patient, flippedProstArr, flippedProstEdgeArr, flippedVarArr, flippedVarEdgeArr,
                                flippedInsideArr, flippedInsideEdgeArr, lesionArr, flippedVarZoneArr, "4", "orig")

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
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)

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

    def lesionData(self, patient, prostArr, prostEdge, varArr, varEdge, insideArr, insideEdge, lesionArr,
                   varianceZoneArr, num, version):
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
            distInsideCapsule = -1
            distInsideCapsule_xy = -1
            distInsideCapsule_z = -1
            xy_distInsideCapsule_3D = -1
            xy_distInsideCapsule_xy = -1
            xy_distInsideCapsule_z = -1
            z_distInsideCapsule_3D = -1
            z_distInsideCapsule_xy = -1
            z_distInsideCapsule_z = -1
            outsideCapsule = 0
            outsideVarArea = 0
            insideCapsuleArea = 0
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
                outsideVarArr = []
                outsideProstArr = []
                insideProstArr = []
                prostCoordsTemp = []
                prostCoordsTemp_xy = []
                prostCoordsTemp_z = []
                prostCoords = []
                prostCoords_xy = []
                prostCoords_z = []
                varCoordsTemp = []
                varCoordsTemp_xy = []
                varCoordsTemp_z = []
                varCoords = []
                varCoords_xy = []
                varCoords_z = []
                insideCoordsTemp = []
                insideCoordsTemp_xy = []
                insideCoordsTemp_z = []
                insideCoords = []
                insideCoords_xy = []
                insideCoords_z = []


                # create array of lesion outside of outerVariance zone, prostate, and innerVariance zone respectively,
                # then create array of edges for those arrays
                outsideVarArr = np.where(varArr == 0, probLesionArr, 0)
                outsideVarEdge = self.createEdge(patient, allLesions, outsideVarArr,
                                                 "_outsideVarEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                     val) + ".nii.gz")
                outsideVarNZ = outsideVarEdge.nonzero()

                outsideProstArr = np.where(prostArr == 0, probLesionArr, 0)
                outsideProstEdge = self.createEdge(patient, allLesions, outsideProstArr,
                                                   "_outsideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                       val) + ".nii.gz")
                outsideProstNZ = outsideProstEdge.nonzero()

                # this array is full portion of lesion that's inside prostate
                insideProstArr = np.where(prostArr == 1, probLesionArr, 0)
                #this array is portion of lesion that's inside prostate but within variance zone
                innerVarArr = np.where(insideArr == 0, probLesionArr, 0)
                insideProstEdge = self.createEdge(patient, allLesions, innerVarArr,
                                                  "_insideProstEdge_" + version + '_thresh' + num + "_lesion" + str(
                                                      val) + ".nii.gz")
                innerVarNZ = insideProstEdge.nonzero()


                outsideCapsuleArea = len(outsideProstArr.nonzero()[0])
                print("outsideCapsule new: ", outsideCapsuleArea)
                outsideVarianceArea = len(outsideVarArr.nonzero()[0])
                print("outsideVariance new: ", outsideVarianceArea)
                insideCapsuleArea = len(insideProstArr.nonzero()[0])
                print("insideCapsule new: ", insideCapsuleArea)

                if outsideCapsuleArea > 2*insideCapsuleArea:
                    # if more than 2/3 of the lesion is outside of the organ (likely false call):
                    print("more outside than inside")
                    continue

                ## Create portion of capsule/outerVariance/innerVariance
                # that lesion passes through to check distance against:
                if outsideCapsuleArea != 0:
                    prostIntersectArr = np.where(probLesionArr == 1, prostEdge, 0)
                    prostIntersectImg = sitk.GetImageFromArray(prostIntersectArr)
                    prostIntersectImg.CopyInformation(allLesions)
                    sitk.WriteImage(prostIntersectImg, os.path.join(saveFolder, "capsuleIntersection_thresh" + version
                                                                + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                    prostEdgeNZ = prostIntersectArr.nonzero()
                    if outsideVarianceArea != 0:
                        varIntersectArr = np.where(probLesionArr == 1, prostEdge, 0)
                        varIntersectImg = sitk.GetImageFromArray(varIntersectArr)
                        varIntersectImg.CopyInformation(allLesions)
                        sitk.WriteImage(varIntersectImg, os.path.join(saveFolder,
                                                     "varianceIntersection_thresh" + version
                                                     + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                        varEdgeNZ = varIntersectArr.nonzero()
                else:
                    insideIntersectArr = np.where(probLesionArr == 1, insideEdge, 0)
                    insideIntersectImg = sitk.GetImageFromArray(insideIntersectArr)
                    insideIntersectImg.CopyInformation(allLesions)
                    sitk.WriteImage(insideIntersectImg, os.path.join(saveFolder, "insideIntersection_thresh" + version
                                                                + "_lesion" + num + "_" + str(val) + ".nii.gz"))
                    insideEdgeNZ = insideIntersectArr.nonzero()








                ## Find distance away from capsule
                if outsideCapsuleArea != 0:
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

                    if outsideVarianceArea != 0:
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
                            for prostVox in range(len(varEdgeNZ[0])):
                                dist = np.sqrt((spacing[2] * (outsideVarNZ[0][vox] - varEdgeNZ[0][prostVox])) ** 2 +
                                               (spacing[1] * (outsideVarNZ[1][vox] - varEdgeNZ[1][prostVox])) ** 2 +
                                               (spacing[0] * (outsideVarNZ[2][vox] - varEdgeNZ[2][prostVox])) ** 2)
                                dist_xy = np.sqrt(
                                    (spacing[1] * (outsideVarNZ[1][vox] - varEdgeNZ[1][prostVox])) ** 2 +
                                    (spacing[0] * (outsideVarNZ[2][vox] - varEdgeNZ[2][prostVox])) ** 2)
                                dist_z = spacing[2] * abs(outsideVarNZ[0][vox] - varEdgeNZ[0][prostVox])
                                # save full coordinate info for point with minimum 3D distance
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_xy = dist_xy
                                    min_dist_z = dist_z
                                    varCoordsTemp = [
                                        'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                        str(outsideVarNZ[2][vox]) + ', prostate: ' + str(varEdgeNZ[0][prostVox])
                                        + ',' + str(varEdgeNZ[1][prostVox]) + ',' + str(varEdgeNZ[2][prostVox])]
                                # save full coordinate info for point with minimum xy distance
                                if outsideVarNZ[0][vox] == varEdgeNZ[0][prostVox]:
                                    if dist_xy < min_xy_dist_xy:
                                        min_xy_dist_3D = dist
                                        min_xy_dist_xy = dist_xy
                                        min_xy_dist_z = dist_z
                                        varCoordsTemp_xy = [
                                            'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                            str(outsideVarNZ[2][vox]) + ', prostate: ' + str(varEdgeNZ[0][prostVox])
                                            + ',' + str(varEdgeNZ[1][prostVox]) + ',' + str(varEdgeNZ[2][prostVox])]
                                # save full coordinate info for point with minimum xy distance
                                if (outsideVarNZ[2][vox] == varEdgeNZ[2][prostVox]) & \
                                        (outsideVarNZ[1][vox] == varEdgeNZ[1][prostVox]):
                                    if dist_z < min_z_dist_z:
                                        min_z_dist_3D = dist
                                        min_z_dist_xy = dist_xy
                                        min_z_dist_z = dist_z
                                        varCoordsTemp_z = [
                                            'lesion:' + str(outsideVarNZ[0][vox]) + ',' + str(outsideVarNZ[1][vox]) + ',' +
                                            str(outsideVarNZ[2][vox]) + ', prostate: ' + str(varEdgeNZ[0][prostVox])
                                            + ',' + str(varEdgeNZ[1][prostVox]) + ',' + str(varEdgeNZ[2][prostVox])]
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
                    print('len insideProstEdge:', len(innerVarNZ[0]))
                    # if lesion outside of prostate variance:
                    for vox in range(len(innerVarNZ[0])):
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
                        for prostVox in range(len(insideEdgeNZ[0])):
                            dist = np.sqrt((spacing[2] * (innerVarNZ[0][vox] - insideEdgeNZ[0][prostVox])) ** 2 +
                                           (spacing[1] * (innerVarNZ[1][vox] - insideEdgeNZ[1][prostVox])) ** 2 +
                                           (spacing[0] * (innerVarNZ[2][vox] - insideEdgeNZ[2][prostVox])) ** 2)
                            dist_xy = np.sqrt((spacing[1] * (innerVarNZ[1][vox] - insideEdgeNZ[1][prostVox])) ** 2 +
                                              (spacing[0] * (innerVarNZ[2][vox] - insideEdgeNZ[2][prostVox])) ** 2)
                            dist_z = spacing[2] * abs(innerVarNZ[0][vox] - insideEdgeNZ[0][prostVox])
                            # save full coordinate info for point with minimum 3D distance
                            if dist < min_dist:
                                min_dist = dist
                                min_dist_xy = dist_xy
                                min_dist_z = dist_z
                                insideCoordsTemp = [
                                    'lesion:' + str(innerVarNZ[0][vox]) + ',' + str(innerVarNZ[1][vox]) + ',' +
                                    str(innerVarNZ[2][vox]) + ', prostate: ' + str(insideEdgeNZ[0][prostVox])
                                    + ',' + str(insideEdgeNZ[1][prostVox]) + ',' + str(insideEdgeNZ[2][prostVox])]
                            # if on same slice, save full coordinate info for point with minimum xy distance
                            if innerVarNZ[0][vox] == insideEdgeNZ[0][prostVox]:
                                if dist_xy < min_xy_dist_xy:
                                    min_xy_dist_3D = dist
                                    min_xy_dist_xy = dist_xy
                                    min_xy_dist_z = dist_z
                                    insideCoordsTemp_xy = [
                                        'lesion:' + str(innerVarNZ[0][vox]) + ',' + str(innerVarNZ[1][vox]) + ',' +
                                        str(innerVarNZ[2][vox]) + ', prostate: ' + str(insideEdgeNZ[0][prostVox])
                                        + ',' + str(insideEdgeNZ[1][prostVox]) + ',' + str(insideEdgeNZ[2][prostVox])]
                            # save full coordinate info for point with minimum xy distance
                            if (innerVarNZ[2][vox] == insideEdgeNZ[2][prostVox]) & \
                                    (innerVarNZ[1][vox] == insideEdgeNZ[1][prostVox]):
                                if dist_z < min_z_dist_z:
                                    min_z_dist_3D = dist
                                    min_z_dist_xy = dist_xy
                                    min_z_dist_z = dist_z
                                    insideCoordsTemp_z = [
                                        'lesion:' + str(innerVarNZ[0][vox]) + ',' + str(innerVarNZ[1][vox]) + ',' +
                                        str(innerVarNZ[2][vox]) + ', prostate: ' + str(insideEdgeNZ[0][prostVox])
                                        + ',' + str(insideEdgeNZ[1][prostVox]) + ',' + str(insideEdgeNZ[2][prostVox])]
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
                            insideCoords_z = insideCoordsTemp_z

                print(patient, version, num, 'lesion' + str(val), insideCapsuleArea, outsideCapsuleArea, outsideVarianceArea,
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
                                      insideCapsuleArea, outsideCapsuleArea, outsideVarianceArea,
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
