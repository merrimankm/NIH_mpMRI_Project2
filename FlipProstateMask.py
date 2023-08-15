import SimpleITK as sitk
import numpy as np
import csv
import os


class flipMask():
    def __init__(self):

        self.save_folder = r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data3'
        self.var = 6.154


    def flipAndCreateVariance(self):


        for p in range(1, 556):
            # patient name should follow format 'SURG-00X'
            patient_id = 'SURG-' + str(p + 1000)[1:]
            print(patient_id)

            self.flipMask(patient_id)

            prostPath = os.path.join(self.save_folder, patient_id, "wp_bt_undilated-Flipped.nii.gz")

            prost = sitk.ReadImage(prostPath)
            prostArr = sitk.GetArrayFromImage(prost)
            prostEdgeArr = self.createEdge(prostArr)

            varZoneArr = self.dilateProst(patient_id, prostPath)
            varZoneEdge = self.createEdge(varZoneArr)
            insideEdgeArr = np.where(prostArr == 1, varZoneEdge, 0)
            varianceEdgeArr = np.where(prostArr == 0, varZoneEdge, 0)
            fullVarArr = np.where(varZoneArr+prostArr > 0, 1, 0)

            prostEdge = sitk.GetImageFromArray(prostEdgeArr)
            prostEdge.CopyInformation(prost)
            insideEdge = sitk.GetImageFromArray(insideEdgeArr)
            insideEdge.CopyInformation(prost)
            varianceEdge = sitk.GetImageFromArray(varianceEdgeArr)
            varianceEdge.CopyInformation(prost)
            fullVar = sitk.GetImageFromArray(fullVarArr)
            fullVar.CopyInformation(prost)

            sitk.WriteImage(prostEdge, os.path.join(self.save_folder, patient_id, "wp_prostEdge-Flipped.nii.gz"))
            sitk.WriteImage(insideEdge, os.path.join(self.save_folder, patient_id, "wp_insideVarEdge-Flipped.nii.gz"))
            sitk.WriteImage(varianceEdge, os.path.join(self.save_folder, patient_id, "wp_outsideVarEdge-Flipped.nii.gz"))
            sitk.WriteImage(fullVar, os.path.join(self.save_folder, patient_id, "wp_fullVar-Flipped.nii.gz"))


    def flipMask(self, patient):
        prostImg = sitk.ReadImage(os.path.join(self.save_folder, patient, 'wp_bt_undilated.nii.gz'))
        prostArr = sitk.GetArrayFromImage(prostImg)
        arr_shape = prostArr.shape
        prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
        flippedProst = np.zeros(arr_shape, dtype=int)
        midline = int(round(sum(prostNZ[2]) / len(prostNZ[2])))
        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            # if that voxel contains lesion, voxel is portion of capsule with lesion contact
            flippedProst[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], (2 * midline - prostNZ[2][prostVoxel])] = 1

        newname = os.path.join(self.save_folder, patient, "wp_bt_undilated-Flipped.nii.gz")
        FlippedMask = sitk.GetImageFromArray(flippedProst)
        FlippedMask.CopyInformation(prostImg)
        sitk.WriteImage(FlippedMask, newname)

        return


    def dilateProst(self, patient_id, imgpath):

        prost = sitk.ReadImage(imgpath)
        [x_space, y_space, z_space] = prost.GetSpacing()
        z = int(np.round(self.var/z_space))
        xy = int(np.round(self.var/x_space))

        prostArr = sitk.GetArrayFromImage(prost)
        arr_shape = prostArr.shape
        varZoneArr = np.zeros(arr_shape, dtype=int)
        insideArr = np.zeros(arr_shape, dtype=int)
        prostNZ = prostArr.nonzero() # saved as tuple in z,y,x order

        arr_size = prost.GetSize()
        sizeX = arr_size[0]
        sizeY = arr_size[1]
        sizeZ = arr_size[2]

        for prostVoxel in range(len(prostNZ[0])):
            # if voxel above or below current voxel is 0, voxel is on the edge
            if (prostNZ[0][prostVoxel] - 1) > -1: # if z position greater than 0 (if looking one slice below won't put us out of range)
                if prostArr[prostNZ[0][prostVoxel] - 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0: # if voxel is on edge in z direction
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for zVox in range(1, z+1):
                        if (prostNZ[0][prostVoxel]-zVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel]-zVox, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                        if (prostNZ[0][prostVoxel]+zVox) < (arr_shape[0]):
                            varZoneArr[prostNZ[0][prostVoxel]+zVox, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            if (prostNZ[0][prostVoxel] + 1) < arr_shape[0]: # if z position less than maximum z position of image - 1 (if looking one slice above won't put us out of range)
                if prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for zVox in range(1, z+1):
                        if (prostNZ[0][prostVoxel]-zVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel]-zVox, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                        if (prostNZ[0][prostVoxel]+zVox) < (arr_shape[0]):
                            varZoneArr[prostNZ[0][prostVoxel]+zVox, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
            # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
            if (prostNZ[1][prostVoxel] - 1) > -1: # if looking one voxel anterior won't put us out of range
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]-1, prostNZ[2][prostVoxel]] == 0:  # if voxel is on edge in y direction
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for yVox in range(1, xy+1):
                        if (prostNZ[1][prostVoxel]-yVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]-yVox, prostNZ[2][prostVoxel]] = 1
                        if (prostNZ[1][prostVoxel]+yVox) < (arr_shape[1]):
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]+yVox, prostNZ[2][prostVoxel]] = 1
            if (prostNZ[1][prostVoxel] + 1) < arr_shape[1]: # if looking one voxel posterior above won't put us out of range
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]+1, prostNZ[2][prostVoxel]] == 0: # if voxel is on edge in y direction
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for yVox in range(1, xy+1):
                        if (prostNZ[1][prostVoxel]-yVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]-yVox, prostNZ[2][prostVoxel]] = 1
                        if (prostNZ[1][prostVoxel]+yVox) < (arr_shape[1]):
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel]+yVox, prostNZ[2][prostVoxel]] = 1
            # if voxel to right or left of current voxel is 0, voxel is on the edge
            if (prostNZ[2][prostVoxel] - 1) > -1: # if looking one voxel left won't put us out of range
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]-1] == 0:  # if voxel is on edge in x direction
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for xVox in range(1, xy+1):
                        if (prostNZ[2][prostVoxel]-xVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]-xVox] = 1
                        if (prostNZ[2][prostVoxel]+xVox) < (arr_shape[2]):
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]+xVox] = 1
            if (prostNZ[2][prostVoxel] + 1) < arr_shape[1]: # if looking one voxel right above won't put us out of range
                if prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]+1] == 0: # if voxel is on edge in x direction
                    varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
                    for xVox in range(1, xy+1):
                        if (prostNZ[2][prostVoxel]-xVox) > -1:
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]-xVox] = 1
                        if (prostNZ[2][prostVoxel]+xVox) < (arr_shape[2]):
                            varZoneArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]+xVox] = 1


        newname1 = os.path.join(self.save_folder, patient_id, 'wp_bt_fullVarZone-Flipped.nii.gz')
        dilatedMask1 = sitk.GetImageFromArray(varZoneArr)
        dilatedMask1.CopyInformation(prost)
        sitk.WriteImage(dilatedMask1, newname1)

        insideArr = np.where((prostArr-varZoneArr) > 0, 1, 0)
        newname2 = os.path.join(self.save_folder, patient_id, 'wp_bt_inside-Flipped.nii.gz')
        dilatedMask2 = sitk.GetImageFromArray(insideArr)
        dilatedMask2.CopyInformation(prost)
        sitk.WriteImage(dilatedMask2, newname2)


        return varZoneArr

    def createEdge(self, prostArr):
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

        return capsule

if __name__ == '__main__':
    c = flipMask()
    c.flipAndCreateVariance()
#    c.create_csv_files()
    print('Mask creation successful')