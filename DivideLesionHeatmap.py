import SimpleITK as sitk
import numpy as np
# import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from scipy.ndimage import label, generate_binary_structure

prost = sitk.ReadImage(r'T:\MIP\Katie_Merriman\Project2Data\DilatedProstate_data2\SURG-001\output\lesion_prob.nii')
prostArr = sitk.GetArrayFromImage(prost)
prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
testArr = np.where(prostArr>0.05,1,0)
testArrNZ = testArr.nonzero()
arr_size = prostArr.shape
capsule = np.zeros(arr_size, dtype = int)

# find array of x,y,z tuples corresponding to voxels of prostNZ that are on edge of prostate array
# and also adjacent to lesion voxels outside of prostate
for prostVoxel in range(len(prostNZ[0])):
    # if voxel above or below current voxel is 0, voxel is on the edge
    # if that voxel contains lesion, voxel is portion of capsule with lesion contact
    if prostArr[prostNZ[0][prostVoxel] - 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
    elif prostArr[prostNZ[0][prostVoxel] + 1, prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
    # if voxel anterior or posterior of current voxel is 0, voxel is on the edge
    elif prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] - 1, prostNZ[2][prostVoxel]] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
    elif prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel] + 1, prostNZ[2][prostVoxel]] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
    # if voxel to right or left of current voxel is 0, voxel is on the edge
    elif prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] - 1] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1
    elif prostArr[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel] + 1] == 0:
        capsule[prostNZ[0][prostVoxel], prostNZ[1][prostVoxel], prostNZ[2][prostVoxel]] = 1

newname1 = r'T:\MIP\Katie_Merriman\Project2Data\edgeTest\SURG-001_lesions.nii.gz'
dilatedMask1 = sitk.GetImageFromArray(testArr)
dilatedMask1.CopyInformation(prost)
sitk.WriteImage(dilatedMask1, newname1)

labeled_array, num_features = label(testArr)

newname1 = r'T:\MIP\Katie_Merriman\Project2Data\edgeTest\SURG-001_lesions_labeled.nii.gz'
dilatedMask1 = sitk.GetImageFromArray(labeled_array)
dilatedMask1.CopyInformation(prost)
sitk.WriteImage(dilatedMask1, newname1)

for i in range(num_features):
    val = i+1
    lesionArr = np.where(labeled_array == val, 1, 0)
    newname1 = r'T:\MIP\Katie_Merriman\Project2Data\edgeTest\SURG-001_lesion'+ str(val) + '.nii.gz'
    dilatedMask1 = sitk.GetImageFromArray(lesionArr)
    dilatedMask1.CopyInformation(prost)
    sitk.WriteImage(dilatedMask1, newname1)