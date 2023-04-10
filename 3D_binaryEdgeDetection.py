import SimpleITK as sitk
import numpy as np
# import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature

prost = sitk.ReadImage(r'T:\MIP\Katie_Merriman\Project2Data\NVIDIA_output\Anonymized_NIfTIs_WP\Anonymized_NIfTIs_WP\SURG-001_WP.nii')
prostArr = sitk.GetArrayFromImage(prost)
prostNZ = prostArr.nonzero()  # saved as tuple in z,y,x order
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

newname1 = r'T:\MIP\Katie_Merriman\Project2Data\edgeTest\SURG-001.nii.gz'
dilatedMask1 = sitk.GetImageFromArray(capsule)
dilatedMask1.CopyInformation(prost)
sitk.WriteImage(dilatedMask1, newname1)


# If lesion outside prostate:
#       if lesion outside prostate_variance:
#           EPE = 3
#           (if too many false positives, may need to adjust to determine dist or area)
#       else:
#           calculate outside dist with prostate
#              Start with if dist is 3/4 of variance, EPE 3
#                         if dist is between 1/4 to 3/4 of variance, EPE 2
#                         if dist is less than 1/4, EPE 1
#                         will likely want to finesse with area
# else:calculate inside dist. with prostate
#       if within 1/4 of variance to capsule, EPE1
#       else EPE 0
#       May want to finesse by calculating area within 1/4 variance of capsule?
#                   As calculating distance, could add up number of pixels with min distance < 1/4 variance