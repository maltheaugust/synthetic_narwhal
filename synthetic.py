#Code by Malthe Bresler (s214631) and Carl Nørlund (s214624)

import numpy as np
import tifffile as tif
import cv2
from PIL import Image
import random as r
import SimpleITK as sitk
from generate_mask import seed_positions
from scipy.ndimage import distance_transform_cdt
from tqdm import tqdm


def salt_and_pepper_noise(data, salt_value = 165, pepper_value = 80, N = 10000):
    '''
    Adds salt and pepper noise to the 3d-image layer by layer.
    '''
    D, N, M = data.shape

    s_and_p = np.array([pepper_value, salt_value], dtype=np.uint8)
    for i in tqdm(range(D)):
        positions = seed_positions((N, M), N+r.randint(-int(N/10), int(N/10))) #Used to sample points to add noise

        for position in positions:
            x = position[0]
            y = position[1]

            data[i][-y, x] = np.random.choice(s_and_p)
    
    return data


def speckle_noise(data, percent=6, tiling_chance = 0.8):
    '''
    Adds speckle noise to the 3D image layer by layer.

    For ordinary speckle noise, pass a tiling chance of 0.
    Tiling chance controls the chance of the noise being 2 wide.
    '''

    speckle_max = int(percent/100*255)

    D, N, M = data.shape

    for i in tqdm(range(D)):
        noise = np.random.randint(-speckle_max, speckle_max, (N, M), dtype=np.int16)
        
        if tiling_chance >0: #Enter tiling loop if tiling chance
            for k in range(1, M, 2):
                for j in range(N):
                    if r.random() < tiling_chance: #Noise in the data seems to tile in 2s
                        noise[j, k] = noise[j, k-1]

        current_layer = data[i].astype(np.int16)
        current_layer += noise
        data[i] = current_layer.astype(np.uint8)
    
    return data


def wind_noise(data, strength=2):
    '''
    This function adds a wind-effect where pixels might move to the side.
    Strength is in pixels and determines the maximum offset that might appear.
    Used on ridge data and not on the final image.

    In the context of the data, this is used to replicate the unpredictability
    of whatever scanning equipment was used.
    '''

    D, N, M = data.shape

    strengths = np.arange(0, strength+1)
    p = [1/((2)**(k+3)) for k in range(len(strengths))]
    p[0] += 1-sum(p)

    for i in range(D):
        current_layer = data[i]

        for k in range(N):
            s = np.random.choice(strengths, p = p )

            if s > 0:
                current_layer[k, s:] = current_layer[k, :-s]
                current_layer[k, 0:s] = 0

        data[i] = current_layer


    return data






def color_ridges(ridges_data, inpaint):
    '''
    This function colours in the ridges. Constants are based on observations.
    Use to fill in values of binary 3d ridges.

    NOTE: mult_constants has a first value of zero because it made some loop code cleaner
    '''
    white_val = 189
    values = [37, 39, 41] #Base values
    mult_constants = [0, 4, 3.1, 1.9, 1.2] #Multiplication constants 


    _, N, _ = ridges_data.shape

    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(1)
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetForegroundValue(255)

    sitk_image = sitk.GetImageFromArray(ridges_data)

    sitk_image = dilate_filter.Execute(sitk_image) #Dilate image to get volume for white lines
    white_data = sitk.GetArrayFromImage(sitk_image)


    white_data[white_data>0] = white_val
    



    distance_transform = distance_transform_cdt(ridges_data)

    #Smoothen the edges
    distance_transform[np.logical_and(distance_transform>0, distance_transform<=1)] = values[2]
    distance_transform[np.logical_and(distance_transform>1, distance_transform<=2)] =values[1]
    distance_transform[np.logical_and(distance_transform>2, distance_transform<5)] = values[0]
    
    distance_transform = distance_transform.astype(np.float32)
    ridges_data = ridges_data.astype(np.float32)
    ridges_data[ridges_data>0] = 1
    

    

    for k in range(len(mult_constants)-1): #Makeshift distance transform going in only 1 axis, makes ridges going out of plane appear lighter

        for i in range(1,N-1): 
            current_layer = ridges_data[i]
            prev_layer = ridges_data[i-1]
            next_layer = ridges_data[i+1]
            mask = np.logical_or(prev_layer==mult_constants[k], next_layer==mult_constants[k])
            mask = np.logical_and(mask, current_layer==1)
            mask = np.logical_and(mask, inpaint[i]>105) 
            ridges_data[i][mask] = mult_constants[k+1]

    distance_transform *= ridges_data
    
    return distance_transform.astype(np.uint8), white_data.astype(np.uint8)




def color_and_combine(save_path, ridges_path, inpaint_path, visualize=False, visualize_layer=400):
    '''
    Wrapper function to make loops creating multiples tifs of data easier. 

    Calls all other functions in the correct order with appropriate parameters. 
    For general use, simply call this function.
    '''


    ridges_data = tif.imread(ridges_path)
    ridges_data = np.rollaxis(ridges_data, axis=1)
    inpaint = tif.imread(inpaint_path)

    ridges_data, white_data = color_ridges(ridges_data=ridges_data, inpaint=inpaint)


    white_mask = np.logical_and(ridges_data == 0, white_data != 0)  #Only place white contrast lines where ridge data does not exist
    white_mask = np.logical_and(white_mask, inpaint>105)            #Do not place the white contrast lines ontop of blobs
    ridges_data[white_mask] = white_data[white_mask]

    ridges_data = wind_noise(ridges_data, strength=2)


    inpaint[ridges_data>0] = ridges_data[ridges_data>0]             

    for i in range(inpaint.shape[0]):                               #Blur image to unite ridges and inpainted data
        inpaint[i] = cv2.GaussianBlur(inpaint[i], (3,3), 0)
        

    inpaint = speckle_noise(inpaint, 6)                             #Add noise that was lost in the blur
    
    if visualize:
        image = Image.fromarray(inpaint[visualize_layer])
        image.show()


    tif.imwrite(save_path, inpaint)



if __name__ == "__main__":

    save_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/data_05_v2.tif"
    ridges_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/mask_05_v2.tif"
    inpaint_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/inpaint_05_v2.tif"


    color_and_combine(save_path=save_path, ridges_path=ridges_path, inpaint_path=inpaint_path)




    #NOTE:
    #   - Currently the wind noise is not added to the mask


    #Ideas:
    #   -Dilate more in inpainting?