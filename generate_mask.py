#Authors: Malthe Bresler (s214631), Carl Nørlund (s214624)

import numpy as np
from math import cos, sin, pi
import random as r
from tqdm import tqdm
import tifffile as tif
import SimpleITK as sitk
import perlin
from skimage.morphology import skeletonize_3d
from scipy.ndimage import convolve, gaussian_filter




def apply_noise(data):
    '''
    This function convolves the data with a noisy kernel.
    Used to add noise to thickened ridges, called in populate_data()
    '''

    p = perlin.Perlin(r.randint(2,100))
    kernel = np.random.normal(scale=0.1, size=(3,3,3))
    kernel *= 10
    


    noisy_data = gaussian_filter(data, sigma=3, radius=1)
    noisy_data = gaussian_filter(noisy_data, sigma=3, radius=1)
    noisy_data = gaussian_filter(noisy_data, sigma=2, radius=1)
    noisy_data[noisy_data > 0] = 1
    #noisy_data.astype(np.float32)

    #noisy_data = convolve(noisy_data, kernel, mode = "constant", cval=0)


    return noisy_data.astype(np.uint8)


def thicken(array, dilation_steps = 3):
    '''
    This function takes an ndarray containing one ridge,
    dilates it until edges touch and then finds the centerline.
    It then dilates again to thicken the line

    Used to thicken simulated ridges. Called in populate_data()
    '''
    
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(3) 
    dilate_filter.SetKernelType(sitk.sitkBall)
    dilate_filter.SetForegroundValue(1)

    array[array>0]=1


    sitk_image = sitk.GetImageFromArray(array)
    
    for _ in range(dilation_steps):
        sitk_image = dilate_filter.Execute(sitk_image)

    data = sitk.GetArrayFromImage(sitk_image)
    array2 = skeletonize_3d(data).astype(np.uint8) 

    sitk_image = sitk.GetImageFromArray(array2)


    for _ in range(1): 
        sitk_image = dilate_filter.Execute(sitk_image)

    array2 = sitk.GetArrayFromImage(sitk_image)

    array += array2

    sitk_image = sitk.GetImageFromArray(array)

    dilate_filter.SetKernelRadius(2)

    for _ in range(2):  
        sitk_image = dilate_filter.Execute(sitk_image)

    array = sitk.GetArrayFromImage(sitk_image)
    


    return array.astype(np.uint8)


def angled_sine_wave(A, omega, phi, angle, ts):
    '''
    Creates a sine wave with the given parameters and rotates it using
    a rotation matrix

    -ts should be an array of t-values to run the parameterized rotated sine wave over.
    -ts should have a relatively high resolution to avoid noticeable jumps when converting to an image
    '''

    rotation_matrix = np.array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])

    values = np.zeros((2,len(ts)))
    for i,t in enumerate(ts):
        evaluation = rotation_matrix@np.array([[t],[A*sin(omega*t + phi)]])
        values[:,i] = evaluation[:,0]


    return values





def randomized_sum_of_sines(N, A_range, omega_range, phi_range, angle_range, ts=np.arange(0, 1500, 0.05)):
    '''
    Takes ranges of allowed values (given in lists of two elements or tuples) and creates n randomized sines, 
    where each parameter is sampled within the range.

    -ts should be an array of t-values to run the parameterized rotated sine wave over.
    -ts should have a relatively high resolution to avoid noticeable jumps when converting to an image
    '''
    
    sines = np.zeros((N,2,len(ts)))
    As = np.linspace(A_range[0], A_range[1], 50)
    Omegas = np.linspace(omega_range[0], omega_range[1], 50)
    Phis = np.linspace(phi_range[0], phi_range[1], 50)
    Angles = np.linspace(angle_range[0], angle_range[1],10)

    for n in range(N):
        A = r.choice(As)
        omega = r.choice(Omegas)
        phi = r.choice(Phis)
        angle = r.choice(Angles)

        sines[n] = angled_sine_wave(A = A, omega = omega, phi = phi, angle = angle, ts = ts)

    return np.sum(sines, axis=0)


def generate_ridge():
    '''
    Generates a ridge. Parameters are not given as an input as they have been finely tuned
    by trial and error beforehand.
    Generates two lines for x and y (running along z) for varying thickness
    - k controls thickness of the ridges (apart from the sine variation)
    - ts controls length of ridges #IMPORTANT# Change this value here.
    - N controls how many sines are summed. higher N approximates a line
    '''

    k=4 #was 4
    N_it = 8 # was 8
    ts = np.arange(0,120, 0.05)

    sine = randomized_sum_of_sines(N = N_it, A_range = [0.8, 1.2], omega_range = [0.05, 0.4], phi_range=[0, 1], angle_range = [1/6*pi-0.25, 1/6*pi+0.125], ts = ts)
    ysine = randomized_sum_of_sines(N = N_it, A_range = [1.0, 1.2], omega_range = [0.05, 0.4], phi_range=[0, 1], angle_range = [1/60*pi-0.05, 1/60*pi+0.05], ts = ts)

    p = perlin.Perlin(r.randint(2,100))
    perlin_noise = np.array([abs(p.one(i)) for i in range(2*len(ts))])
    perlin_noise = perlin_noise/np.max(perlin_noise)*3
    perlin_noise_1 = perlin_noise[:len(ts)]
    perlin_noise_2 = perlin_noise[len(ts):]


    #Is this perlin noise wack????? yes but it works
    #NOTE: omega range was [0.05, 0.4]

    z = np.round(sine[0], decimals=5).astype(np.int32)

    x = -np.round(sine[1]+perlin_noise_2-randomized_sum_of_sines(N = N_it+r.randint(1,4), A_range = [0.3, 0.8], omega_range = [0.05, 1], phi_range=[pi, 2*pi], angle_range = [0,0], ts = ts)[1], decimals=5).astype(np.int32)#, decimals=5).astype(np.int32) #Flip sign to match data
    x2 = np.round(x.copy()-perlin_noise_1-k-randomized_sum_of_sines(N = N_it+r.randint(1,4), A_range = [0.3, 0.8], omega_range = [0.05, 1], phi_range=[pi, 2*pi], angle_range = [0,0], ts = ts)[1], decimals=5).astype(np.int32)

    y = np.round(ysine[1]+perlin_noise_1-randomized_sum_of_sines(N = N_it+r.randint(3,5), A_range = [0.3, 0.55], omega_range = [0.05, 1], phi_range=[pi, 2*pi], angle_range = [0,0], ts = ts)[1], decimals=5).astype(np.int32)#, decimals=5).astype(np.int32)
    y2 = np.round(y.copy()-perlin_noise_2-k-randomized_sum_of_sines(N = N_it+r.randint(3,5), A_range = [0.3, 0.55], omega_range = [0.05, 1], phi_range=[pi, 2*pi], angle_range = [0,0], ts = ts)[1], decimals=5).astype(np.int32)

    return x, x2, y, y2, z


def seed_positions(D, N_points):
    '''
    Takes dimensions of image and N ridges to create origins for.
    Use to create N points on a 2d plane and is used to initialize points for
    simulating ridges.
    Samples points using uniform distribution.
    '''

    N, M = D

    sample = np.array([np.random.uniform(0, M-1, N_points), np.random.uniform(0, N-1, N_points)])
    sample = np.round(sample, decimals=0).astype(np.int32)
    return sample.T
    



def populate_data(data, ridge_positions):
    '''
    Takes data and x,y pairs for ridge positions.
    Use to generate ridges for every position in ridge positions, 
    thereby populating a 3d data set.
    '''
    D, N, M = data.shape

    
    for position in tqdm(ridge_positions):
        current_data = np.zeros((D, N, M), np.uint8)
        x_origin = position[0]
        y_origin = position[1]


        x, x2, y, y2, z = generate_ridge()
        #x3, x4, y3, y4, z2 = generate_ridge()

        x += x_origin
        x2 += x_origin

        y += y_origin
        y2 += y_origin

        mask = np.logical_and(abs(y)<M, abs(x)<N) #Masks ensure no out of range error
        mask = np.logical_and(mask, abs(z)<D)

        mask2 = np.logical_and(abs(y2)<M, abs(x2)<N)
        mask2 = np.logical_and(mask2, abs(z)<D)

        

        current_data[z[mask], -y[mask], x[mask]] = 1
        current_data[z[mask2], -y2[mask2], x2[mask2]] = 1

        
        current_data = thicken(current_data)
        noisy_data = apply_noise(current_data)

        if not data[noisy_data>0].any() >0: #Never place a line in another
            data += noisy_data
    
    data[data > 0] = 255
    return data.astype(np.uint8)


#Not used in the pipeline, but useful for debugging
def update_img(img, x, y, origin_position):
    '''
    Takes an image, a ridge and an origin position (see function seed_positions)
    Use to update the image one ridge at a time, given a ridge and an origin_position
    '''
    temp_x = x.copy()
    temp_y = y.copy()

    origin_x, origin_y = origin_position

    temp_x+=origin_x
    temp_y+=origin_y
    N, M = img.shape

    mask = np.logical_and(abs(temp_y)<M, abs(temp_x)<N) #Ensures no out of range error

    img[-temp_y[mask],temp_x[mask]]=255

    return img


if __name__ == "__main__":
    

    

    data = np.zeros((600,600,600)).astype(np.uint8) #50 padding on each side for the best ridges (and convolution)
    

    n_ridges=250
    n_test = 8
    sample = seed_positions((600,600), N_points=n_ridges)


    mask = populate_data(data, sample)
    mask = mask[50:550, 50:550, 50:550] #Resize to 500, 500, 500
    mask = mask.astype(np.uint8)

    tif.imwrite("C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/mask_05_v2.tif", mask)




#TODO:
#Improving the ridges:
#   -Add different noise?
#   -Consider attempting placing a new ridge, if the ridge touches another
#
#Finish the data:
#   -Distance transform
#   -Merge with inpainted background




