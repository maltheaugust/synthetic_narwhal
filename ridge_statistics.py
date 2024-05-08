import tifffile as tif
import numpy as np
import cc3d
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d
from tqdm import tqdm


def return_statistics(ridges_data):
    '''
    This function computes statistics for ridge data. Used to compare synthetic data to true data
    '''
    cc = cc3d.connected_components(ridges_data)

    distance_transform = distance_transform_edt(ridges_data)
    skeleton = skeletonize_3d(ridges_data)

    skeleton_thickness = distance_transform[skeleton==255] #Thickness statistics
    thickness = (skeleton_thickness.mean(), skeleton_thickness.std())

    slopes_x = np.zeros((cc.max()))
    slopes_y = np.zeros((cc.max()))

    for i in tqdm(range(1,cc.max())): 
        current_ridge = (cc == i).astype(np.uint8)

        z_coords, y_coords, x_coords = np.where(current_ridge>0)


        run = z_coords.max() - z_coords.min()
        if run==0:
            continue

        slopes_y[i] = (y_coords.max()-y_coords.min()) / run
        slopes_x[i] = (x_coords.max()-x_coords.min()) / run




    x_mask = np.logical_and(np.isnan(slopes_x)==False, np.isinf(slopes_x)==False)
    y_mask = np.logical_and(np.isnan(slopes_y)==False, np.isinf(slopes_y)==False)

    x_slope = (slopes_x[x_mask].mean(), slopes_x[x_mask].std())
    y_slope = (slopes_y[y_mask].mean(), slopes_y[y_mask].std())

    #Compute the density
    flattened_ridges = np.reshape(ridges_data, -1)
    density = sum(flattened_ridges==255)/len(flattened_ridges)


    return density, thickness, x_slope, y_slope



if __name__ == "__main__":

    ridges_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/annotated2.tif"
    ridges_data = tif.imread(ridges_path)

    density, thickness, x_slope, y_slope = return_statistics(ridges_data=ridges_data)

    print(f"The density ridges in the data is: {density}\n")
    print(f"The mean thickness of the ridges is: {thickness[0]} with a spread of: {thickness[1]}\n")
    print(f"The mean slope of the ridges in x is {x_slope[0]} with a spread of: {x_slope[1]}\n")
    print(f"The mean slope of the ridges in y is {y_slope[0]} with a spread of: {y_slope[1]}")

# %%
