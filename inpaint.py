from simple_lama_inpainting import SimpleLama
from PIL import Image
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2


if __name__ == "__main__":
    save_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/"
    img_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/scan_07_crop.tif"
    mask_path = "C:/Users/malth/Documents/DTU/Sjette Semester/Bachelor/Data/scan_07_crop_mask.tif"
    d_kernel = np.ones((5,5), np.uint8)

    image_0 = tif.imread(img_path)
    mask_0 = tif.imread(mask_path)

    image_1 = np.rollaxis(image_0, axis=1)
    mask_1 = np.rollaxis(mask_0, axis=1)

    image_2 = np.rollaxis(image_0, axis=2)
    mask_2 = np.rollaxis(mask_0, axis=2)


    simple_lama = SimpleLama()

    M, N, C = image_0.shape


    images = [image_0, image_1, image_2]
    masks = [mask_0, mask_1, mask_2]
    results = [np.zeros((M, N, C))]*3 #I don't remember why I did this, TODO: fix it?


    for i in tqdm(range(M)):
        img_i = Image.fromarray(images[1][i]).convert('RGB')

        _, cc, stats, _ = cv2.connectedComponentsWithStats(masks[1][i])

        to_keep = np.squeeze(np.argwhere(stats[:,4]>10))

        for k in range(1,len(to_keep)): #Get rid of connected components smaller than a threshold, so inpainting does not inpaint blobs in the backgroudn
            keep_val = to_keep[k]
            cc[cc==keep_val] = 255
        
        cc[cc<255] = 0

        mask_i_dilated = cv2.dilate(cc.astype(np.uint8), d_kernel, iterations=2) #Dilate for better inpainting

        mask_i = Image.fromarray(mask_i_dilated, mode="L")

        results[1][i] = np.array(simple_lama(img_i, mask_i).convert('L'))[3:503,3:503] #Get rid of padding


    result = (results[1]).astype(np.uint8)

    tif.imwrite(save_path+"inpaint_01_v2.tif", result)