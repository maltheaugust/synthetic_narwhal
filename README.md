# synthetic_narwhal
Synthetic data generation for the Narwhal data set



## Required Packages:
  - Numpy
  - skimage
  - scipy
  - Pillow
  - cv2
  - tqdm
  - tifffile (https://pypi.org/project/tifffile/)
  - SimpleITK (https://pypi.org/project/SimpleITK/)
  - perlin (https://pypi.org/project/perlin/)
  - simple_lama_inpainting (https://pypi.org/project/simple-lama-inpainting/)  #for inpainting with the LaMa model


## Usage:
inpaint.py inpaints the dentine ridges out of the data set, requires intermediate mask (we used anders')

generate_mask.py generates N ridges in a 500x500x500 space. Hyperparameters are not parsed into the functions, and thus must be edited manually if so wished.

synthetic.py combines generated ridges and inpainted background, colouring the ridges in the process. Colouring is hardcoded and therefore might not work for other ridge sizes.

ridge_statistics.py returns statistics for a mask. Used to compare ground truth information to synthetic data.
