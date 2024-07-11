import os
import numpy as np
import cv2

def convert_png_to_npy(png_dir, npy_dir):
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    
    # Sort the files to ensure consistent numbering
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])
    
    for index, file_name in enumerate(png_files, start=1):
        png_path = os.path.join(png_dir, file_name)
        
        # Create new file name with padding
        new_file_name = f"M01_01_Depth_{index:010d}.npy"
        npy_path = os.path.join(npy_dir, new_file_name)
        
        # Read the PNG image as grayscale
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        print(f"Processing: {file_name}")
        print(f"Original dtype: {image.dtype}")
        print(f"Shape: {image.shape}")
        
        # Convert uint8 to uint16
        image = image.astype(np.uint16)
        image = image * 257  # Scale up the values (optional)
        
        # Save the image as a .npy file
        np.save(npy_path, image)
        print(f'Converted and renamed: {png_path} to {npy_path}')

if __name__ == '__main__':
    png_directory = '/home/filip_lund_andersen/depth_im'  # Replace with your PNG directory path
    npy_directory = '/home/filip_lund_andersen/depth_npy'  # Replace with your NPY directory path
    
    convert_png_to_npy(png_directory, npy_directory)