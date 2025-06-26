# BRAIN-MRI-SEGMENTATION
Overview
This project implements a brain MRI segmentation system using a deep learning model (UNet) to identify abnormal regions in brain MRI scans. The backend is built with Flask and PyTorch, while the frontend is a React application providing a user-friendly interface for uploading MRI images and visualizing segmentation results.

Features
Robust UNet model for brain MRI segmentation.

Supports TIFF, PNG, JPG image formats with robust preprocessing.

Flask backend with CORS enabled for React frontend compatibility.

React frontend with modern UI for image upload, segmentation, and visualization.

Visualization overlays predicted masks on MRI images.

Dataset
The model is trained on the LGG MRI Segmentation dataset from Kaggle. This dataset contains brain MRI images and corresponding tumor masks.

Dataset Structure
Each patient folder contains MRI images (*.tif) and corresponding mask files (*_mask.tif).

Not all slices contain tumors; masks with non-zero pixels indicate tumor presence.

Backend
Built with Flask and PyTorch.

Implements UNet architecture for segmentation.

Handles image preprocessing including TIFF support using OpenCV, Pillow, and tifffile.

Provides /segment endpoint for segmentation mask output.

Provides /visualize endpoint for overlay visualization of predicted masks.

CORS enabled for frontend integration.

Frontend
Built with React.

Allows users to upload brain MRI images.

Provides buttons to run segmentation and visualize predicted masks.

Displays original image, segmentation mask, and overlay visualization.

Modern, responsive, and visually appealing UI.

Installation
Backend
Clone the repository.

Install Python dependencies:

text
pip install -r requirements.txt
Place the pretrained model file unet_mri.pth in the backend directory.

Run the Flask app:

text
python app.py
Frontend
Navigate to the frontend directory.

Install dependencies:

text
npm install
Start the React app:

text
npm start
Usage
Open the React frontend in your browser (usually at http://localhost:3000).

Upload a brain MRI image (TIFF, PNG, JPG).

Click "Get Segmentation Mask" to get the binary mask.

Click "Visualize Predicted Mask" to see the mask overlay on the original image.

Notes
The model expects 3-channel images resized to 256x256 pixels.

Grayscale images are automatically converted to 3-channel.

For best results, use images similar to the training dataset.

The backend includes detailed logging for debugging.

Troubleshooting
If segmentation fails, check backend logs for errors.

Ensure tifffile and other dependencies are installed.

Enable CORS in backend for frontend communication.

License
This project is licensed under the MIT License.

Acknowledgments
Dataset from Kaggle: LGG MRI Segmentation by Mateusz Buda.

UNet architecture inspired by original paper by Olaf Ronneberger et al.

Frontend design inspired by modern UI principles.
