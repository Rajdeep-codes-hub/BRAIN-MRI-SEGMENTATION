import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_dilation

# ---- UNet Architecture ----
class double_convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = double_convolution(in_channels, out_channels)
    def forward(self, x):
        return self.conv(x)

class downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample_block, self).__init__()
        self.Downconv = nn.Sequential(
            nn.MaxPool2d(2,2),
            double_convolution(in_channels, out_channels)
        )
    def forward(self, x):
        return self.Downconv(x)

class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_convolution(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        return self.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = downsample_block(64, 128)
        self.down2 = downsample_block(128, 256)
        self.down3 = downsample_block(256, 512)
        self.down4 = downsample_block(512, 512)
        self.up1 = upsample_block(1024, 256)
        self.up2 = upsample_block(512, 128)
        self.up3 = upsample_block(256, 64)
        self.up4 = upsample_block(128, 64)
        self.outc = OutConv(64, num_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ---- Helper Functions ----
def preprocess_image(img_bytes, target_size=(256, 256)):
    print("Starting image preprocessing...")
    nparr = np.frombuffer(img_bytes, np.uint8)
    print(f"Image bytes length: {len(img_bytes)}")
    img = None

    # Try OpenCV
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is not None:
            print("Image loaded with OpenCV.")
    except Exception as e:
        print(f"OpenCV failed: {e}")

    # Try PIL if OpenCV fails
    if img is None:
        try:
            pil_img = Image.open(BytesIO(img_bytes))
            img = np.array(pil_img)
            print("Image loaded with PIL.")
        except Exception as e:
            print(f"PIL failed: {e}")

    # Try tifffile if PIL fails
    if img is None:
        try:
            img = tifffile.imread(BytesIO(img_bytes))
            print("Image loaded with tifffile.")
        except Exception as e:
            print(f"tifffile failed: {e}")

    if img is None:
        print("All image loading methods failed.")
        raise ValueError("Could not decode image with OpenCV, PIL, or tifffile.")

    print(f"Original image shape: {img.shape}")

    # Convert to 3-channel BGR if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print("Converted grayscale to BGR.")
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        print("Converted BGRA to BGR.")
    elif len(img.shape) == 3 and img.shape[2] == 3:
        print("Image already 3-channel BGR/RGB.")

    orig_shape = img.shape[:2]
    img = cv2.resize(img, target_size)
    print(f"Resized image to {target_size}.")
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    print(f"Image tensor shape: {tensor.shape}")
    return tensor, orig_shape

def postprocess_mask(prediction, orig_shape):
    print("Starting mask postprocessing...")
    mask = prediction.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]))
    print(f"Postprocessed mask shape: {mask.shape}")
    return mask

def generate_plot(img, mask, prediction):
    print("Generating visualization plot...")
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0

    mask_np = mask.squeeze().cpu().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()
    prediction_bin = (prediction_np > 0.5).astype(np.uint8)

    prediction_edges = prediction_bin - binary_dilation(prediction_bin)
    ground_truth_edges = mask_np - binary_dilation(mask_np)

    rgb_img = np.stack([img_gray, img_gray, img_gray], axis=-1)
    rgb_img[ground_truth_edges > 0, 0] = 1  # Red
    rgb_img[prediction_edges > 0, 1] = 1    # Green

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_img)
    ax.axis('off')

    handles = [
        mpatches.Patch(color='red', label='Ground Truth'),
        mpatches.Patch(color='green', label='Predicted Abnormality')
    ]
    fig.legend(handles=handles, loc='upper right')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    print("Visualization plot generated.")
    return buf

# ---- Flask App ----
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = UNet(in_channels=3, num_classes=1)
print("UNet model instantiated.")
model.load_state_dict(torch.load('unet_mri.pth', map_location=device))
print("Model weights loaded.")
model.to(device)
model.eval()
print("Model moved to device and set to eval mode.")

@app.route('/segment', methods=['POST'])
def segment():
    try:
        print("Received /segment request.")
        file = request.files['image']
        img_bytes = file.read()
        print(f"File received: {file.filename}, size: {len(img_bytes)} bytes")

        # Preprocess
        input_tensor, orig_shape = preprocess_image(img_bytes)
        input_tensor = input_tensor.to(device)
        print("Image preprocessed and moved to device.")

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        print("Model inference completed.")

        # Postprocess
        mask = postprocess_mask(output, orig_shape)
        print("Mask postprocessed.")

        # Return mask
        pil_img = Image.fromarray(mask)
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        print("Returning segmentation mask as response.")
        return send_file(buf, mimetype='image/png', download_name='segmentation.png')

    except Exception as e:
        print(f"Error in /segment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        print("Received /visualize request.")
        file = request.files['image']
        img_bytes = file.read()
        print(f"File received: {file.filename}, size: {len(img_bytes)} bytes")

        # Preprocess
        input_tensor, orig_shape = preprocess_image(img_bytes)
        input_tensor = input_tensor.to(device)
        print("Image preprocessed and moved to device.")

        # Get ground truth if available
        mask_tensor = None
        if 'mask' in request.files:
            mask_file = request.files['mask']
            mask_bytes = mask_file.read()
            mask_tensor, _ = preprocess_image(mask_bytes)
            mask_tensor = mask_tensor[:, 0:1, :, :]
            print("Ground truth mask loaded and preprocessed.")
        else:
            mask_tensor = torch.zeros((1, 1, input_tensor.shape[2], input_tensor.shape[3]))
            print("No ground truth mask provided, using dummy mask.")

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        print("Model inference completed.")

        # Generate visualization plot
        plot_buf = generate_plot(input_tensor, mask_tensor, output)
        print("Returning visualization image as response.")
        return send_file(plot_buf, mimetype='image/png', download_name='visualization.png')

    except Exception as e:
        print(f"Error in /visualize: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    print("Received request at root endpoint.")
    return "Brain MRI Segmentation API"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
