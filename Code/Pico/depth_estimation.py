import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Use CPU for Raspberry Pi
device = torch.device("cpu")
midas.to(device)

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to MiDaS input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Initialize the camera
CAMERA_INDEX = 0  # Change if using an external USB camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def estimate_depth(frame):
    """Predict depth from a single camera frame using MiDaS."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = midas(input_tensor)  # Run inference

    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)

    return depth_map

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture frame.")
            break

        depth_map = estimate_depth(frame)

        # Display results
        cv2.imshow("Original Image", frame)
        cv2.imshow("Depth Map", depth_map)

        # Save frame every 5 seconds
        cv2.imwrite("depth_map.jpg", depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
