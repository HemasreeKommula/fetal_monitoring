from PIL import Image
import numpy as np

# Load and preprocess the image
img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
img = img.resize((224, 224))                  # Resize to match model input

img_array = np.array(img) / 255.0             # Normalize to [0, 1]
img_array = np.expand_dims(img_array, axis=-1)  # Shape: (224, 224, 1)
img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 224, 224, 1)
