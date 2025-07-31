import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("fetal_health_model.h5")

# Create a dummy image (grayscale, 128x128)
dummy_image = np.random.rand(1, 128, 128, 1)

# Run prediction
pred = model.predict(dummy_image)

# Display results
print("Prediction probabilities:", pred)
print("Predicted class:", np.argmax(pred))
