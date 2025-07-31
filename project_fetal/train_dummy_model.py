import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Step 1: Generate Dummy Data ---
num_samples = 300
img_height, img_width = 128, 128
num_classes = 3  # Normal, Suspect, Pathological

# Random grayscale images (simulate ultrasound)
X = np.random.rand(num_samples, img_height, img_width, 1).astype('float32')

# Random labels (0=Normal, 1=Suspect, 2=Pathological)
y = np.random.randint(0, num_classes, size=(num_samples,))
y = to_categorical(y, num_classes=num_classes)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# --- Step 2: Build Model ---
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Step 3: Train Model ---
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)

# --- Step 4: Save Model ---
model.save("fetal_health_model.h5")
print("✅ Model saved as fetal_health_model.h5")
