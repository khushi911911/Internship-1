#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import os
import cv2
import hashlib
from PIL import Image
print("Pillow is installed!")


# In[7]:


train_dir="C:/Users/PIXEL-8Z27/Desktop/proj/test"
test_dir="C:/Users/PIXEL-8Z27/Desktop/proj/train"


# In[ ]:





# data cleaning

# In[3]:


# img_size=(48,48)

# def get_hash(image_path):
#   with open(image_path,"rb") as f:
#     return hashlib.md5(f.read()).hexdigest()

# hashes=set()

# for emotion in os.listdir(train_dir):
#   emotion_path=os.path.join(train_dir,emotion)

#   for filename in os.listdir(emotion_path):
#     img_path=os.path.join(emotion_path,filename)

#     try:
#       img=cv2.imread(img_path)
#       if img is None:
#         print(f"Corrupt image found and removed: {img_path}")
#         os.remove(img_path)
#         continue

#       img_hash=get_hash(img_path)
#       if img_hash in hashes:
#         print(f"Duplicate image found and removed: {img_path}")
#         os.remove(img_path)
#         continue

#       else:
#         hashes.add(img_hash)

#       img=Image.open(img_path).convert("L")
#       img=img.resize(img_size)
#       img.save(img_path)

#     except Exception as e:
#       print(f"Error processing {img_path}: {e}")
#       os.remove(img_path)


# Data Augmentation

# In[5]:


train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

# In[8]:


train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical'
)


# CNN Model

# 

# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential([
    # Convolutional layers
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),


    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    # Flatten and fully connected layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(7, activation='softmax')  # 7 classes for emotions
])

# Show model summary
model.summary()


# In[9]:


from keras.optimizers import SGD

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),  # Using SGD optimizer with momentum
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[12]:


history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=200,
    verbose=1
)


# In[14]:


test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")


# In[14]:


# Save the entire model
model.save('Model2.h5')  # Saves model in .h5 format


# In[1]:


from keras.models import load_model

# Load the model
model = load_model('Model2.h5')

# Now, you can use `model.predict()` for inference


# In[ ]:


# import numpy as np
# from tensorflow.keras.preprocessing import image # type: ignore

# # Load an image for testing
# img_path = "C:/Users/Lenovo/Desktop/Project-1/test/disgust/PrivateTest_3881740.jpg"
# img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
# img_array = image.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)  # Make it batch format

# # Predict emotion
# predictions = model.predict(img_array)
# predicted_class = np.argmax(predictions)

# # Emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# print("Predicted Emotion:", emotion_labels[predicted_class])


# In[2]:


import numpy as np
import cv2
# Emotion labels (Modify as per your dataset)
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y+h, x:x+w]

        # Resize to match model input size (48x48)
        face = cv2.resize(face, (48, 48))

        # Normalize & reshape
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension (grayscale)
        face = face / 255.0  # Normalize

        # Predict emotion
        prediction = model.predict(face)
        emotion_label = np.argmax(prediction)
        emotion_text = emotions[emotion_label]

        # Draw a rectangle around face & put emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Live Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("emotion_model.h5")  # change to your model file
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48))   # resize to training input size
    face = face.astype("float") / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    prediction = model.predict(face)
    return emotion_labels[np.argmax(prediction)]



# %%
