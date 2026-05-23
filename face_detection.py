import cv2,os,numpy as np
import matplotlib.pyplot as plt

# Load image
imagePath = "face1.jpeg"
img = cv2.imread(imagePath)

# Convert to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Detect faces
faces = face_classifier.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40, 40)
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

#Extracted faces

# Create output folder
output_folder = "/Users/naiwritaborah/Gitam/Dec-April 2025-26/CSEN1031 Artificial Intelligence Applications/Extracted Faces"

# Draw rectangles and extract faces
face_count = 0

for (x, y, w, h) in faces:
    face_count += 1

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Crop face from original image
    face_roi = img[y:y+h, x:x+w]

    # Sharpening kernel
    sharpen_kernel = [[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]]


    kernel = np.array(sharpen_kernel)
    face_roi = cv2.filter2D(face_roi, -1, kernel)

    # Save inside folder
    output_path = os.path.join(output_folder, f"face_{face_count}.jpeg")
    cv2.imwrite(output_path, face_roi)

# Convert for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Total faces extracted: {face_count}")
print(f"Faces saved inside folder: {output_folder}")
