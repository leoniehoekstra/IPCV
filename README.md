This project applies real-time AR filters to faces captured from a webcam. Filters include:

Face Warp – A bulge/distortion effect applied to the face.
Face Augmentation – Glasses over the face with rotation tracking.
Motion Tracking & Interaction – Glasses react to face turning and sparkle effect is applied when eyebrows are lifted
The project uses Dlib for facial landmark detection and OpenCV for image processing and display.

Requirements:

Python 3.8+
OpenCV (cv2)
Dlib (dlib)
Numpy (numpy)

Important: Dlib requires the pretrained landmark predictor file: shape_predictor_68_face_landmarks.dat (68-point facial landmark detector)
Download it from Dlib’s official source
Extract .bz2 file to shape_predictor_68_face_landmarks.dat and place it in the project folder.

Controls:

1 – Face Warp
2 – Face Augmentation (glasses with smoothing)
3 – Motion Tracking & Interaction (glasses + sparkles)
ESC – Exit
