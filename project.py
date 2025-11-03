import cv2
import dlib
import numpy as np

# DLIB SETUP
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to the pretrained dlib landmarks predictor
detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor(predictor_path)  # Landmark predictor

# LOAD GLASSES 
glasses_middle = cv2.imread("glasses_middle.png", cv2.IMREAD_UNCHANGED)  # Middle/straight glasses image
glasses_left = cv2.imread("glasses_left.png", cv2.IMREAD_UNCHANGED)      # Left-tilted glasses
glasses_right = cv2.imread("glasses_right.png", cv2.IMREAD_UNCHANGED)    # Right-tilted glasses

# SETTINGS 
BULGE_STRENGTH = 0.8  # Strength of the face bulge effect

# SMOOTHING VARIABLES 
prev_x, prev_y, prev_roll, prev_scale, prev_angle = 0,0,0,0,0  # Previous positions/angles for smoothing
alpha_pos, alpha_scale, alpha_angle, alpha = 0.2,0.2,0.2,0.5  # Smoothing factors
buffer_len = 5  # Number of frames to average for smoothing glasses movement
nose_buffer, left_eye_buffer, right_eye_buffer = [], [], []  # Buffers to store recent positions

# HELPER FUNCTIONS 
def get_face_center(points):
    return points[30]

def bulge_face_proper(img, points, strength=BULGE_STRENGTH):
    h, w = img.shape[:2]
    cx, cy = get_face_center(points)
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    nx = (X - cx) / w
    ny = (Y - cy) / h
    r = np.sqrt(nx**2 + ny**2)
    r_new = np.clip(r ** (1.0/strength), 0, 1)
    X_new = cx + nx * w * r_new / (r + 1e-6)
    Y_new = cy + ny * h * r_new / (r + 1e-6)
    return cv2.remap(img, X_new.astype(np.float32), Y_new.astype(np.float32), 
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def smooth_value(prev, current, alpha):
    return prev * (1 - alpha) + current * alpha

def smooth_position(prev_x, prev_y, target_x, target_y, alpha=0.5):
    if prev_x == 0:
        return target_x, target_y
    return int(prev_x * (1 - alpha) + target_x * alpha), int(prev_y * (1 - alpha) + target_y * alpha)

def overlay_transparent(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame
    alpha_mask = overlay[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_mask
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (alpha_mask * overlay[:, :, c] + alpha_frame * frame[y:y+h, x:x+w, c])
    return frame

# EYEBROW LIFT TRIGGER 
def eyebrows_lifted(landmarks):
    """
    Detect if eyebrows are lifted enough to trigger sparkles.
    Normalized by eye distance for stability at different camera distances.
    """
    points = np.array([[p.x, p.y] for p in landmarks.parts()])
    left_eyebrow = points[17:22]
    right_eyebrow = points[22:27]
    left_eye = points[36:42]
    right_eye = points[42:48]

    # Distance between eyes
    eye_dist = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
    if eye_dist == 0:
        return False

    # Normalized vertical lift
    left_lift = (np.mean(left_eye[:,1]) - np.mean(left_eyebrow[:,1])) / eye_dist
    right_lift = (np.mean(right_eye[:,1]) - np.mean(right_eyebrow[:,1])) / eye_dist

    # Threshold: adjust between 0.18–0.25 if needed
    normalized_threshold = 0.4
    return left_lift > normalized_threshold or right_lift > normalized_threshold

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Current filter and filter names
current_function = 1
filter_names = {1:"Warp", 2:"Face augmentation", 3:"Motion tracking and interaction"}

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        left_eye = np.mean(points[36:42], axis=0).astype(int)
        right_eye = np.mean(points[42:48], axis=0).astype(int)
        nose = points[27]

        if current_function == 1:
            frame = bulge_face_proper(frame, points, BULGE_STRENGTH)

        elif current_function == 2:
            # Glasses AR with smoothing
            nose_buffer.append(nose)
            left_eye_buffer.append(left_eye)
            right_eye_buffer.append(right_eye)
            if len(nose_buffer) > buffer_len:
                nose_buffer.pop(0)
                left_eye_buffer.pop(0)
                right_eye_buffer.pop(0)
            avg_nose = np.mean(nose_buffer, axis=0).astype(int)
            avg_left_eye = np.mean(left_eye_buffer, axis=0).astype(int)
            avg_right_eye = np.mean(right_eye_buffer, axis=0).astype(int)
            dx = avg_right_eye[0] - avg_left_eye[0]
            dy = avg_right_eye[1] - avg_left_eye[1]
            roll_angle = -np.degrees(np.arctan2(dy, dx))
            roll_angle = prev_roll * (1-alpha) + roll_angle * alpha
            prev_roll = roll_angle
            eye_dist = np.linalg.norm(avg_right_eye - avg_left_eye)
            scale_factor = 2.5
            sung_h, sung_w = glasses_middle.shape[:2]
            new_w = int(sung_w * scale_factor * eye_dist / sung_w)
            new_h = int(sung_h * scale_factor * eye_dist / sung_w)
            smooth_x, smooth_y = smooth_position(prev_x, prev_y, avg_nose[0], avg_nose[1], alpha)
            prev_x, prev_y = smooth_x, smooth_y
            sung_resized = cv2.resize(glasses_middle, (new_w,new_h))
            max_side = int(np.sqrt(new_w**2 + new_h**2))
            canvas = np.zeros((max_side,max_side,4), dtype=np.uint8)
            x_offset = (max_side - new_w)//2
            y_offset = (max_side - new_h)//2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = sung_resized
            M = cv2.getRotationMatrix2D((max_side//2,max_side//2), roll_angle, 1)
            sung_rotated = cv2.warpAffine(canvas, M, (max_side,max_side),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            x1 = int(smooth_x - max_side//2)
            y1 = int(smooth_y - max_side//2)
            frame = overlay_transparent(frame, sung_rotated, x1, y1)

        elif current_function == 3:
            # Glasses AR with sparkle trigger
            sparkle_active = eyebrows_lifted(landmarks)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            roll_angle = -np.degrees(np.arctan2(dy, dx))
            face_center_x = (face.left() + face.right()) / 2
            nose_x = points[30][0]
            yaw = (nose_x - face_center_x) / (face.right() - face.left())

            if yaw < -0.08:
                glasses = glasses_left
                offset_x, offset_y = 25, 25
                scale_factor = 3.5
            elif yaw > 0.08:
                glasses = glasses_right
                offset_x, offset_y = -25, 25
                scale_factor = 3.5
            else:
                glasses = glasses_middle
                offset_x, offset_y = 0, 25
                scale_factor = 2.5

            eye_dist = np.linalg.norm(left_eye - right_eye)
            sung_h, sung_w = glasses.shape[:2]
            new_w = int(sung_w * scale_factor * eye_dist / sung_w)
            new_h = int(sung_h * scale_factor * eye_dist / sung_w)

            if prev_x == 0:
                smooth_x, smooth_y = nose[0], nose[1]
                smooth_scale = new_w
                smooth_angle = roll_angle
            else:
                smooth_x = int(smooth_value(prev_x, nose[0], alpha_pos))
                smooth_y = int(smooth_value(prev_y, nose[1], alpha_pos))
                smooth_scale = int(smooth_value(prev_scale, new_w, alpha_scale))
                smooth_angle = smooth_value(prev_angle, roll_angle, alpha_angle)

            prev_x, prev_y = smooth_x, smooth_y
            prev_scale = smooth_scale
            prev_angle = smooth_angle

            sung_resized = cv2.resize(glasses, (smooth_scale, int(new_h * (smooth_scale / new_w))),
                                      interpolation=cv2.INTER_AREA)
            max_side = int(np.sqrt(sung_resized.shape[0]**2 + sung_resized.shape[1]**2))
            canvas = np.zeros((max_side, max_side, 4), dtype=np.uint8)
            x_offset_canvas = (max_side - sung_resized.shape[1]) // 2
            y_offset_canvas = (max_side - sung_resized.shape[0]) // 2
            canvas[y_offset_canvas:y_offset_canvas+sung_resized.shape[0],
                   x_offset_canvas:x_offset_canvas+sung_resized.shape[1]] = sung_resized

            center = (max_side // 2, max_side // 2)
            M = cv2.getRotationMatrix2D(center, smooth_angle, 1)
            sung_rotated = cv2.warpAffine(canvas, M, (max_side, max_side),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))

            x1 = int(smooth_x - max_side//2 + offset_x)
            y1 = int(smooth_y - max_side//2 - 20 + offset_y)
            x2 = x1 + max_side
            y2 = y1 + max_side

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue

            alpha_mask = sung_rotated[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_mask
            for c in range(3):
                frame[y1:y2, x1:x2, c] = (alpha_mask * sung_rotated[:, :, c] +
                                          alpha_frame * frame[y1:y2, x1:x2, c])

            if sparkle_active:
                alpha_glass = sung_rotated[:, :, 3]
                contours, _ = cv2.findContours(alpha_glass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_shifted = cnt + np.array([[x1, y1]])
                    for i in range(0, len(cnt_shifted), 2):
                        px, py = cnt_shifted[i][0]
                        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                            size = np.random.randint(2, 4)
                            angle = np.random.uniform(0, 2*np.pi)
                            pts = np.array([
                                [px + int(size*np.cos(angle)), py + int(size*np.sin(angle))],
                                [px + int(size*np.cos(angle + 2.1)), py + int(size*np.sin(angle + 2.1))],
                                [px + int(size*np.cos(angle + 4.2)), py + int(size*np.sin(angle + 4.2))]
                            ])
                            color = tuple(np.random.randint(180, 256, 3).tolist())
                            cv2.fillPoly(frame, [pts], color)

    cv2.putText(frame, "Press 1: Warp | 2: face augmentation | 3: Motion tracking and interaction | ESC: Exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Current Filter: {filter_names[current_function]}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Live AR Filters", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1'):
        current_function = 1
    elif key == ord('2'):
        current_function = 2
    elif key == ord('3'):
        current_function = 3

cap.release()
cv2.destroyAllWindows()
