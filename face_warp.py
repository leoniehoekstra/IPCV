import cv2
import numpy as np
from math import pi, sqrt, atan2, cos, sin

def invert_affine_transformation_2x3(affine_matrix):
    affine_matrix_3x3 = np.vstack([affine_matrix, [0, 0, 1]])
    try:
        affine_matrix_inverse_3x3 = np.linalg.inv(affine_matrix_3x3)
        return affine_matrix_inverse_3x3[:2, :]
    except np.linalg.LinAlgError:
        return np.eye(2, 3)

def apply_affine_transformation_to_points(points, transformation_matrix):
    if len(points) == 0:
        return points
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    points_transformed = (transformation_matrix @ points_homogeneous.T).T
    return points_transformed

def convert_to_3x3_matrix(matrix_2x3):
    matrix_3x3 = np.eye(3, dtype=np.float32)
    matrix_3x3[:2,:] = matrix_2x3
    return matrix_3x3

def compose_affine_transformations(transformation_1, transformation_2):
    combined = convert_to_3x3_matrix(transformation_1) @ convert_to_3x3_matrix(transformation_2)
    return combined[:2,:]

def rectangle_to_corner_points(rectangle):
    x, y, width, height = rectangle
    return np.array([[x,y],[x+width,y],[x,y+height],[x+width,y+height]], dtype=np.float32)

def draw_polygon(image, points, color=(0,255,0)):
    points = points.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)

def initialize_face_region_interactive(camera):
    print("Position your face in the center and press SPACE to lock")

    while True:
        success, frame = camera.read()
        if not success:
            print("Error no frame")
            return None, None, None

        frame = cv2.flip(frame, 1)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        height, width = frame.shape[:2]

        cv2.circle(display_frame, (width // 2, height // 2), 5, (0, 255, 0), -1)
        cv2.rectangle(display_frame, (width // 2 - 80, height // 2 - 100), (width // 2 + 80, height // 2 + 100), (0, 255, 0), 2)

        cv2.putText(display_frame, "Position face in center, press SPACE to lock", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press ESC to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Face initialisation - Position & press SPACE", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            break
        elif key == 27:
            return None, None, None

    print("Locking face ROI")

    template_size = 160
    center_x, center_y = width // 2, height // 2
    template_x = max(0, center_x - template_size // 2)
    template_y = max(0, center_y - template_size // 2)
    template_x = min(template_x, width - template_size)
    template_y = min(template_y, height - template_size)

    template = grayscale[template_y:template_y + template_size, template_x:template_x + template_size].copy()

    search_margin = 30
    search_x_min = max(0, center_x - template_size // 2 - search_margin)
    search_y_min = max(0, center_y - template_size // 2 - search_margin)
    search_x_max = min(width - template_size, center_x - template_size // 2 + search_margin)
    search_y_max = min(height - template_size, center_y - template_size // 2 + search_margin)

    search_region = grayscale[search_y_min:search_y_max + template_size,
                              search_x_min:search_x_max + template_size]

    normalized_cross_correlation_map = cv2.matchTemplate(search_region, template, cv2.TM_CCORR_NORMED)
    _, max_correlation_value, _, max_correlation_location = cv2.minMaxLoc(normalized_cross_correlation_map)

    region_of_interest_x = search_x_min + max_correlation_location[0]
    region_of_interest_y = search_y_min + max_correlation_location[1]

    region_of_interest_rectangle = (region_of_interest_x, region_of_interest_y, template_size, template_size)

    print(f"ROI locked at ({region_of_interest_x}, {region_of_interest_y}) (NCC={max_correlation_value:.3f})")
    cv2.destroyWindow("Face initialisation - Position & press SPACE")

    return region_of_interest_rectangle, template, grayscale

def detect_corner_features(grayscale, region_of_interest_rectangle):
    x, y, width, height = region_of_interest_rectangle
    region_of_interest_grayscale = grayscale[y:y+height, x:x+width]

    corner_points = cv2.goodFeaturesToTrack(region_of_interest_grayscale, maxCorners=120,
                                            qualityLevel=0.01, minDistance=5,
                                            blockSize=7, useHarrisDetector=False)

    if corner_points is None or len(corner_points) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    corner_points = corner_points.copy()
    corner_points[:, 0, 0] += x
    corner_points[:, 0, 1] += y

    print(f"Found {len(corner_points)} corners in ROI")
    return corner_points

def track_corner_points(previous_grayscale, current_grayscale, initial_points):
    if initial_points is None or len(initial_points) == 0:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.uint8)

    lucas_kanade_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    tracked_points, tracking_status, error = cv2.calcOpticalFlowPyrLK(previous_grayscale, current_grayscale, initial_points, None, **lucas_kanade_params)

    return tracked_points, tracking_status

def estimate_affine_transformation(previous_points, current_points):
    if len(previous_points) < 3 or len(current_points) < 3:
        return np.eye(2, 3, dtype=np.float32)

    affine_matrix, inliers = cv2.estimateAffine2D(previous_points, current_points, ransacReprojThreshold=3)

    if affine_matrix is None:
        print("error affine?")
        return np.eye(2, 3, dtype=np.float32)

    return affine_matrix

def stabilize_frame_to_canonical(frame, inverse_transformation_matrix, region_of_interest_rectangle, canonical_size=(200, 240)):
    height, width = frame.shape[:2]
    canonical_width, canonical_height = canonical_size

    x_canonical, y_canonical = np.meshgrid(np.arange(canonical_width, dtype=np.float32),
                                           np.arange(canonical_height, dtype=np.float32))

    coordinates_homogeneous = np.stack([x_canonical, y_canonical, np.ones_like(x_canonical)], axis=2)
    coordinates_in_frame = (inverse_transformation_matrix @ coordinates_homogeneous[..., :, np.newaxis]).squeeze(-1)

    mapping_x = coordinates_in_frame[..., 0].astype(np.float32)
    mapping_y = coordinates_in_frame[..., 1].astype(np.float32)

    warped_patch = cv2.remap(frame, mapping_x, mapping_y, cv2.INTER_LINEAR)

    return warped_patch

def warp_radial_bulge_effect(patch, strength=0.35):
    height, width = patch.shape[:2]
    center_x, center_y = width / 2, height / 2

    maximum_radius = np.sqrt((width/2)**2 + (height/2)**2)

    x_destination, y_destination = np.meshgrid(np.arange(width, dtype=np.float32),
                                               np.arange(height, dtype=np.float32))

    x_relative = x_destination - center_x
    y_relative = y_destination - center_y

    radius = np.sqrt(x_relative**2 + y_relative**2)
    angle = np.arctan2(y_relative, x_relative)

    radius_normalized = np.clip(radius / maximum_radius, 0, 1)
    radius_source = radius * (1 - strength * (1 - radius_normalized**2))

    x_source = radius_source * np.cos(angle) + center_x
    y_source = radius_source * np.sin(angle) + center_y

    warped_image = cv2.remap(patch, x_source, y_source, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped_image

def warp_radial_shrink_effect(patch, strength=0.35):
    height, width = patch.shape[:2]
    center_x, center_y = width / 2, height / 2

    maximum_radius = np.sqrt((width/2)**2 + (height/2)**2)

    x_destination, y_destination = np.meshgrid(np.arange(width, dtype=np.float32),
                                               np.arange(height, dtype=np.float32))

    x_relative = x_destination - center_x
    y_relative = y_destination - center_y

    radius = np.sqrt(x_relative**2 + y_relative**2)
    angle = np.arctan2(y_relative, x_relative)

    radius_normalized = np.clip(radius / maximum_radius, 0, 1)
    radius_source = radius * (1 + strength * (1 - radius_normalized**2))

    x_source = radius_source * np.cos(angle) + center_x
    y_source = radius_source * np.sin(angle) + center_y

    warped_image = cv2.remap(patch, x_source, y_source, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped_image

def warp_twist_effect(patch, strength=0.3):
    height, width = patch.shape[:2]
    center_x, center_y = width / 2, height / 2

    maximum_radius = np.sqrt((width/2)**2 + (height/2)**2)

    x_destination, y_destination = np.meshgrid(np.arange(width, dtype=np.float32),
                                               np.arange(height, dtype=np.float32))

    x_relative = x_destination - center_x
    y_relative = y_destination - center_y

    radius = np.sqrt(x_relative**2 + y_relative**2)
    angle = np.arctan2(y_relative, x_relative)

    radius_normalized = radius / maximum_radius
    angle_source = angle + strength * (1 - radius_normalized) * pi

    x_source = radius * np.cos(angle_source) + center_x
    y_source = radius * np.sin(angle_source) + center_y

    warped_image = cv2.remap(patch, x_source, y_source, cv2.INTER_LINEAR)

    return warped_image

def create_soft_mask(size, erosion_pixels=3):
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.ellipse(mask, (width // 2, height // 2), (width // 2 - 10, height // 2 - 10),
                0, 0, 360, 255, -1)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, erosion_kernel, iterations=erosion_pixels // 2)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    return mask

def draw_region_of_interest_and_corner_points(frame, region_of_interest_rectangle, corner_points):
    if region_of_interest_rectangle is not None:
        x, y, width, height = region_of_interest_rectangle
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    if corner_points is not None and len(corner_points) > 0:
        for point in corner_points[:20]:
            x, y = int(point[0, 0]), int(point[0, 1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame

def draw_status_information(frame, status_text_lines):
    vertical_offset = 30
    for line_index, text_line in enumerate(status_text_lines):
        cv2.putText(frame, text_line, (10, vertical_offset + line_index * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    region_of_interest_rectangle = None
    template = None
    initial_corner_points = None
    previous_grayscale = None
    affine_matrix = np.eye(2, 3, dtype=np.float32)

    current_effect_type = 'bulge'
    current_effect_strength = 0.35
    canonical_size = (200, 240)

    frames_since_corner_reseed = 0

    print("=" * 70)
    print("Face warp")
    print("=" * 70)
    print("Press SPACE to initialise face ROI (put face centered)")
    print("Keys:")
    print("  1 = Bulge effect")
    print("  2 = Shrink effect")
    print("  3 = Twist effect")
    print("  + / - keys = increase/decrease effect strength")
    print("  SPACE = re-lock face ROI")
    print("  ESC = quit")
    print("=" * 70)

    frame_number = 0

    while True:
        if region_of_interest_rectangle is None:
            region_of_interest_rectangle, template, grayscale = initialize_face_region_interactive(camera)
            if region_of_interest_rectangle is None:
                break

            roi_x, roi_y, roi_width, roi_height = region_of_interest_rectangle
            canonical_size = (roi_width, roi_height)
            initial_translation_matrix = np.array([[1,0,roi_x],[0,1,roi_y]], dtype=np.float32)
            accumulated_affine_transformation = np.eye(2, 3, dtype=np.float32)

            previous_corner_points = detect_corner_features(grayscale, region_of_interest_rectangle)
            initial_corner_points = previous_corner_points.copy()

            previous_grayscale = grayscale.copy()
            frame_number = 0
            continue

        success, frame = camera.read()
        if not success:
            print("Error no frame")
            break

        frame = cv2.flip(frame, 1)
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_number += 1

        current_corner_points, tracking_status = track_corner_points(previous_grayscale, grayscale, previous_corner_points)

        if tracking_status is not None:
            good_previous_corner_points = previous_corner_points[tracking_status.flatten() == 1]
            good_current_corner_points = current_corner_points[tracking_status.flatten() == 1]
        else:
            good_previous_corner_points = np.empty((0, 1, 2), dtype=np.float32)
            good_current_corner_points = np.empty((0, 1, 2), dtype=np.float32)

        motion_transformation = np.eye(2, 3, dtype=np.float32)
        if len(good_previous_corner_points) >= 3 and len(good_current_corner_points) >= 3:
            motion_transformation = estimate_affine_transformation(good_previous_corner_points.reshape(-1, 2), good_current_corner_points.reshape(-1, 2))

        accumulated_affine_transformation = compose_affine_transformations(motion_transformation, accumulated_affine_transformation)

        if len(good_current_corner_points) < 20:
            frames_since_corner_reseed += 1
            if frames_since_corner_reseed >= 2:
                print(f"Frame {frame_number}: reseedin corners")
                previous_corner_points = detect_corner_features(grayscale, rectangle_to_corner_points(accumulated_affine_transformation @ np.array([[0,0,1],[canonical_size[0],0,1],[0,canonical_size[1],1]]).T).astype(int))
                frames_since_corner_reseed = 0
        else:
            frames_since_corner_reseed = 0

        transformation_canonical_to_image = compose_affine_transformations(accumulated_affine_transformation, initial_translation_matrix)
        transformation_image_to_canonical = invert_affine_transformation_2x3(transformation_canonical_to_image)

        canonical_width, canonical_height = canonical_size
        canonical_patch = cv2.warpAffine(frame, transformation_image_to_canonical, (canonical_width, canonical_height), flags=cv2.INTER_LINEAR)

        if current_effect_type == 'bulge':
            warped_canonical_patch = warp_radial_bulge_effect(canonical_patch, current_effect_strength)
        elif current_effect_type == 'shrink':
            warped_canonical_patch = warp_radial_shrink_effect(canonical_patch, current_effect_strength)
        elif current_effect_type == 'twist':
            warped_canonical_patch = warp_twist_effect(canonical_patch, current_effect_strength)
        else:
            warped_canonical_patch = canonical_patch.copy()

        canonical_mask = create_soft_mask((canonical_width, canonical_height), erosion_pixels=2).astype(np.float32) / 255.0
        full_image_mask = cv2.warpAffine((canonical_mask*255).astype(np.uint8), transformation_canonical_to_image,
                                    (frame.shape[1], frame.shape[0]),
                                    flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        full_image_effect = cv2.warpAffine(warped_canonical_patch, transformation_canonical_to_image,
                                     (frame.shape[1], frame.shape[0]),
                                     flags=cv2.INTER_LINEAR)

        alpha_blending = full_image_mask[..., None] if frame.ndim == 3 else full_image_mask
        frame = (frame.astype(np.float32) * (1 - alpha_blending) +
                 full_image_effect.astype(np.float32) * alpha_blending).astype(np.uint8)

        canonical_corner_points = np.array([[0,0],[canonical_width,0],[0,canonical_height],[canonical_width,canonical_height]], dtype=np.float32)
        region_polygon = apply_affine_transformation_to_points(canonical_corner_points, transformation_canonical_to_image)
        draw_polygon(frame, region_polygon, (0, 255, 0))

        previous_corner_points = good_current_corner_points.reshape(-1, 1, 2) if len(good_current_corner_points) > 0 else previous_corner_points

        status_text_lines = [
            f"Effect: {current_effect_type.upper()} (Strength: {current_effect_strength:.2f})",
            f"Tracked points: {len(good_current_corner_points)} / {len(previous_corner_points)}",
            f"1=bulge  2=shrink effect  3=twisting  +/-=strength  SPACE=face relock  ESC=Quit"
        ]
        frame = draw_status_information(frame, status_text_lines)

        cv2.imshow("Face warp", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Exit")
            break
        elif key == ord('1'):
            current_effect_type = 'bulge'
            print("Effect: BULGE")
        elif key == ord('2'):
            current_effect_type = 'shrink'
            print("Effect: SHRINK")
        elif key == ord('3'):
            current_effect_type = 'twist'
            print("Effect: TWIST")
        elif key == ord('+') or key == ord('='):
            current_effect_strength = min(0.6, current_effect_strength + 0.05)
            print(f"Strength: {current_effect_strength:.2f}")
        elif key == ord('-'):
            current_effect_strength = max(0.0, current_effect_strength - 0.05)
            print(f"Strength: {current_effect_strength:.2f}")
        elif key == ord(' '):
            print("Re-locking face ROI")
            region_of_interest_rectangle = None
            template = None
            initial_corner_points = None

        previous_grayscale = grayscale.copy()

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
