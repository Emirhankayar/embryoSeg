import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import PIL.ImageDraw as ImageDraw
from view import ImageScroll


def preprocess_embryo_images(
    path_or_pil,
    contrast_factor=1.3,
    clahe_clip=2.5,
    expand_factor=1.5,
    target_size=(224, 224),
    mask_opacity=50,
):
    """
    1) enhance contrast (CLAHE used to help detection),
    2) detect zona pellucida (HoughCircles or contour fallback),
    3) crop externally around the detected circle,
    4) final contrast enhancement + resize to target_size,
    5) return PIL image and processor(...) tensor inputs.
    """
    # load PIL image if path given
    if isinstance(path_or_pil, str):
        pil = Image.open(path_or_pil).convert("RGB")
    else:
        pil = path_or_pil.convert("RGB")

    # Convert to OpenCV format (BGR) for detection
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Use CLAHE to boost local contrast for detection
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(12, 12))
    gray_clahe = clahe.apply(gray)

    # Try HoughCircles first (tends to work well for roughly circular zona)
    circles = cv2.HoughCircles(
        gray_clahe,
        method=cv2.HOUGH_GRADIENT,
        dp=1,  # lower=more accurate, higher=faster
        minDist=1,  # min distance between circles
        param1=50,  # if low -> more edge points
        # accumulator threshold (lower -> more detections)
        param2=120,
        minRadius=30,  # min radius to detect
        maxRadius=160,
    )  # max radius to detect

    h, w = gray.shape
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # choose the circle with largest radius (if multiple)
        idx = np.argmax(circles[:, 2])
        cx, cy, r = circles[idx]
        detection_method = "hough"
    else:
        # fallback: threshold + largest contour -> minEnclosingCircle
        # small blur to reduce noise
        blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
        _, th = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # find contours
        contours, _ = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # last resort: use the center of the image and a big radius
            cx, cy = w // 2, h // 2
            r = int(min(w, h) * 0.4)
            detection_method = "fallback_center"
        else:
            # pick the largest contour by area
            largest = max(contours, key=cv2.contourArea)
            (cx_f, cy_f), r_f = cv2.minEnclosingCircle(largest)
            cx, cy, r = int(cx_f), int(cy_f), int(r_f)
            detection_method = "contour"

    # expand radius a bit to crop *external* to the zona pellucida
    r_exp = int(r * float(expand_factor))

    # bounding box (clamped to image)
    left = max(0, cx - r_exp)
    top = max(0, cy - r_exp)
    right = min(w, cx + r_exp)
    bottom = min(h, cy + r_exp)

    # ensure non-zero area
    if right - left <= 0 or bottom - top <= 0:
        left, top, right, bottom = 0, 0, w, h

    # Crop from original PIL (RGB)
    cropped = pil.crop((left, top, right, bottom))

    # Final contrast enhancement on cropped image
    enhancer = ImageEnhance.Contrast(cropped)
    cropped_contrast = enhancer.enhance(contrast_factor)

    # Optional autocontrast to remove extremes (keeps colors)
    cropped_contrast = ImageOps.autocontrast(cropped_contrast, cutoff=0)

    # Resize to target
    final = cropped_contrast.resize(target_size, resample=Image.BILINEAR)

    # Overlay mask
    tolerance = 20  # pixels
    mask_array = create_circular_mask(pil.size[::-1], (cx, cy), r)
    mask_array = cv2.dilate(
        mask_array,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
        ),
    )
    mask_img = Image.fromarray(mask_array).convert("L")
    mask_rgba = Image.new("RGBA", pil.size, (0, 255, 0, 0))  # green mask
    mask_rgba.putalpha(mask_img.point(lambda p: mask_opacity if p > 0 else 0))
    overlay = pil.convert("RGBA")
    overlay = Image.alpha_composite(overlay, mask_rgba)

    return final, dict(
        detection_method=detection_method,
        circle=(int(cx), int(cy), int(r)),
        bbox=(left, top, right, bottom),
        overlay=overlay,  # image with overlay
        mask=mask_array,  # raw binary mask
    )


def create_circular_mask(image_shape, center, radius):
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = center
    mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8) * 255
    return mask


image_path = "/home/capitan/Documents/blastodata/BLASTO/D2013.02.19_S0675_I141_2"


def process_fn(img):
    pil_img = Image.fromarray(img)
    # best so far cf = 1.4, cc=2.7, ef = 1.4
    preprocessed_pil, info = preprocess_embryo_images(
        pil_img, contrast_factor=1.4, clahe_clip=2.7, expand_factor=1.4, mask_opacity=50
    )
    print("Detection info:", info)
    # can be a mask, overlay etc.
    return np.array(info["overlay"])


ImageScroll(image_path, process_fn)
