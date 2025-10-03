import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoImageProcessor
from view import ImageScroll

# assume processor already loaded:
# processor = AutoImageProcessor.from_pretrained("ihlab/FEMI", use_fast=True)


def preprocess_embryo_image_for_vitmae(
    path_or_pil,
    contrast_factor=1.3,
    clahe_clip=2.5,
    expand_factor=1.5,
    target_size=(224, 224),
    visualize=False,
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

    if visualize:
        # show detection overlay and final crop (matplotlib required)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].set_title("Original")
        axes[0].imshow(pil)
        axes[0].axis("off")
        # overlay circle
        overlay = pil.copy()
        draw = overlay.convert("RGBA")
        import PIL.ImageDraw as ImageDraw

        d = ImageDraw.Draw(draw)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  outline=(255, 0, 0, 180), width=3)
        axes[1].set_title(f"Detected (method={detection_method})")
        axes[1].imshow(draw)
        axes[1].axis("off")
        axes[2].set_title("Cropped -> enhanced -> resized")
        axes[2].imshow(final)
        axes[2].axis("off")
        plt.show()

    return final, dict(
        detection_method=detection_method,
        circle=(int(cx), int(cy), int(r)),
        bbox=(left, top, right, bottom),
    )


image_path = "/home/capitan/Documents/blastodata/BLASTO/D2013.02.19_S0675_I141_2"

def process_fn(img):
    from PIL import Image

    pil_img = Image.fromarray(img)
    preprocessed_pil, info = preprocess_embryo_image_for_vitmae(
        pil_img, contrast_factor=1.4, expand_factor=1.4, visualize=False
    )
    print("Detection info:", info)
    return cv2.cvtColor(np.array(preprocessed_pil), cv2.COLOR_RGB2GRAY)


ImageScroll(image_path, process_fn)
