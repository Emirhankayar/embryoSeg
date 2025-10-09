import os
import cv2
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import config as cfg

"""
VERIFIED BEST PARAMETERS:
clahe_clip           = 2.0
param1               = 40
param2               = 30
minRadius            = 215
maxRadius            = 250
minDist              = 1000
expand_factor        = 1.4
"""


class BatchCircleDetector:
    def __init__(
        self,
        clahe_clip=2.0,
        param1=40,
        param2=30,
        minRadius=215,
        maxRadius=250,
        minDist=1000,
        expand_factor=1.4,
        target_size=(1024, 1024),
        num_workers=None,
    ):
        self.clahe_clip = clahe_clip
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.minDist = minDist
        self.expand_factor = expand_factor
        self.target_size = target_size
        self.num_workers = num_workers or multiprocessing.cpu_count()

    def preprocess(self, gray):
        """Apply CLAHE preprocessing"""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        processed = clahe.apply(gray)
        return processed

    def detect_bbox(self, img):
        """
        Detect circle and return bounding box
        Returns: (x1, y1, x2, y2, detection_method)
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocess
        processed = self.preprocess(gray)

        # Hough Circles
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=self.minDist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.minRadius,
            maxRadius=self.maxRadius,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            idx = np.argmax(circles[:, 2])
            cx, cy, r = circles[idx]
            method = "hough"
        else:
            blur = cv2.GaussianBlur(processed, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest = max(contours, key=cv2.contourArea)
                (cx, cy), r = cv2.minEnclosingCircle(largest)
                cx, cy, r = int(cx), int(cy), int(r)
                method = "contour"
            else:
                # Last resort: center fallback
                cx, cy = w // 2, h // 2
                r = int(min(w, h) * 0.4)
                method = "fallback"

        # Expand and create bbox
        r_exp = int(r * self.expand_factor)
        x1 = max(0, cx - r_exp)
        y1 = max(0, cy - r_exp)
        x2 = min(w, cx + r_exp)
        y2 = min(h, cy + r_exp)

        return x1, y1, x2, y2, method

    @staticmethod
    def process_single_image(args):
        img_path, params, target_size = args

        try:
            path_parts = Path(img_path).parts

            label = None
            for part in path_parts:
                if part == cfg.blasto_dir_label:
                    label = 1
                    break
                elif part == cfg.noblasto_dir_label:
                    label = 0
                    break

            if label is None:
                return None, f"Could not determine label for {img_path}"

            # Load and resize image
            img = cv2.imread(img_path)
            if img is None:
                return None, f"Failed to load {img_path}"

            # Resize BEFORE any preprocessing
            img_resized = cv2.resize(img, target_size)

            # Create detector instance for this worker
            detector = BatchCircleDetector(**params)

            # Detect bbox
            x1, y1, x2, y2, method = detector.detect_bbox(img_resized)

            # Get just the image filename
            img_name = os.path.basename(img_path)

            result = {
                "Image": img_name,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "Label": label,
                "DetectionMethod": method,
            }

            return result, None

        except Exception as e:
            return None, f"Error processing {img_path}: {str(e)}"

    def process_directory(self, root_dir, output_csv="bbox.csv"):
        image_extensions = (".jpg")

        print(f"Scanning directory: {root_dir}")
        all_files = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)

        print(f"Found {len(all_files)} images")
        print(f"Using {self.num_workers} worker processes")

        params = {
            "clahe_clip": self.clahe_clip,
            "param1": self.param1,
            "param2": self.param2,
            "minRadius": self.minRadius,
            "maxRadius": self.maxRadius,
            "minDist": self.minDist,
            "expand_factor": self.expand_factor,
            "target_size": self.target_size,
        }

        args_list = [(img_path, params, self.target_size) for img_path in all_files]

        results = []
        errors = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.process_single_image, args): args[0]
                for args in args_list
            }

            # Process results as they complete
            with tqdm(total=len(all_files), desc="Processing images") as pbar:
                for future in as_completed(futures):
                    result, error = future.result()

                    if error:
                        errors.append(error)
                    elif result:
                        results.append(result)

                    pbar.update(1)

        if errors:
            print(f"\n{len(errors)} errors occurred:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        # Dataframe creation
        if results:
            df = pl.DataFrame(results)
            df.write_csv(output_csv)
            print(f"\nProcessed {len(results)} images successfully")
            print(f"Results saved to: {output_csv}")

            # Print summary
            print("\n=== Summary ===")
            print(df.group_by("Label").agg(pl.count()).sort("Label"))
            print(f"\nDetection methods:")
            print(df.group_by("DetectionMethod").agg(pl.count()))
        else:
            print("No images were processed successfully")

        return df if results else None

def main():
    detector = BatchCircleDetector(
        clahe_clip=2.0,
        param1=40,
        param2=30,
        minRadius=215,
        maxRadius=250,
        minDist=1000,
        expand_factor=1.4,
        target_size=(1024, 1024),
        num_workers=None,
    )

    root_dir = cfg.embryo_base_path
    output_csv = cfg.csv_path

    df = detector.process_directory(root_dir, output_csv)

    if df is not None:
        print("\n=== First 5 rows ===")
        print(df.head(5))


if __name__ == "__main__":
    main()
