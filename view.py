import os
import cv2
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import config as cfg

"""
==================================================
CURRENT PARAMETERS
==================================================
clahe_clip           = 2.1
gaussian_blur        = 0
median_blur          = 0
bilateral            = 0
morph_open           = 0
morph_close          = 0
sharpen              = 0
param1               = 40
param2               = 25
minRadius            = 220
maxRadius            = 245
minDist              = 1060
expand_factor        = 1.4
==================================================
"""
class BBoxVisualizer:
    def __init__(self, csv_path, image_root_dir):
        """
        Visualize bounding boxes from CSV on actual images
        
        Args:
            csv_path: Path to bbox.csv
            image_root_dir: Root directory where images are stored
        """
        self.csv_path = csv_path
        self.image_root_dir = image_root_dir
        self.current_idx = 0
        
        # Load CSV
        print(f"Loading CSV: {csv_path}")
        self.df = pl.read_csv(csv_path)
        print(f"Loaded {len(self.df)} records")
        
        # Create image path lookup
        self.image_paths = self._build_image_lookup()
        
        # Filter to only images that exist
        self.valid_indices = [i for i in range(len(self.df)) 
                             if self.df[i, "Image"] in self.image_paths]
        
        if not self.valid_indices:
            raise ValueError("No matching images found!")
        
        print(f"Found {len(self.valid_indices)}/{len(self.df)} images")
        
        self.setup_gui()
        self.update_display()
    
    def _build_image_lookup(self):
        """Build a dictionary mapping filename -> full path"""
        lookup = {}
        
        print(f"Scanning for images in: {self.image_root_dir}")
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for root, dirs, files in os.walk(self.image_root_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(root, file)
                    # Use just filename as key
                    lookup[file] = full_path
        
        print(f"Found {len(lookup)} total images in directory")
        return lookup
    
    def setup_gui(self):
        """Setup matplotlib GUI"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        ax_info = plt.axes([0.45, 0.05, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, '< Previous')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_info = Button(ax_info, 'Print Info')
        
        self.btn_prev.on_clicked(lambda e: self.prev_image())
        self.btn_next.on_clicked(lambda e: self.next_image())
        self.btn_info.on_clicked(lambda e: self.print_current_info())
    
    def update_display(self):
        """Display current image with bbox"""
        if self.current_idx >= len(self.valid_indices):
            self.current_idx = 0
        
        # Get current record
        actual_idx = self.valid_indices[self.current_idx]
        row = self.df[actual_idx]
        
        img_name = row["Image"][0]
        x1 = row["x1"][0]
        y1 = row["y1"][0]
        x2 = row["x2"][0]
        y2 = row["y2"][0]
        label = row["Label"][0]
        method = row["DetectionMethod"][0]
        
        # Get image path
        if img_name not in self.image_paths:
            print(f"Warning: Image not found: {img_name}")
            return
        
        img_path = self.image_paths[img_name]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading: {img_path}")
            return
        
        # Resize to 1024x1024 (same as processing)
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bbox
        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add info text
        label_text = cfg.blasto_dir_label if label == 1 else cfg.noblasto_dir_label
        cv2.putText(img_with_bbox, f"Label: {label_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img_with_bbox, f"Method: {method}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img_with_bbox, f"BBox: ({x1},{y1}) -> ({x2},{y2})", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display
        self.ax.clear()
        self.ax.imshow(img_with_bbox)
        self.ax.set_title(
            f'[{self.current_idx + 1}/{len(self.valid_indices)}] {img_name}\n'
            f'Path: {img_path}',
            fontsize=10
        )
        self.ax.axis('off')
        plt.draw()
    
    def next_image(self):
        """Go to next image"""
        self.current_idx = (self.current_idx + 1) % len(self.valid_indices)
        self.update_display()
    
    def prev_image(self):
        """Go to previous image"""
        self.current_idx = (self.current_idx - 1) % len(self.valid_indices)
        self.update_display()
    
    def print_current_info(self):
        """Print current image info to console"""
        actual_idx = self.valid_indices[self.current_idx]
        row = self.df[actual_idx]
        
        print("\n" + "="*60)
        print(f"Image: {row['Image'][0]}")
        print(f"Path: {self.image_paths[row['Image'][0]]}")
        print(f"BBox: x1={row['x1'][0]}, y1={row['y1'][0]}, x2={row['x2'][0]}, y2={row['y2'][0]}")
        print(f"Label: {row['Label'][0]} ({cfg.blasto_dir_label if row['Label'][0]==1 else cfg.noblasto_dir_label})")
        print(f"Detection Method: {row['DetectionMethod'][0]}")
        print("="*60 + "\n")
    
    def run(self):
        """Show the visualizer"""
        print("\n=== Controls ===")
        print("Click 'Next >' or 'Previous <' to navigate")
        print("Click 'Print Info' to see details in console")
        print("Close window to exit")
        print("================\n")
        plt.show()


def main():
    # Configuration
    csv_path = cfg.csv_path
    image_root_dir = cfg.embryo_base_path

    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
    
    if not os.path.exists(image_root_dir):
        print(f"Error: Image directory not found at {image_root_dir}")
        return
    
    try:
        visualizer = BBoxVisualizer(csv_path, image_root_dir)
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
