import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
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
param2               = 40
minRadius            = 200
maxRadius            = 245
minDist              = 1010
expand_factor        = 1.4
==================================================
"""
class CircleDetectorGUI:
    def __init__(self, image_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        self.current_idx = 0
        
        # Parameters
        self.params = {
            'clahe_clip': 2.1,
            'gaussian_blur': 0,
            'median_blur': 0,
            'bilateral': 0,
            'morph_open': 0,
            'morph_close': 0,
            'sharpen': 0,
            'param1': 40,
            'param2': 40,
            'minRadius': 200,
            'maxRadius': 245,
            'minDist': 1000,
            'expand_factor': 1.4,
        }
        
        self.updating = False
        self.setup_gui()
        self.update_image()
    
    def setup_gui(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 2, figure=self.fig, height_ratios=[10, 1, 8], 
                     hspace=0.4, wspace=0.3)
        
        # Image displays
        self.ax_preprocessed = self.fig.add_subplot(gs[0, 0])
        self.ax_result = self.fig.add_subplot(gs[0, 1])
        
        # Navigation buttons
        ax_nav = self.fig.add_subplot(gs[1, :])
        ax_nav.axis('off')
        
        ax_prev = plt.axes([0.30, 0.47, 0.08, 0.03])
        ax_update = plt.axes([0.40, 0.47, 0.08, 0.03])
        ax_print = plt.axes([0.50, 0.47, 0.08, 0.03])
        ax_next = plt.axes([0.60, 0.47, 0.08, 0.03])
        
        self.btn_prev = Button(ax_prev, '< Previous')
        self.btn_update = Button(ax_update, 'UPDATE')
        self.btn_print = Button(ax_print, 'Print Params')
        self.btn_next = Button(ax_next, 'Next >')
        
        self.btn_prev.on_clicked(lambda e: self.prev_image())
        self.btn_update.on_clicked(lambda e: self.update_image())
        self.btn_print.on_clicked(lambda e: self.print_params())
        self.btn_next.on_clicked(lambda e: self.next_image())
        
        # Sliders area
        ax_sliders = self.fig.add_subplot(gs[2, :])
        ax_sliders.axis('off')
        
        # Create sliders
        slider_specs = [
            ('clahe_clip', 0.5, 5.0, 'CLAHE Clip', 0.1),
            ('gaussian_blur', 0, 15, 'Gaussian Blur (0=off)', 2),
            ('median_blur', 0, 15, 'Median Blur (0=off)', 2),
            ('bilateral', 0, 15, 'Bilateral (0=off)', 1),
            ('morph_open', 0, 15, 'Morph Open (0=off)', 2),
            ('morph_close', 0, 15, 'Morph Close (0=off)', 2),
            ('sharpen', 0, 1, 'Sharpen (0/1)', 1),
            ('param1', 10, 100, 'Param1', 1),
            ('param2', 10, 100, 'Param2', 1),
            ('minRadius', 20, 500, 'Min Radius', 5),
            ('maxRadius', 50, 500, 'Max Radius', 5),
            ('minDist', 5, 2000, 'Min Distance', 50),
            ('expand_factor', 1.0, 2.5, 'Expand Factor', 0.05),
        ]
        
        self.sliders = {}
        y_start = 0.35
        y_step = 0.025
        
        for i, (name, vmin, vmax, label, step) in enumerate(slider_specs):
            ax = plt.axes([0.15, y_start - i*y_step, 0.7, 0.015])
            slider = Slider(ax, label, vmin, vmax, 
                          valinit=self.params[name], valstep=step)
            slider.on_changed(lambda val, n=name: self.queue_update(n, val))
            self.sliders[name] = slider
    
    def queue_update(self, name, val):
        """Just store parameter value, don't update display"""
        self.params[name] = val
    
    def preprocess(self, gray):
        """Apply preprocessing steps"""
        processed = gray.copy()
        
        # CLAHE (always)
        clahe = cv2.createCLAHE(clipLimit=self.params['clahe_clip'], 
                               tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        
        # Gaussian Blur
        if self.params['gaussian_blur'] > 0:
            ksize = int(self.params['gaussian_blur'])
            if ksize % 2 == 0:
                ksize += 1
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
        
        # Median Blur
        if self.params['median_blur'] > 0:
            ksize = int(self.params['median_blur'])
            if ksize % 2 == 0:
                ksize += 1
            processed = cv2.medianBlur(processed, ksize)
        
        # Bilateral
        if self.params['bilateral'] > 0:
            d = int(self.params['bilateral'])
            processed = cv2.bilateralFilter(processed, d, d*2, d/2)
        
        # Morph Open
        if self.params['morph_open'] > 0:
            ksize = int(self.params['morph_open'])
            if ksize % 2 == 0:
                ksize += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Morph Close
        if self.params['morph_close'] > 0:
            ksize = int(self.params['morph_close'])
            if ksize % 2 == 0:
                ksize += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # Sharpen
        if self.params['sharpen'] > 0:
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    def detect_and_draw(self, img):
        """Detect circles and create visualization"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocess
        processed = self.preprocess(gray)
        
        # Hough Circles
        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(self.params['minDist']),
            param1=int(self.params['param1']),
            param2=int(self.params['param2']),
            minRadius=int(self.params['minRadius']),
            maxRadius=int(self.params['maxRadius'])
        )
        
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        method = "NONE"
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            idx = np.argmax(circles[:, 2])
            cx, cy, r = circles[idx]
            method = f"HOUGH ({len(circles)} circles)"
            
            # Draw detected circle (blue)
            cv2.circle(result, (cx, cy), r, (0, 0, 255), 3)
            
            # Draw expanded circle (red)
            r_exp = int(r * self.params['expand_factor'])
            cv2.circle(result, (cx, cy), r_exp, (255, 0, 0), 3)
            
            # Draw bbox (green)
            x1, y1 = max(0, cx - r_exp), max(0, cy - r_exp)
            x2, y2 = min(w, cx + r_exp), min(h, cy + r_exp)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Text
            cv2.putText(result, f"{method}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(result, f"r: {r} -> {r_exp}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(result, "NO CIRCLES", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return processed, result
    
    def update_image(self):
        """Load and display current image"""
        self.updating = True
        
        img_path = self.image_paths[self.current_idx]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load: {img_path}")
            self.updating = False
            return
        
        img = cv2.resize(img, (1024, 1024))
        
        processed, result = self.detect_and_draw(img)
        
        # Display
        self.ax_preprocessed.clear()
        self.ax_preprocessed.imshow(processed, cmap='gray')
        self.ax_preprocessed.set_title('Preprocessed (CLAHE + filters)')
        self.ax_preprocessed.axis('off')
        
        self.ax_result.clear()
        self.ax_result.imshow(result)
        self.ax_result.set_title(
            f'Detection Result [{self.current_idx+1}/{len(self.image_paths)}]\n'
            f'{os.path.basename(img_path)}'
        )
        self.ax_result.axis('off')
        
        plt.draw()
        self.updating = False
    
    def next_image(self):
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.update_image()
    
    def prev_image(self):
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self.update_image()
    
    def print_params(self):
        """Print current parameters"""
        print("\n" + "="*50)
        print("CURRENT PARAMETERS")
        print("="*50)
        for key, val in self.params.items():
            print(f"{key:20s} = {val}")
        print("="*50 + "\n")
    
    def run(self):
        """Show the GUI"""
        plt.show()


if __name__ == "__main__":
    IMAGE_DIR="/run/media/capitan/Emu/blastodata_orig/BLASTO/D2013.10.01_S0838_I141_2"
    try:
        gui = CircleDetectorGUI(IMAGE_DIR)
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


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
