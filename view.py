import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import glob


class ImageScroll:
    def __init__(self, image_dir, process_fn=None, interval=100):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_paths)} images")
        self.current_idx = 0
        self.process_fn = process_fn
        self.interval = interval
        self.play_direction = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        ax_play_prev = plt.axes([0.3, 0.02, 0.1, 0.04])
        ax_pause = plt.axes([0.45, 0.02, 0.1, 0.04])
        ax_play_next = plt.axes([0.6, 0.02, 0.1, 0.04])

        self.btn_play_next = Button(ax_play_next, ">")
        self.btn_play_prev = Button(ax_play_prev, "<")
        self.btn_pause = Button(ax_pause, "||")

        self.btn_play_next.on_clicked(self.start_next)
        self.btn_play_prev.on_clicked(self.start_prev)
        self.btn_pause.on_clicked(self.pause)

        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self.auto_play)
        self.timer.start()

        self.display_image()
        plt.show()

    def display_image(self):
        img_path = self.image_paths[self.current_idx]
        print(
            f"\nDisplaying image: {
              os.path.splitext(os.path.basename(img_path))[0]}"
        )
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error loading {img_path}")
            return
        if self.process_fn is not None:
            img = self.process_fn(img)
        self.ax.clear()
        # Check if image is RGB or grayscale
        if len(img.shape) == 3:
            self.ax.imshow(img)  # RGB
        else:
            self.ax.imshow(img, cmap="gray")  # Grayscale
        self.ax.set_title(
            f"Image {
                          self.current_idx + 1}/{len(self.image_paths)}: {os.path.basename(img_path)}"
        )
        self.ax.axis("off")
        plt.draw()

    def next_image(self):
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.display_image()

    def prev_image(self):
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self.display_image()

    def start_next(self, event):
        self.play_direction = 1

    def start_prev(self, event):
        self.play_direction = -1

    def pause(self, event):
        self.play_direction = 0

    def auto_play(self):
        if self.play_direction == 1:
            self.next_image()
        elif self.play_direction == -1:
            self.prev_image()
