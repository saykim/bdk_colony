import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QFileDialog, QMessageBox,
    QDoubleSpinBox, QSizePolicy, QGroupBox, QGridLayout,
    QMainWindow, QAction, QStyle, QStyleFactory,
    QComboBox, QWidgetAction, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
import json
import datetime
from PIL import Image

class SAMLoaderThread(QThread):
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_type, checkpoint, device):
        super().__init__()
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device

    def run(self):
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            self.loaded.emit(predictor)
        except Exception as e:
            self.error.emit(str(e))

class PredictionThread(QThread):
    predicted = pyqtSignal(object)  # 수정: object_count 제거
    error = pyqtSignal(str)

    def __init__(self, predictor, input_points, input_labels):
        super().__init__()
        self.predictor = predictor
        self.input_points = np.array(input_points)
        self.input_labels = np.array(input_labels)

    def run(self):
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=self.input_points,
                point_labels=self.input_labels,
                multimask_output=False
            )
            mask = masks[0]
            self.predicted.emit(mask)  # 수정: object_count 제거
        except Exception as e:
            self.error.emit(str(e))

class InteractiveMaskGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive SAM Control Panel")
        self.setMinimumSize(1400, 900)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QHBoxLayout(self.main_widget)

        # Create side panel with vertical layout
        self.control_panel = QVBoxLayout()
        main_layout.addLayout(self.control_panel, 0)

        # Initialize variables
        self.iou_var = 0.8
        self.score_var = 0.5
        self.contrast_var = 1.0
        self.grayscale_var = False
        self.morphology_var = False
        self.sharpen_var = False
        self.edge_enhance_var = False
        self.min_colony_size = 20.0
        self.max_colony_size = 300.0

        self.input_points = []
        self.input_labels = []

        # Setup GUI controls
        self.setup_controls()

        # Setup matplotlib figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas, 1)

        # Initialize status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")

        # Initialize SAM components in a separate thread
        self.setup_sam()

        # Setup interactive events
        self.setup_interactive()

        self.set_status("Ready")

        # Setup menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction(QIcon.fromTheme("document-open"), "Load Image", self)
        load_action.setShortcut("Ctrl+O")
        load_action.setStatusTip("Load an image")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "Save Image", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save the current results")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Theme menu
        theme_menu = menubar.addMenu("&Theme")

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        theme_action = QWidgetAction(self)
        theme_action.setDefaultWidget(self.theme_combo)
        theme_menu.addAction(theme_action)

    def change_theme(self, theme):
        if theme == "Dark":
            QApplication.instance().setStyle(QStyleFactory.create("Fusion"))
            dark_palette = self.create_dark_palette()
            QApplication.instance().setPalette(dark_palette)
            self.canvas.setStyleSheet("background-color: #2b2b2b;")
        else:
            QApplication.instance().setStyle(QStyleFactory.create("Fusion"))
            QApplication.instance().setPalette(QApplication.style().standardPalette())
            self.canvas.setStyleSheet("background-color: #ffffff;")
        self.update()

    def create_dark_palette(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        return dark_palette

    def setup_controls(self):
        """Setup control panel widgets"""
        # SAM Parameters Group
        sam_group = QGroupBox("SAM Parameters")
        sam_layout = QVBoxLayout()

        # IOU Threshold
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU Threshold")
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.iou_var)
        self.iou_spin.valueChanged.connect(self.delayed_update)
        self.iou_spin.setToolTip("Set the Intersection over Union (IOU) threshold")
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_spin)
        sam_layout.addLayout(iou_layout)

        # Score Threshold
        score_layout = QHBoxLayout()
        score_label = QLabel("Score Threshold")
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setValue(self.score_var)
        self.score_spin.valueChanged.connect(self.delayed_update)
        self.score_spin.setToolTip("Set the score threshold")
        score_layout.addWidget(score_label)
        score_layout.addWidget(self.score_spin)
        sam_layout.addLayout(score_layout)

        # Colony Size Filters
        colony_size_layout = QHBoxLayout()
        colony_min_label = QLabel("Min Colony Size")
        self.colony_min_spin = QDoubleSpinBox()
        self.colony_min_spin.setRange(0.0, 100000.0)
        self.colony_min_spin.setSingleStep(100.0)
        self.colony_min_spin.setValue(self.min_colony_size)
        self.colony_min_spin.valueChanged.connect(self.delayed_update)
        self.colony_min_spin.setToolTip("Set the minimum colony size (in pixels)")

        colony_max_label = QLabel("Max Colony Size")
        self.colony_max_spin = QDoubleSpinBox()
        self.colony_max_spin.setRange(0.0, 100000.0)
        self.colony_max_spin.setSingleStep(100.0)
        self.colony_max_spin.setValue(self.max_colony_size)
        self.colony_max_spin.valueChanged.connect(self.delayed_update)
        self.colony_max_spin.setToolTip("Set the maximum colony size (in pixels)")

        colony_size_layout.addWidget(colony_min_label)
        colony_size_layout.addWidget(self.colony_min_spin)
        colony_size_layout.addWidget(colony_max_label)
        colony_size_layout.addWidget(self.colony_max_spin)
        sam_layout.addLayout(colony_size_layout)

        sam_group.setLayout(sam_layout)
        self.control_panel.addWidget(sam_group)

        # Image Processing Group
        img_proc_group = QGroupBox("Image Processing")
        img_proc_group_layout = QVBoxLayout()

        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Contrast")
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 2.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(self.contrast_var)
        self.contrast_spin.valueChanged.connect(self.delayed_update)
        self.contrast_spin.setToolTip("Adjust the image contrast")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_spin)
        img_proc_group_layout.addLayout(contrast_layout)

        # Checkboxes
        self.grayscale_cb = QCheckBox("Grayscale")
        self.grayscale_cb.stateChanged.connect(self.delayed_update)
        self.grayscale_cb.setToolTip("Convert image to grayscale")
        img_proc_group_layout.addWidget(self.grayscale_cb)

        self.morphology_cb = QCheckBox("Morphology")
        self.morphology_cb.stateChanged.connect(self.delayed_update)
        self.morphology_cb.setToolTip("Apply morphological operations")
        img_proc_group_layout.addWidget(self.morphology_cb)

        self.sharpen_cb = QCheckBox("Sharpen")
        self.sharpen_cb.stateChanged.connect(self.delayed_update)
        self.sharpen_cb.setToolTip("Sharpen the image")
        img_proc_group_layout.addWidget(self.sharpen_cb)

        # Edge Enhancement
        self.edge_enhance_cb = QCheckBox("Edge Enhancement")
        self.edge_enhance_cb.stateChanged.connect(self.delayed_update)
        self.edge_enhance_cb.setToolTip("Enhance edges in the image")
        img_proc_group_layout.addWidget(self.edge_enhance_cb)

        img_proc_group.setLayout(img_proc_group_layout)
        self.control_panel.addWidget(img_proc_group)

        # Buttons Group
        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Image")
        self.load_btn.setIcon(QIcon.fromTheme("document-open"))
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setToolTip("Load an image file")
        buttons_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setToolTip("Save the current results")
        buttons_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset Points")
        self.reset_btn.setIcon(QIcon.fromTheme("edit-clear"))
        self.reset_btn.clicked.connect(self.reset_points)
        self.reset_btn.setToolTip("Reset all input points")
        buttons_layout.addWidget(self.reset_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setIcon(QIcon.fromTheme("application-exit"))
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setToolTip("Quit the application")
        buttons_layout.addWidget(self.quit_btn)

        buttons_group.setLayout(buttons_layout)
        self.control_panel.addWidget(buttons_group)

        # Add stretch to push widgets to the top
        self.control_panel.addStretch()

    def set_status(self, message):
        """Update status message"""
        if len(message) > 50:
            message = message[:47] + "..."
        self.status_bar.showMessage(message)

    def setup_sam(self):
        """Initialize SAM model in a separate thread"""
        self.set_status("Loading SAM model...")

        # Check if model file exists
        self.sam_checkpoint = Path("sam_vit_b_01ec64.pth")
        if not self.sam_checkpoint.is_file():
            QMessageBox.critical(self, "Error", f"SAM model file not found: {self.sam_checkpoint}")
            sys.exit(1)

        self.model_type = "vit_b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam_loader_thread = SAMLoaderThread(self.model_type, str(self.sam_checkpoint), self.device)
        self.sam_loader_thread.loaded.connect(self.on_sam_loaded)
        self.sam_loader_thread.error.connect(self.on_sam_error)
        self.sam_loader_thread.start()

    def on_sam_loaded(self, predictor):
        """Handle SAM model loaded signal"""
        self.predictor = predictor
        self.set_status("SAM model loaded successfully")

    def on_sam_error(self, error_message):
        """Handle SAM model load error"""
        QMessageBox.critical(self, "Error", f"Failed to initialize SAM model: {error_message}")
        self.set_status("Error loading SAM model")
        sys.exit(1)

    def load_image(self):
        """Load and process new image with Windows compatibility"""
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            image_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an image file",
                "",
                "Image files (*.jpg *.jpeg *.png *.bmp)",
                options=options
            )

            if not image_path:
                return

            self.load_image_from_path(image_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            self.set_status("Error loading image")

    def load_image_from_path(self, image_path):
        """Load image from a given path"""
        try:
            self.set_status("Loading image...")

            # Convert path to Path object for better OS compatibility
            image_path = Path(image_path).resolve()

            # Load image using PIL for better compatibility
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                self.original_image = np.array(img)

            # Convert RGB to BGR for OpenCV
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)

            # Limit image size if needed
            max_size = 1024
            h, w = self.original_image.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                self.original_image = cv2.resize(self.original_image, new_size, interpolation=cv2.INTER_AREA)

            self.update_image()
            self.reset_points()
            self.set_status("Image loaded successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            self.set_status("Error loading image")

    def update_image(self):
        """Update image with current processing parameters"""
        if hasattr(self, 'original_image'):
            try:
                self.set_status("Processing image...")

                # Process image
                self.processed_image = self.preprocess_image(
                    self.original_image.copy(),
                    self.grayscale_cb.isChecked(),
                    self.contrast_spin.value(),
                    self.morphology_cb.isChecked(),
                    self.sharpen_cb.isChecked(),
                    self.edge_enhance_cb.isChecked()
                )

                # Update SAM predictor
                self.predictor.set_image(self.processed_image)

                # Store current axis limits if they exist
                if self.ax.images:
                    xlim = self.ax.get_xlim()
                    ylim = self.ax.get_ylim()
                else:
                    xlim = ylim = None

                # Update display
                self.ax.clear()
                self.ax.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))

                # Remove ticks
                self.ax.set_xticks([])
                self.ax.set_yticks([])

                self.canvas.draw()

                # Reapply points and predictions if they exist
                if self.input_points:
                    self.predict_and_show()

                self.set_status("Image processing complete")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Image processing failed: {str(e)}")
                self.set_status("Error processing image")

    def delayed_update(self):
        """Delayed update to prevent excessive processing"""
        QTimer.singleShot(500, self.update_image)

    def preprocess_image(self, image, grayscale=False, contrast=1.0, morphology=False, sharpen=False, edge_enhance=False):
        """Preprocess image with selected options"""
        try:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to float32 for processing
            image = image.astype(np.float32) / 255.0

            if grayscale:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                image = np.stack([gray, gray, gray], axis=-1) / 255.0

            if contrast != 1.0:
                image = np.clip(image * contrast, 0, 1)

            if morphology:
                kernel = np.ones((3, 3), np.uint8)
                image = cv2.morphologyEx((image * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                image = image.astype(np.float32) / 255.0

            if sharpen:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                image = cv2.filter2D(image, -1, kernel)
                image = np.clip(image, 0, 1)

            if edge_enhance:
                # Apply Canny edge detection
                edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
                # Convert edges to RGB
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) / 255.0
                # Overlay edges on the original image
                image = np.clip(image + edges_rgb, 0, 1)

            # Convert back to BGR uint8
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")

    def setup_interactive(self):
        """Setup interactive matplotlib events"""
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.suptitle("Left click: Object Point (green)\nRight click: Background Point (red)")

        # Enable drag and drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle drop event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                image_path = urls[0].toLocalFile()
                self.load_image_from_path(image_path)

    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes == self.ax:
            clicked_point = np.array([event.xdata, event.ydata])
            tolerance = 10  # 픽셀 단위의 허용 오차

            # 현재 포인트들 중 가장 가까운 포인트를 찾습니다.
            if self.input_points:
                points_array = np.array(self.input_points)
                distances = np.linalg.norm(points_array - clicked_point, axis=1)
                min_distance = distances.min()
                min_index = distances.argmin()

                if min_distance < tolerance:
                    # 해당 포인트가 이미 존재하면 삭제합니다.
                    del self.input_points[min_index]
                    del self.input_labels[min_index]
                    self.predict_and_show()
                    return

            # 새로운 포인트 추가
            if event.button == 1:  # Left click for object
                label = 1
            elif event.button == 3:  # Right click for background
                label = 0
            else:
                return

            self.input_points.append([event.xdata, event.ydata])
            self.input_labels.append(label)
            self.predict_and_show()

    def reset_points(self):
        """Reset all points and update display"""
        self.input_points = []
        self.input_labels = []
        if hasattr(self, 'processed_image'):
            # Store current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))

            # Restore axis limits and remove ticks
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.canvas.draw()
            self.set_status("Points reset")

    def show_mask(self, mask):
        """Display segmentation mask"""
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        self.ax.imshow(mask_image)

    def show_points(self, coords, labels):
        """Display input points"""
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        if len(pos_points) > 0:
            self.ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
                           marker='*', s=375, label='Object Points')
        if len(neg_points) > 0:
            self.ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
                           marker='*', s=375, label='Background Points')

    def count_and_draw_bounding_boxes(self, mask):
        """Count objects and draw bounding boxes based on size filters"""
        try:
            min_size = self.colony_min_spin.value()
            max_size = self.colony_max_spin.value()
            object_count = 0

            if np.sum(mask) > 0:
                # Find contours to calculate area
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_size <= area <= max_size:
                        x, y, w, h = cv2.boundingRect(cnt)
                        self.ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=2,
                                                       edgecolor='yellow', facecolor='none'))
                        self.ax.text(x, y, str(object_count + 1), color='yellow',
                                    fontsize=12, fontweight='bold')
                        object_count += 1
            return object_count
        except Exception as e:
            raise Exception(f"Error in bounding box creation: {str(e)}")

    def predict_and_show(self):
        """Make predictions and update display"""
        if not self.input_points:
            return

        try:
            self.set_status("Making prediction...")

            # Start prediction in a separate thread
            self.prediction_thread = PredictionThread(self.predictor, self.input_points, self.input_labels)
            self.prediction_thread.predicted.connect(self.on_prediction_complete)
            self.prediction_thread.error.connect(self.on_prediction_error)
            self.prediction_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction setup failed: {str(e)}")
            self.set_status("Error during prediction")
            self.reset_points()

    def on_prediction_complete(self, mask):
        """Handle prediction complete signal"""
        try:
            # Store current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            # Update display
            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            self.show_mask(mask)
            self.show_points(np.array(self.input_points), np.array(self.input_labels))

            object_count = self.count_and_draw_bounding_boxes(mask)  # 수정: 정확한 객체 수 계산
            self.ax.set_title(f"Detected objects: {object_count}")  # 수정: 올바른 객체 수 표시

            # Restore axis limits and remove ticks
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.canvas.draw()

            self.set_status(f"Prediction complete - {object_count} objects detected")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display prediction: {str(e)}")
            self.set_status("Error during prediction display")

    def on_prediction_error(self, error_message):
        """Handle prediction error signal"""
        QMessageBox.critical(self, "Error", f"Prediction failed: {error_message}")
        self.set_status("Error during prediction")
        self.reset_points()

    def save_image(self):
        """Save current results including image and analysis data"""
        if not hasattr(self, 'processed_image') or not hasattr(self, 'input_points') or not self.input_points:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return

        try:
            # Create timestamp for unique filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create results directory if it doesn't exist
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)

            # Save matplotlib figure
            figure_path = results_dir / f'result_{timestamp}.png'
            self.fig.savefig(str(figure_path), bbox_inches='tight', dpi=300)

            # Save analysis data
            analysis_data = {
                'points': self.input_points,
                'labels': self.input_labels,
                'settings': {
                    'iou_threshold': self.iou_spin.value(),
                    'score_threshold': self.score_spin.value(),
                    'min_colony_size': self.colony_min_spin.value(),
                    'max_colony_size': self.colony_max_spin.value(),
                    'contrast': self.contrast_spin.value(),
                    'grayscale': self.grayscale_cb.isChecked(),
                    'morphology': self.morphology_cb.isChecked(),
                    'sharpen': self.sharpen_cb.isChecked(),
                    'edge_enhance': self.edge_enhance_cb.isChecked()
                }
            }

            analysis_path = results_dir / f'analysis_{timestamp}.json'
            with open(analysis_path, 'w') as f:
                json.dump(analysis_data, f, indent=4)

            self.set_status("Results saved successfully")
            QMessageBox.information(self, "Success", "Results saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
            self.set_status("Error saving results")

    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(self, 'Quit',
                                     'Are you sure you want to quit?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)

    # Set default theme to Dark
    app.setStyle(QStyleFactory.create("Fusion"))
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    gui = InteractiveMaskGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")
