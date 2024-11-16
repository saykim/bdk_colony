# Interactive SAM Control Panel

## Overview

**Interactive SAM Control Panel** is a Python-based desktop application built with **PyQt5** that leverages the **Segment Anything Model (SAM)** to perform interactive image segmentation. Users can load images, click on colony-like structures within the image to generate segmentation masks, and accurately count the detected objects. The application is optimized for performance and is fully compatible with Windows PCs.

## Features

- **Interactive Segmentation:** Click on objects within an image to generate precise segmentation masks using SAM.
- **Accurate Object Counting:** Automatically detects and counts individual objects based on user input.
- **Real-time Image Processing:** Apply various image processing techniques such as grayscale conversion, contrast adjustment, morphological operations, sharpening, and edge enhancement.
- **User-Friendly Interface:** Intuitive GUI built with PyQt5, featuring easy-to-use controls and real-time feedback.
- **Multi-threading:** Ensures smooth and responsive user experience by handling heavy computations in separate threads.
- **Theme Support:** Switch between Dark and Light themes to suit user preferences.
- **Result Saving:** Save segmented images and analysis data for future reference.
- **Drag and Drop Support:** Easily load images by dragging and dropping them into the application window.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/interactive-sam-control-panel.git
cd interactive-sam-control-panel
