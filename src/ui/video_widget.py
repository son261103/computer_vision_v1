from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import cv2
import numpy as np
from datetime import datetime


class VideoWidget(QWidget):
    # Custom signals
    screenshot_taken = pyqtSignal(str)  # Emit screenshot path
    double_clicked = pyqtSignal()  # For fullscreen toggle

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_frame = None
        self.zoom_factor = 1.0
        self.pan_position = QPoint(0, 0)
        self.last_mouse_pos = None
        self.is_panning = False
        self.setup_context_menu()

    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for zooming and panning
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #ff69b4;
                border-radius: 5px;
            }
        """)

        # Enable mouse tracking for hover effects
        self.video_label.setMouseTracking(True)

        # Add label to scroll area
        self.scroll_area.setWidget(self.video_label)
        layout.addWidget(self.scroll_area)

        # Style the widget
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
            }
            QScrollBar:horizontal {
                height: 12px;
                background: #2d2d2d;
                border-radius: 6px;
            }
            QScrollBar:vertical {
                width: 12px;
                background: #2d2d2d;
                border-radius: 6px;
            }
            QScrollBar::handle {
                background: #ff69b4;
                border-radius: 5px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
        """)

        # Install event filter for keyboard shortcuts
        self.installEventFilter(self)

    def setup_context_menu(self):
        """Setup right-click context menu"""
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.context_menu = QMenu(self)
        self.context_menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #ff69b4;
            }
            QMenu::item:selected {
                background-color: #ff69b4;
            }
        """)

        # Add menu actions
        self.context_menu.addAction("Take Screenshot", self.take_screenshot)
        self.context_menu.addAction("Reset Zoom", self.reset_zoom)
        self.context_menu.addSeparator()
        self.context_menu.addAction("Copy Frame", self.copy_frame)

    def update_frame(self, frame):
        """Update the displayed frame with support for zoom and pan"""
        if frame is None:
            return

        try:
            self.current_frame = frame.copy()

            # Convert frame to RGB
            if len(frame.shape) == 2:  # Grayscale
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # BGR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get display dimensions
            label_size = self.video_label.size()
            frame_height, frame_width = rgb_frame.shape[:2]

            # Calculate scaling while considering zoom
            base_scale = min(label_size.width() / frame_width,
                             label_size.height() / frame_height)
            scale = base_scale * self.zoom_factor

            # Calculate new dimensions
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Create black background
            background = np.zeros((label_size.height(), label_size.width(), 3),
                                  dtype=np.uint8)

            # Resize frame
            scaled_frame = cv2.resize(rgb_frame, (new_width, new_height),
                                      interpolation=cv2.INTER_AREA)

            # Calculate centering position with pan offset
            y_offset = (label_size.height() - new_height) // 2 + self.pan_position.y()
            x_offset = (label_size.width() - new_width) // 2 + self.pan_position.x()

            # Ensure offsets stay within bounds
            x_offset = max(min(x_offset, label_size.width() - new_width), 0)
            y_offset = max(min(y_offset, label_size.height() - new_height), 0)

            # Place scaled frame on background
            try:
                # Create region of interest
                roi = background[y_offset:y_offset + new_height,
                      x_offset:x_offset + new_width]
                if roi.shape == scaled_frame.shape:
                    background[y_offset:y_offset + new_height,
                    x_offset:x_offset + new_width] = scaled_frame
            except ValueError as e:
                print(f"Error placing frame on background: {e}")
                return

            # Convert to QImage
            h, w, ch = background.shape
            bytes_per_line = ch * w
            qt_image = QImage(background.data, w, h, bytes_per_line,
                              QImage.Format.Format_RGB888)

            # Set pixmap
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            print(f"Error updating frame: {str(e)}")

    def clear_display(self):
        """Clear the display and reset zoom/pan"""
        try:
            self.current_frame = None
            self.zoom_factor = 1.0
            self.pan_position = QPoint(0, 0)

            # Create black image
            size = self.video_label.size()
            black_image = np.zeros((size.height(), size.width(), 3),
                                   dtype=np.uint8)

            # Convert to QImage
            h, w, ch = black_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(black_image.data, w, h, bytes_per_line,
                              QImage.Format.Format_RGB888)

            # Set pixmap
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            print(f"Error clearing display: {str(e)}")

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_frame is not None:
            # Calculate zoom factor
            zoom_in = event.angleDelta().y() > 0
            zoom_factor = 1.1 if zoom_in else 0.9

            # Update zoom within limits
            self.zoom_factor *= zoom_factor
            self.zoom_factor = max(0.5, min(self.zoom_factor, 5.0))

            # Update display
            self.update_frame(self.current_frame)

    def mousePressEvent(self, event):
        """Handle mouse press for panning and context menu"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.MouseButton.RightButton:
            self.show_context_menu(event.pos())

    def mouseDoubleClickEvent(self, event):
        """Handle double click for fullscreen toggle"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for panning"""
        if self.is_panning and self.last_mouse_pos is not None:
            # Calculate movement
            delta = event.pos() - self.last_mouse_pos
            self.pan_position += delta
            self.last_mouse_pos = event.pos()

            # Update display
            if self.current_frame is not None:
                self.update_frame(self.current_frame)

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None

    def show_context_menu(self, pos):
        """Show context menu at position"""
        self.context_menu.exec(self.mapToGlobal(pos))

    def take_screenshot(self):
        """Take a screenshot of current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            self.screenshot_taken.emit(filename)

    def copy_frame(self):
        """Copy current frame to clipboard"""
        if self.current_frame is not None:
            pixmap = self.video_label.pixmap()
            if pixmap:
                QApplication.clipboard().setPixmap(pixmap)

    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_position = QPoint(0, 0)
        if self.current_frame is not None:
            self.update_frame(self.current_frame)

    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
        else:
            self.clear_display()

    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts"""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                # Toggle pause/play
                return True
            elif event.key() == Qt.Key.Key_R:
                # Reset zoom
                self.reset_zoom()
                return True
            elif event.key() == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                # Take screenshot
                self.take_screenshot()
                return True
        return super().eventFilter(obj, event)