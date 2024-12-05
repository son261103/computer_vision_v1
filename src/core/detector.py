import cv2
import numpy as np
import yaml
import time
import os
from pathlib import Path
from src.models.yolo_detector import YOLOv8Detector
from src.utils.video_stream import VideoStream
from src.utils.visualization import Visualizer


class Detector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the detector with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.model = YOLOv8Detector(config_path)
        self.video_stream = VideoStream(config_path)
        self.visualizer = Visualizer(config_path)
        self.reset_stats()
        self.selected_classes = set()  # Store selected classes for filtering

    def _load_config(self, config_path):
        """Load configuration file with proper encoding"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config with UTF-8: {e}")
            with open(config_path, 'r', encoding='latin-1') as f:
                return yaml.safe_load(f)

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.config['paths']['input'],
            self.config['paths']['output'],
            self.config['paths']['weights']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def set_selected_classes(self, classes):
        """Set which classes should be detected"""
        self.selected_classes = set(classes)
        if hasattr(self, 'model'):
            self.model.set_selected_classes(classes)

    def process_video(self, video_path: str):
        """Process a video file and detect objects"""
        try:
            if not self.video_stream.start_stream(video_path):
                raise ValueError(f"Could not open video: {video_path}")

            start_time = time.time()
            self.reset_stats()
            frame_count = 0
            total_frames = self.video_stream.total_frames

            while True:
                frame = self.video_stream.read_frame()
                if frame is None:
                    break

                frame_count += 1
                detections, processed_frame = self.model.detect(frame)
                self.update_stats(detections)

                # Calculate progress and timing
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Add visualizations
                processed_frame = self.process_frame_visualization(
                    processed_frame,
                    detections,
                    {
                        'progress': progress,
                        'fps': current_fps,
                        'frame_count': frame_count,
                        'total_frames': total_frames,
                        'elapsed_time': elapsed_time
                    }
                )

                # Write and display frame
                self.video_stream.write_frame(processed_frame)
                cv2.imshow('Traffic Detection System', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error processing video: {str(e)}")
        finally:
            self.video_stream.release()
            cv2.destroyAllWindows()

    def process_frame_visualization(self, frame, detections, stats):
        """Add visualizations to the processed frame"""
        # Draw detections
        frame = self.visualizer.draw_detections(frame, detections)

        # Draw FPS counter
        if self.config['video']['draw_fps']:
            frame = self.visualizer.draw_fps(frame, stats['fps'])

        # Draw progress bar
        frame = self.visualizer.draw_progress(
            frame,
            stats['progress'],
            stats['frame_count'],
            stats['total_frames']
        )

        # Create and add info panel
        info = self.create_info_dict(stats)
        frame = self.visualizer.create_info_panel(frame, info)

        return frame

    def create_info_dict(self, stats):
        """Create dictionary with current detection information"""
        info = {
            'Total Vehicles': self.stats['total_detections'],
            'Frame': f"{stats['frame_count']}/{stats['total_frames']}",
            'FPS': f"{stats['fps']:.1f}",
            'Time': f"{stats['elapsed_time']:.1f}s"
        }

        # Add counts for each vehicle type
        for class_type, count in self.class_stats.items():
            if count > 0:  # Only show classes that have been detected
                info[class_type] = count

        return info

    def update_stats(self, detections):
        """Update detection statistics"""
        filtered_detections = [
            det for det in detections
            if not self.selected_classes or det['class_type'] in self.selected_classes
        ]

        self.stats['total_detections'] += len(filtered_detections)

        for det in filtered_detections:
            class_type = det['class_type']
            self.class_stats[class_type] = self.class_stats.get(class_type, 0) + 1

    def reset_stats(self):
        """Reset all statistics counters"""
        self.stats = {'total_detections': 0}
        self.class_stats = {
            'Xe may': 0,
            'Xe dap': 0,
            'o to': 0,
            'Xe buyt': 0,
            'Xe tai': 0,
            'den giao thong': 0,
            'bien bao': 0
        }

    def process_frame(self, frame):
        """Process a single frame for detection"""
        if frame is None:
            return None, []

        try:
            detections, processed_frame = self.model.detect(frame)

            # Filter detections based on selected classes
            if self.selected_classes:
                detections = [
                    det for det in detections
                    if det['class_type'] in self.selected_classes
                ]

            return processed_frame, detections
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame, []

    def get_stats(self):
        """Get current detection statistics"""
        return {
            'total': self.stats['total_detections'],
            'by_class': self.class_stats
        }

    def get_video_info(self):
        """Get information about the current video"""
        return self.video_stream.get_video_info()

    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for detections"""
        if hasattr(self, 'model'):
            self.model.conf_threshold = threshold