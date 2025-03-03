import cv2
import yaml
import time
import numpy as np
import os
from pathlib import Path
from datetime import datetime


class VideoStream:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize video stream with configuration"""
        self.config = self._load_config(config_path)
        self.resize_width = self.config['video']['resize_width']
        self.resize_height = self.config['video']['resize_height']
        self.fps = self.config['video']['fps']
        self.save_output = self.config['video']['save_output']
        self.auto_save = self.config['video'].get('auto_save', True)

        # Video handling attributes
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.total_frames = 0
        self.start_time = None
        self.current_fps = 0
        self.paused = False
        self.original_fps = 0

        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 30  # Keep track of last 30 frames

    def _load_config(self, config_path):
        """Load configuration file with error handling"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def start_stream(self, video_path: str):
        """Start video stream from file"""
        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            self._setup_video_info()

            if self.save_output:
                output_path = self._get_output_path(video_path)
                self.writer = self._initialize_writer(output_path)

            self.start_time = time.time()
            self.frame_count = 0
            self.paused = False
            return True

        except Exception as e:
            print(f"Error starting video stream: {e}")
            return False

    def _setup_video_info(self):
        """Setup video information and parameters"""
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read_frame(self):
        """Read and process a frame from video"""
        if self.cap is None or self.paused:
            return None

        frame_time = time.time()
        ret, frame = self.cap.read()

        if not ret:
            return None

        try:
            # Resize frame
            frame = cv2.resize(frame, (self.resize_width, self.resize_height),
                               interpolation=cv2.INTER_AREA)

            self.frame_count += 1

            # Update FPS calculation
            self._update_fps(frame_time)

            return frame

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def _update_fps(self, frame_time):
        """Update FPS calculations"""
        processing_time = time.time() - frame_time
        self.processing_times.append(processing_time)

        # Keep only recent processing times
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)

        # Calculate current FPS
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            self.current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

    def write_frame(self, frame):
        """Write processed frame to output file"""
        if self.save_output and self.writer is not None and self.auto_save:
            try:
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    self.writer.write(frame)
            except Exception as e:
                print(f"Error writing frame: {e}")

    def get_progress(self):
        """Get current progress percentage"""
        if self.total_frames > 0:
            return (self.frame_count / self.total_frames) * 100
        return 0

    def get_fps(self):
        """Get current FPS"""
        return self.current_fps

    def get_estimated_time(self):
        """Get estimated time remaining"""
        if self.current_fps <= 0 or self.total_frames <= 0:
            return "Unknown"

        frames_remaining = self.total_frames - self.frame_count
        seconds_remaining = frames_remaining / self.current_fps

        minutes = int(seconds_remaining // 60)
        seconds = int(seconds_remaining % 60)

        return f"{minutes:02d}:{seconds:02d}"

    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        if not self.paused:
            self.start_time = time.time() - (self.frame_count / self.fps)
        return self.paused

    def release(self):
        """Release all resources"""
        try:
            if self.cap is not None:
                self.cap.release()
            if self.writer is not None:
                self.writer.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error releasing resources: {e}")

    def _get_output_path(self, input_path):
        """Generate output file path"""
        try:
            input_name = Path(input_path).stem
            output_dir = Path(self.config['paths']['output'])
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_format = self.config['video'].get('save_format', 'mp4')

            # Add detection info to filename
            output_name = f"{input_name}_detected_{timestamp}.{save_format}"
            return str(output_dir / output_name)

        except Exception as e:
            print(f"Error generating output path: {e}")
            return str(output_dir / f"output_{timestamp}.{save_format}")

    def _initialize_writer(self, output_path):
        """Initialize video writer"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            return cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.resize_width, self.resize_height)
            )
        except Exception as e:
            print(f"Error initializing writer: {e}")
            return None

    def get_video_info(self):
        """Get comprehensive video information"""
        return {
            'original_width': self.original_width,
            'original_height': self.original_height,
            'resize_width': self.resize_width,
            'resize_height': self.resize_height,
            'total_frames': self.total_frames,
            'current_frame': self.frame_count,
            'original_fps': self.original_fps,
            'current_fps': self.current_fps,
            'progress': self.get_progress(),
            'estimated_time': self.get_estimated_time(),
            'is_paused': self.paused
        }

    def is_opened(self):
        """Check if video is opened"""
        return self.cap is not None and self.cap.isOpened()

    def seek_to_frame(self, frame_number):
        """Seek to specific frame"""
        if self.cap is not None and frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_count = frame_number

    def __del__(self):
        """Cleanup on deletion"""
        self.release()