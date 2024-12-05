import os
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the YOLO detector with configuration"""
        self.config = self._load_config(config_path)
        self.device = self.config['model']['device']
        self.model = self._load_model()
        self.conf_threshold = self.config['model']['confidence_threshold']
        self.input_size = self.config['model']['input_size']
        self.selected_classes = None
        self.update_classes()

    def _load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def _load_model(self):
        """Load YOLO model"""
        try:
            weights_path = os.path.join(self.config['paths']['weights'], 'yolo11x.pt')
            if os.path.exists(weights_path):
                model = YOLO(weights_path)
            else:
                print("Downloading YOLO model...")
                model = YOLO('yolo11x')
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                model.save(weights_path)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def update_classes(self):
        """Update the list of classes to detect"""
        self.classes = []
        for class_name, class_ids in self.config['classes'].items():
            self.classes.extend(class_ids)
        self.classes = list(set(self.classes))  # Remove duplicates

    def set_selected_classes(self, classes):
        """Set specific classes to detect"""
        self.selected_classes = set(classes) if classes else None

    def detect(self, frame):
        """Perform detection on a frame"""
        if frame is None:
            return [], None

        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )[0]

            # Process results
            detections = self._process_results(results)

            # Draw detections on frame copy
            processed_frame = self._draw_detections(frame.copy(), detections)

            return detections, processed_frame

        except Exception as e:
            print(f"Error during detection: {e}")
            return [], frame

    def _process_results(self, results):
        """Process YOLO results into a standard format"""
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                try:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = float(box.conf)
                    class_id = int(box.cls)

                    # Skip if class is not selected
                    if self.selected_classes is not None and \
                            self._get_class_type(class_id) not in self.selected_classes:
                        continue

                    # Skip if class_id not in configured classes
                    if class_id not in self.classes:
                        continue

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id),
                        'class_type': self._get_class_type(class_id)
                    })
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
        return detections

    def _get_class_type(self, class_id):
        """Map class ID to Vietnamese type name"""
        types = {
            1: "Xe dap",  # Bicycle
            2: "o to",  # Car
            3: "Xe may",  # Motorcycle
            5: "Xe buyt",  # Bus
            7: "Xe tai",  # Truck
            9: "den giao thong",  # Traffic Light
            11: "bien bao"  # Stop Sign
        }
        return types.get(class_id, "Không xác định")

    def _get_class_name(self, class_id):
        """Map class ID to configuration class name"""
        for category, ids in self.config['classes'].items():
            if class_id in ids:
                return category
        return 'unknown'

    def _draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        vis_config = self.config['visualization']
        stats = {}

        for det in detections:
            try:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                class_type = det['class_type']

                # Get color for class
                color = tuple(vis_config['colors'].get(class_name, [255, 255, 255]))
                stats[class_type] = stats.get(class_type, 0) + 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                              vis_config['box_thickness'])

                # Create and draw label
                label = f"{class_type}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    vis_config['font_scale'],
                    vis_config['text_thickness']
                )

                # Draw label background
                cv2.rectangle(frame,
                              (x1, y1 - label_h - 10),
                              (x1 + label_w, y1),
                              color, -1)

                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            vis_config['font_scale'],
                            (255, 255, 255),
                            vis_config['text_thickness'])

            except Exception as e:
                print(f"Error drawing detection: {e}")
                continue

        # Draw statistics if enabled
        if self.config['stats']['show_count']:
            self._draw_stats(frame, stats)

        return frame

    def _draw_stats(self, frame, stats):
        """Draw detection statistics on frame"""
        if not stats:
            return frame

        y_pos = 30
        padding = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Draw background for stats
        max_width = 0
        total_height = 0

        # Calculate dimensions
        for obj_type, count in stats.items():
            text = f"{obj_type}: {count}"
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, w)
            total_height += h + padding

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5),
                      (max_width + 15, total_height + 15),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_pos = 25
        for obj_type, count in stats.items():
            text = f"{obj_type}: {count}"
            cv2.putText(frame, text, (10, y_pos),
                        font, font_scale,
                        (255, 255, 255), thickness)
            y_pos += 25

        return frame

    def set_confidence(self, confidence):
        """Set detection confidence threshold"""
        self.conf_threshold = confidence

    def __call__(self, frame):
        """Make the class callable"""
        return self.detect(frame)