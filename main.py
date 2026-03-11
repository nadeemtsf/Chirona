"""
Main entry point for the Chirona Sign Language application.

Initializes the hardware, machine learning models, and feature extractors
to translate real-time hand gestures into actionable sign language translation.
"""

import pickle
import cv2
import time
import sys
import logging
import numpy as np
import tkinter as tk
from tkinter import ttk

from core.config_manager import config_mgr
from core.sign_classifier import SignClassifier
from core.hand_detector import HandDetector
from utils.text_overlay import draw_prediction, draw_sentence_builder_ui
from core.feature_extractor import FeatureExtractor
from core.sentence_builder import SentenceBuilder
from utils.prediction_smoother import PredictionSmoother

from config import (
    CAMERA_INDEX, CAM_WIDTH, CAM_HEIGHT,
    COLOR_PRIMARY, WINDOW_TITLE, CONFIDENCE_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChironaApp:
    def __init__(self):
        self._setup()
        
    def _setup(self):
        """Initialize models and hardware."""
        # Initialize hand detector (start in single hand mode)
        self.detector = HandDetector(max_hands=1)
        self.fe = FeatureExtractor(use_z=False)  # Must match training config
        
        # Load trained sign language model
        try: 
            self.classifier = SignClassifier('models/trained_model.pkl')
            logging.warning('sign classifier loaded successfully')
        except FileNotFoundError:
            logging.warning("Trained model not found. Sign language recognition will not work until you run 'train_model.py' to create the model file.")
            self.classifier = None
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.classifier = None
            
        # Initialize webcam
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        if not self.cap.isOpened():
            print("Failed to open camera")
            sys.exit(1)
            
        # Runtime state variables
        self.prev_time = 0
        self.max_hands_mode = 1 # start with single hand mode
        
        self.smoother = PredictionSmoother()
        self.sentence_builder = SentenceBuilder()
        self.displayed_sign = None
        self.displayed_confidence = None
        self.frame_count = 0

    def _process_prediction(self, hand):
        """Extract features, predict gesture, and smooth the output for the UI."""
        landmarks = hand['landmarks']
        
        # Only predict every 3rd frame to improve FPS
        if self.frame_count % 3 == 0:
            # Extract and normalize features
            features = self.fe.extract(landmarks)
            normalized_features = self.fe.normalize(features)

            # Predict gesture
            if self.classifier is not None:
                label, confidence = self.classifier.predict(normalized_features)
                
                # Only process predictions above a low confidence baseline, to avoid noise
                if confidence > 0.0:
                    self.smoother.add_prediction(label)
                
                stable = self.smoother.get_stable_prediction()

                # Update displayed sign if stable prediction is available
                if stable is not None:
                    # In a real scenario we'd track the confidence of the stable reading over the window,
                    # but for now we just show the most recent raw confidence of the frame
                    self.displayed_sign = stable
                    self.displayed_confidence = confidence

    def _handle_keypress(self):
        """Handle keyboard input. Returns False if app should exit."""
        key = cv2.waitKey(1) & 0xFF
        
        # Manually add space with spacebar
        if key == ord(' '):
            self.sentence_builder.add_space()
            
        if key == ord('h'):
            #toggle between 1 and 2 hand modes
            self.max_hands_mode = 2 if self.max_hands_mode == 1 else 1
            self.detector.hands.max_num_hands = self.max_hands_mode
            print(f'Hand dectection mode: {self.max_hands_mode} hand(s)')
            
        if key == ord('s'):
            # Toggle settings visibility
            if self.is_hidden:
                self.root.deiconify()
                self.is_hidden = False
            else:
                self.root.withdraw()
                self.is_hidden = True
            
        # Exit on Escape key or window close
        if key == 27 or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            return False
            
        return True

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Sign Language Translator", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Helpers
        def create_slider(label, key, from_, to, resolution=1.0):
            frame = ttk.Frame(main_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=label).pack(side=tk.LEFT)
            val_label = ttk.Label(frame, text=f"{config_mgr.get(key):.2f}")
            val_label.pack(side=tk.RIGHT)
            
            var = tk.DoubleVar(value=config_mgr.get(key))
            
            def on_change(*args):
                v = var.get()
                if resolution >= 1.0: v = int(v)
                val_label.config(text=f"{v:.2f}")
                config_mgr.set(key, v)
                
            slider = ttk.Scale(frame, from_=from_, to=to, variable=var, command=on_change)
            slider.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
            return var
            
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(main_frame, text="MediaPipe Settings (Requires Apply)").pack()
        
        self.det_conf_var = create_slider("Detection Conf", "DETECTION_CONFIDENCE", 0.1, 1.0, 0.01)
        self.trk_conf_var = create_slider("Tracking Conf", "TRACKING_CONFIDENCE", 0.1, 1.0, 0.01)
        
        def apply_mp():
            self.detector.update_settings(
                self.det_conf_var.get(),
                self.trk_conf_var.get()
            )
            config_mgr.set('DETECTION_CONFIDENCE', self.det_conf_var.get())
            config_mgr.set('TRACKING_CONFIDENCE', self.trk_conf_var.get())
        ttk.Button(main_frame, text="Apply MediaPipe Changes", command=apply_mp).pack(pady=10)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        def save_state():
            config_mgr.save_config()
        ttk.Button(main_frame, text="Save Configuration Defaults", command=save_state).pack(pady=5)
        
        ttk.Label(main_frame, text="Press 's' in video feed to toggle this module", foreground="gray").pack(side=tk.BOTTOM, pady=5)

    def update_frame(self):
        """Timer-driven frame processor fired by Tkinter."""
        success, frame = self.cap.read()
        if not success:
            print("Failed to read frame")
            self.cleanup_and_exit()
            return

        frame = cv2.flip(frame, 1)
        hands_data = self.detector.detect(frame)
        frame = self.detector.draw_hands(frame, hands_data)
        
        self.frame_count += 1

        # Process first detected hand
        if hands_data:
            first_hand = hands_data[0]
            self._process_prediction(first_hand)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time

        # Update sentence builder
        if hands_data:
            self.sentence_builder.update(self.displayed_sign, current_time)
        else:
            self.sentence_builder.update(None, current_time)

        # Display info text overlays
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
        cv2.putText(frame, f'Hands: {self.max_hands_mode}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
        cv2.putText(frame, 'Mode: Sign Language', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 2)
        
        if self.classifier is None:
            cv2.putText(frame, f'Min confidence: {CONFIDENCE_THRESHOLD:.0%}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PRIMARY, 1)

        # Display predicted sign bar if available
        if self.displayed_sign and self.displayed_confidence:
            draw_prediction(frame, self.displayed_sign, self.displayed_confidence)

        # Display sentence builder UI
        draw_sentence_builder_ui(frame, self.sentence_builder, current_time)

        cv2.imshow(WINDOW_TITLE, frame)

        # Break loop if _handle_keypress asks to exit
        if not self._handle_keypress():
            self.cleanup_and_exit()
            return
            
        # Recursive native loop
        self.root.after(10, self.update_frame)

    def cleanup_and_exit(self):
        self.cleanup()
        self.root.quit()

    def run(self):
        """Main application runtime loop."""
        self.root = tk.Tk()
        self.root.title("Chirona Settings")
        self.root.geometry("380x350")
        
        self.is_hidden = False
        self.setup_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)
        self.root.after(10, self.update_frame)
        self.root.mainloop()

        self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ChironaApp()
    app.run()