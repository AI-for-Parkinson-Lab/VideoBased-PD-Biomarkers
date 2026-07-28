
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import spearmanr
sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import urllib.request
import importlib.util

class ft_video_analysis:
    def __init__(self, config):
        self.config = config
        
    def _init_hand_landmarker(self):
            
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(CURRENT_DIR, 'hand_landmarker.task')
    
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            print("hand_landmarker.task not found. Downloading...")
    
            # Mediapipe official model URL (Google-hosted)
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ Downloaded hand_landmarker.task to: {model_path}")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to download model: {e}")
    
        #
        with open(model_path, 'rb') as file:
            model_data = file.read()
    
        return model_data
    def _normalize_keypoints_distance(self, hand_landmarks):
            wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

            wrist_to_index_mcp_length = np.sqrt((index_mcp.x - wrist.x) ** 2 +
                                                (index_mcp.y - wrist.y) ** 2 +
                                                (index_mcp.z - wrist.z) ** 2)

            if wrist_to_index_mcp_length < 1e-5:
                wrist_to_index_mcp_length = 1e-5

            normalized_keypoints = []
            for landmark in hand_landmarks:
                centered_x = landmark.x - wrist.x
                centered_y = landmark.y - wrist.y
                centered_z = landmark.z - wrist.z

                normalized_x = centered_x / wrist_to_index_mcp_length
                normalized_y = centered_y / wrist_to_index_mcp_length
                normalized_z = centered_z / wrist_to_index_mcp_length

                normalized_keypoints.append((normalized_x, normalized_y, normalized_z))
            return normalized_keypoints  
    def _extract_features(self, distances):
         current_dir = os.path.dirname(os.path.abspath(__file__))
         feature_extraction_path = os.path.join(
             current_dir, "..", "feature extraction", "feature_extraction.py"
         )

         spec = importlib.util.spec_from_file_location(
             "feature_extraction_module", feature_extraction_path
         )
         feature_module = importlib.util.module_from_spec(spec)
         spec.loader.exec_module(feature_module)

         extractor = feature_module.feature_ext_analysis({"test_type": "ft"})
         feature_values, feature_names = extractor._extract_features(distances, self.fps)
         feature_columns = feature_names[6:]

         df = pd.DataFrame([feature_values], columns=feature_columns)
         df.to_csv(os.path.join(self.SAVE_DIR, 'features.csv'), index=False)

    def draw_landmarks_on_image(self, rgb_image, detection_result, hand_to_track):
      hand_landmarks_list = detection_result.hand_landmarks
      handedness_list = detection_result.handedness
      annotated_image = np.copy(rgb_image)

      # Define drawing styles
      default_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
      red_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 0, 255))
      blue_drawing_spec = solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(255, 0, 0))  # Blue for keypoints in normalization
      
      # Loop through the detected hands to visualize.
      for idx in range(len(hand_landmarks_list)):
          hand_landmarks = hand_landmarks_list[idx]
          handedness = handedness_list[idx]

          # Draw the key landmarks with distinct colors depending on normalization type
          for i, landmark in enumerate(hand_landmarks):
              landmark_proto = landmark_pb2.NormalizedLandmark(
                  x=landmark.x, y=landmark.y, z=landmark.z
              )
              # Keypoints used in distance-based normalization
              if i in [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]:
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                    None,
                    landmark_drawing_spec=blue_drawing_spec
                )
              elif i in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.THUMB_TIP]:
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                    None,
                    landmark_drawing_spec=red_drawing_spec
                )
              else:
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    landmark_pb2.NormalizedLandmarkList(landmark=[landmark_proto]),
                    None,
                    landmark_drawing_spec=default_drawing_spec
                )


          # Draw the connections
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])
          solutions.drawing_utils.draw_landmarks(
              annotated_image,
              hand_landmarks_proto,
              solutions.hands.HAND_CONNECTIONS,
              default_drawing_spec,
              default_drawing_spec)

          # Get the top left corner of the detected hand's bounding box.
          height, width, _ = annotated_image.shape
          x_coordinates = [landmark.x for landmark in hand_landmarks]
          y_coordinates = [landmark.y for landmark in hand_landmarks]
          text_x = int(min(x_coordinates) * width)
          text_y = int(min(y_coordinates) * height) - 10


      return annotated_image

    def preprocess_and_display_video(self):
        """Process video and display frames with calculated distances or angles."""
        hand_to_track = self.config['hand2track']

        cap = cv2.VideoCapture(self.config['video_path'])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)


        normalized_path = os.path.normpath(self.config['video_path'])
        base_name = os.path.basename(normalized_path)
        subfolder_name = os.path.basename(os.path.dirname(normalized_path))
        self.new_filename = f"{subfolder_name}_{base_name}"
        
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.SAVE_DIR = os.path.join(CURRENT_DIR, "results")       
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        
        save_vid_path = os.path.join(self.SAVE_DIR, self.new_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(save_vid_path, fourcc, self.fps, (width, height))

        sequences = []
        total_frames = 0
        detected_frames = 0
        self.palm_length = []
        model_data = self._init_hand_landmarker()
        all_keypoints = []
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            num_hands=2,
            running_mode=VisionRunningMode.VIDEO
        )
        hand_landmarker = HandLandmarker.create_from_options(options)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Filter hands based on the handedness
            if detection_result.handedness:
                filtered_landmarks = []
                for idx, handedness in enumerate(detection_result.handedness):
                    if handedness[0].category_name == hand_to_track:
    
                                filtered_landmarks.append(detection_result.hand_landmarks[idx])




                detection_result.hand_landmarks = filtered_landmarks

            if detection_result.hand_landmarks:
                detected_frames += 1
                for hand_landmarks in detection_result.hand_landmarks:
                    normalized_keypoints = self._normalize_keypoints_distance(hand_landmarks)
                    index_finger_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    sequence_value = np.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 +
                                             (thumb_tip[1] - index_finger_tip[1]) ** 2 +
                                             (thumb_tip[2] - index_finger_tip[2]) ** 2)


                    sequences.append(sequence_value)

                    # Annotate the frame with the label only (no sequence value)
                    annotated_frame = self.draw_landmarks_on_image(frame, detection_result, hand_to_track)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Annotated Frame', annotated_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    output.write(annotated_frame)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    frame_keypoints = []
                    for landmark in hand_landmarks:
                        frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # Save original x, y, z coordinates
                    all_keypoints.append(frame_keypoints)  # Shape (21, 3) per frame

        cap.release()
        hand_landmarker.close()
        cv2.destroyAllWindows()
        output.release()
        self._extract_features(sequences)

        
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Finger Tapping Video Analysis")

    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--hand2track", type=str, choices=["Left", "Right"], default="Left", help="Hand to track")

    args = parser.parse_args()

    CONFIG = {
        'video_path': args.video_path,
        'hand2track': args.hand2track,
    }

    analyzer = ft_video_analysis(CONFIG)
    analyzer.preprocess_and_display_video()


