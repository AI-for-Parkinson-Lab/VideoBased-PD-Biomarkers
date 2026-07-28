import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from tqdm import tqdm
import pickle
sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 0, 0)  # vibrant red
import requests



class keypointext:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def download_hand_landmarker(self, model_path):
         url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
         if not os.path.exists(model_path):
             print(f"Downloading Mediapipe Hand Landmarker model to {model_path}...")
             response = requests.get(url)
             response.raise_for_status()
             with open(model_path, "wb") as f:
                 f.write(response.content)
             print("Download complete.")
         else:
             print("Hand Landmarker model already exists.")
     
    def _init_hand_landmarker(self):
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        self.download_hand_landmarker(model_path)
        with open(model_path, "rb") as file:
            model_data = file.read()
        return model_data
       
    def angle_signal(self, landmarks, width, height):
            """
            Calculate the angle between the vector connecting wrist to thumb tip
            and the vector connecting wrist to index finger tip.
            """
            # Get wrist, thumb tip, and index finger tip landmarks
            wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
            thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        
            # Convert normalized coordinates to pixel coordinates
            wrist_x, wrist_y = wrist.x * width, wrist.y * height
            thumb_x, thumb_y = thumb_tip.x * width, thumb_tip.y * height
            index_x, index_y = index_tip.x * width, index_tip.y * height
        
            # Create vectors
            vector_wrist_thumb = np.array([thumb_x - wrist_x, thumb_y - wrist_y])
            vector_wrist_index = np.array([index_x - wrist_x, index_y - wrist_y])
        
            # Calculate dot product and magnitudes of the vectors
            dot_product = np.dot(vector_wrist_thumb, vector_wrist_index)
            magnitude_wrist_thumb = np.linalg.norm(vector_wrist_thumb)
            magnitude_wrist_index = np.linalg.norm(vector_wrist_index)
        
            # Avoid division by zero
            if magnitude_wrist_thumb == 0 or magnitude_wrist_index == 0:
                return 0.0
        
            # Calculate cosine of the angle
            cos_angle = dot_product / (magnitude_wrist_thumb * magnitude_wrist_index)
            # Clamp value to avoid domain error in acos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
            # Calculate angle in degrees
            angle = np.degrees(np.arccos(cos_angle))
        
            return angle
    

 
   

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
    
    def draw_landmarks_on_image(self, rgb_image, detection_result, hand_to_track, label):
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
              if self.config['distance']:
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
              else:
                  # Keypoints used in angle-based normalization
                  if i in [mp.solutions.hands.HandLandmark.WRIST, mp.solutions.hands.HandLandmark.THUMB_CMC]:
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
          text_y = int(min(y_coordinates) * height) - MARGIN

          # Draw handedness (left or right hand) and the label on the image.
          cv2.putText(annotated_image, f"Label: {label}",
                      (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                      FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

      return annotated_image

    def trim_irrelevant_actions(self, sequence):
        peaks, _ = find_peaks(sequence, height=np.mean(sequence), prominence=np.mean(sequence))
        
        if len(peaks) > 10:
            start = peaks[0]
            end = peaks[-1]
            trimmed_sequence = sequence[start:end+1]
            removed_start = sequence[:start]
            removed_end = sequence[end+1:]
        else:
            trimmed_sequence = sequence
            removed_start = []
            removed_end = []
  
        return trimmed_sequence, removed_start, removed_end, peaks

    def plot_sequences(self, original_sequence, trimmed_sequence, removed_start, removed_end, peaks, label):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(original_sequence, 'r-', label='Original')
        plt.plot(peaks, np.array(original_sequence)[peaks], 'ko', label='Peaks')
        if len(removed_start) > 0:
            plt.plot(range(len(removed_start)), removed_start, 'b-', label='Removed Start')
        if len(removed_end) > 0:
            plt.plot(range(len(original_sequence) - len(removed_end), len(original_sequence)), removed_end, 'b-', label='Removed End')
        plt.title(f"Original Sequence (Score: {label})")
        plt.xlabel("Frame Index")
        plt.ylabel("Sequence")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(trimmed_sequence, 'g-', label='Trimmed')
        plt.title("Trimmed Sequence")
        plt.xlabel("Frame Index")
        plt.ylabel("Sequence")
        plt.legend()
        plt.grid(True)
        
        plt.show()

    def preprocess_and_display_video(self, video_path, label):
        """Process video and display frames with calculated distances or angles."""
        hand_to_track = 'Right' if '2R' in video_path else 'Left' if '2L' in video_path else None

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)


        normalized_path = os.path.normpath(video_path)
        base_name = os.path.basename(normalized_path)
        subfolder_name = os.path.basename(os.path.dirname(normalized_path))
        self.new_filename = f"{subfolder_name}_{base_name}"
        save_vid_path = os.path.join(self.config['save_path'], self.new_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #output = cv2.VideoWriter(save_vid_path, fourcc, self.fps, (width, height))




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

                            #if self._validate_hand_with_pose(frame, detection_result.hand_landmarks[idx],  handedness[0].category_name):
    
                                filtered_landmarks.append(detection_result.hand_landmarks[idx])
                            #else:
                               #print(f"Pose validation failed for {hand_to_track} hand. Skipping frame.")
                               #self.wrong_hands.append(video_path)



                detection_result.hand_landmarks = filtered_landmarks

            if detection_result.hand_landmarks:
                detected_frames += 1
                for hand_landmarks in detection_result.hand_landmarks:
                    if self.config['distance']:
                        normalized_keypoints = self._normalize_keypoints_distance(hand_landmarks)
                        index_finger_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = normalized_keypoints[mp.solutions.hands.HandLandmark.THUMB_TIP]
                        sequence_value = np.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 +
                                                 (thumb_tip[1] - index_finger_tip[1]) ** 2 +
                                                 (thumb_tip[2] - index_finger_tip[2]) ** 2)
                    else:
                        sequence_value = self.angle_signal(hand_landmarks, width, height)

                    sequences.append(sequence_value)

                    # Annotate the frame with the label only (no sequence value)
                    annotated_frame = self.draw_landmarks_on_image(frame, detection_result, hand_to_track, label)
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    #cv2.imshow('Annotated Frame', annotated_frame_bgr)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
                    #output.write(annotated_frame)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    frame_keypoints = []
                    for landmark in hand_landmarks:
                        frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # Save original x, y, z coordinates
                    all_keypoints.append(frame_keypoints)  # Shape (21, 3) per frame



        if detected_frames / total_frames >= 0.5 :
            
            if self.config['trimmed']==False:
                sequences = np.array(sequences)
                
                #plt.plot(sequences)
                return np.array(sequences), all_keypoints
            elif self.config['trimmed']==True:
                trimmed_sequences, removed_start, removed_end, peaks = self.trim_irrelevant_actions(sequences)
                #self.plot_sequences(sequences, trimmed_sequences, removed_start, removed_end, peaks, label)
                return np.array(trimmed_sequences), all_keypoints

        else:
            print(f"Skipping video {video_path} as keypoints were not detected in more than 50% of the frames.")
            return None

        cap.release()
        hand_landmarker.close()
        #cv2.destroyAllWindows()
        #output.release()

                                    
              
    def run_keypoints(self):

         pickle_file_path = os.path.join(self.config['save_path'], 'video_keypoints.pkl')
         video_labels = pd.read_csv(self.config['vid2score'])
         videos = video_labels['video_path']
         labels = video_labels['score']
         ids =  video_labels['id']

         # Initialize data containers
         all_video_paths = []
         all_distances = []
         all_keypoints = []
         all_labels = []
         all_ids = []
         all_fps = []


            
         for vid, idi, label in tqdm(zip(videos, ids, labels), total=len(videos), desc="Processing videos"):

            self.vid = vid
            distance, p_keypoint = self.preprocess_and_display_video(vid, label)
  
            all_video_paths.append(vid)
            all_distances.append(distance)
            all_keypoints.append(p_keypoint)
            all_labels.append(label)
            all_ids.append(idi)
            all_fps.append(self.fps)

         # Save all accumulated data to pickle once at the end
         final_data = {
             'video_path': all_video_paths,
             'distances': all_distances,
             'keypoints': all_keypoints,
             'label':all_labels, 
             'id':all_ids,
             'fps':all_fps
         }
         with open(pickle_file_path, 'wb') as f:
             pickle.dump(final_data, f)

if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))

    CONFIG = {
        "vid2score": os.path.join(project_root, "data/raw/segmented_ft_vid2score.csv"),
        "save_path": os.path.join(project_root, "data/raw"),
        "distance": False,  # set True for distance-based, False for angle-based
        "trimmed": False,
    }

    os.makedirs(CONFIG["save_path"], exist_ok=True)
    trainer = keypointext(CONFIG)
    trainer.run_keypoints()

    
    
    
    
    
    
    
    
    