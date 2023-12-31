import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json

class StressDetectionApp:
    def __init__(self):
        # Initialize MediaPipe FaceMesh and Hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.landmarks_face = None
        self.landmarks_hand = None

        # Dictionary to store stress features
        self.stress_data = {"blink": [], "eyebrows": [], "lip_movement": [], "emotions": [], "hand_movement": [], "aggregated": []}

        # Placeholder for aggregated stress calculation
        self.aggregated_stress = 0

    def calculate_eye_blink(self):
        if self.landmarks_face is not None:
            # Convert landmarks to NumPy array for calculations
            landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in self.landmarks_face])

            left_eye_landmarks = landmarks_np[159:145:-1]  # Left eye landmarks
            right_eye_landmarks = landmarks_np[386:374:-1]  # Right eye landmarks

            left_eye_aspect_ratio = self.eye_aspect_ratio(left_eye_landmarks)
            right_eye_aspect_ratio = self.eye_aspect_ratio(right_eye_landmarks)

            avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

            return avg_eye_aspect_ratio

    @staticmethod
    def eye_aspect_ratio(eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        C = np.linalg.norm(eye[0] - eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        return ear

    def calculate_eyebrows(self):
        if self.landmarks_face is not None:
            # Extract the landmarks corresponding to the eyebrows
            left_eyebrow_landmarks = self.landmarks_face[55:65]
            right_eyebrow_landmarks = self.landmarks_face[285:295]

            avg_left_eyebrow_height = np.mean([landmark.y for landmark in left_eyebrow_landmarks])
            avg_right_eyebrow_height = np.mean([landmark.y for landmark in right_eyebrow_landmarks])

            eyebrow_asymmetry = np.abs(avg_left_eyebrow_height - avg_right_eyebrow_height)

            return eyebrow_asymmetry

    def calculate_lip_movement(self):
        if self.landmarks_face is not None:
            upper_lip_landmarks = self.landmarks_face[61:65] + [self.landmarks_face[146]]
            lower_lip_landmarks = self.landmarks_face[65:68] + [self.landmarks_face[178]]

            lip_distance = np.mean([landmark.y for landmark in lower_lip_landmarks]) - np.mean(
                [landmark.y for landmark in upper_lip_landmarks])

            return lip_distance

    def calculate_emotions(self):
        return 0  # Placeholder for emotion calculation

    def calculate_hand_movement(self):
        if self.landmarks_hand is not None:
            # Extract the landmarks corresponding to the hand
            hand_landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in self.landmarks_hand.landmark])

            hand_movement = np.mean(np.diff(hand_landmarks_np[:, 1]))

            return hand_movement
        else:
            return 0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe FaceMesh and Hands
        results_face = self.face_mesh.process(rgb_frame)
        results_hand = self.hands.process(rgb_frame)

        if results_face.multi_face_landmarks:
            self.landmarks_face = results_face.multi_face_landmarks[0].landmark

            # Placeholder values for stress detection (you need to replace these with real values)
            blink = self.calculate_eye_blink()
            eyebrows = self.calculate_eyebrows()
            lip_movement = self.calculate_lip_movement()

            hand_movement = self.calculate_hand_movement()

            emotions = self.calculate_emotions()

            self.aggregated_stress = 0.25 * blink + 0.25 * eyebrows + 0.25 * lip_movement + 0.25 * hand_movement
            self.stress_data["blink"].append(blink)
            self.stress_data["eyebrows"].append(eyebrows)
            self.stress_data["lip_movement"].append(lip_movement)
            self.stress_data["hand_movement"].append(hand_movement)
            self.stress_data["emotions"].append(emotions)
            self.stress_data["aggregated"].append(self.aggregated_stress)

            self.draw_landmarks(frame)

        if results_hand.multi_hand_landmarks:
            self.landmarks_hand = results_hand.multi_hand_landmarks[0]

    def draw_landmarks(self, frame):
        h, w, _ = frame.shape
        landmarks_x = [int(landmark.x * w) for landmark in self.landmarks_face]
        landmarks_y = [int(landmark.y * h) for landmark in self.landmarks_face]

        # Draw landmarks on the frame
        for x, y in zip(landmarks_x, landmarks_y):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    def run(self, video_path):
       
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)

   
            cv2.imshow("Stress Detection", frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

       
        output_data = {"seconds": list(range(seconds)),
                       "blink": self.stress_data["blink"],
                       "eyebrows": self.stress_data["eyebrows"],
                       "lip_movement": self.stress_data["lip_movement"],
                       "emotions": self.stress_data["emotions"],
                       "aggregated": self.stress_data["aggregated"]}
        # Save the output JSON
        output_filename = "stress_output.json"  # Corrected filename
        with open(output_filename, "w") as json_file:
            json.dump(output_data, json_file, indent=2)
        print(f"Stress levels per second and aggregated stress data have been saved to '{output_filename}'.")

       
        self.plot_stress()

    def plot_stress(self):
        seconds = range(len(self.stress_data["aggregated"]))

        # Plot Blink
        plt.figure(figsize=(18, 15))
        plt.subplot(3, 2, 1)
        plt.plot(seconds, self.stress_data["blink"], label="Blink")
        plt.xlabel("Time")
        plt.ylabel("Blink Level")
        plt.title('stress detected from eye blinks')

        plt.legend()

        # Plot Eyebrows
        plt.subplot(3, 2, 2)
        plt.plot(seconds, self.stress_data["eyebrows"], label="Eyebrows")
        plt.xlabel("Time")
        plt.ylabel("Eyebrows Level")        
        plt.title('stress detected from eyebrows')

        plt.legend()

        # Plot Lip Movement
        plt.subplot(3, 2, 3)
        plt.plot(seconds, self.stress_data["lip_movement"], label="Lip Movement")
        plt.title('stress detected from Lips')
        plt.xlabel("Time")
        plt.ylabel("Lip Movement Level")
        plt.legend()

        # Plot Hand Movement
        plt.subplot(3, 2, 4)
        plt.plot(seconds, self.stress_data["hand_movement"], label="Hand Movement")
        plt.title('stress detected from Hand Movement')
        plt.xlabel("Time")
        plt.ylabel("Hand Movement Level")
        plt.legend()

        # Plot Emotions
        plt.subplot(3, 2, 5)
        plt.plot(seconds, self.stress_data["emotions"], label="Emotions")
        plt.title('Stress detected from emotions')
        plt.xlabel("Time")
        plt.ylabel("Emotions Level")
        plt.legend()

        # Plot Aggregated Stress
        plt.subplot(3, 2, 6)
        plt.plot(seconds, self.stress_data["aggregated"], label="Aggregated", linestyle='dashed', linewidth=2, color='black')
        plt.plot(seconds, self.stress_data["blink"], label="Blink")
        plt.plot(seconds, self.stress_data["eyebrows"], label="Eyebrows")
        plt.plot(seconds, self.stress_data["lip_movement"], label="Lip Movement")
        plt.plot(seconds, self.stress_data["hand_movement"], label="Hand Movement")
        plt.plot(seconds, self.stress_data["emotions"], label="Emotions")
        plt.xlabel("Time")
        plt.title('Aggregated Stress Levels')
        plt.ylabel("Aggregated Stress Level")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    video_path = "stressed.mp4"
    app = StressDetectionApp()
    app.run(video_path)
