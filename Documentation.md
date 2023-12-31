## Stress Detection System Documentation

# Introduction
This code implements a Stress Detection application utilizing facial and hand landmarks obtained from a video stream. It utilizes the MediaPipe library for face mesh and hand tracking, OpenCV for video processing and visualization, NumPy for numerical operations, and Matplotlib for plotting.
*******************************************************************************************************

- ### Class: StressDetectionApp
Constructor (__init__)
Initializes the StressDetectionApp class.
Creates instances of MediaPipe FaceMesh and Hands.
Initializes variables for facial and hand landmarks, stress data storage, and aggregated stress calculation.

- ### Method: calculate_eye_blink
Calculates the average eye aspect ratio to detect eye blinks.
Method: eye_aspect_ratio
Computes the eye aspect ratio based on the Euclidean distances between eye landmarks.
- ### Method: calculate_eyebrows
Calculates the asymmetry of eyebrow height.
- ### Method: calculate_lip_movement
Calculates the vertical distance between upper and lower lips.
- ### Method: calculate_emotions
Placeholder for emotion calculation based on specific facial landmarks.
Method: calculate_hand_movement
Calculates hand movement based on the average change in Y coordinates of hand landmarks.

- ### Method: process_frame
Processes each frame of the video.
Extracts facial and hand landmarks using MediaPipe.
Calls methods to calculate stress features (eye blink, eyebrow asymmetry, lip movement, hand movement, and emotions).
Draws landmarks on the frame.
Calculates aggregated stress level.
- ### Method: draw_landmarks
Draws facial landmarks on the given frame.
- ### Method: run
Runs the stress detection on a given video file.
Processes video frames, displays results in real-time, and saves the stress data to a JSON file.
Displays stress features plot.
- ### Method: plot_stress
Plots stress features (eye blink, eyebrows, lip movement, hand movement, emotions) over time.
Displays aggregated stress level.

##### Input 
- mp4 video 


https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/7048eb4e-43df-41c2-b912-8993cdb9bce9

******************************************************************************************************************************
![scrennshot_of_processed_video](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/c8c3f4d4-cabb-42d1-831a-8aa2b13f1f8f)

![scrennshot_of_processed_video_](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/08161f70-2cc5-4889-a53e-c3c4f5080b49)




##### Output
- Real-time visualization of stress features.
Aggregated stress level plot.

![blinks](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/53def58e-db77-43cc-8e84-5fd7f6728e5a) <br>
![eb](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/3ae0cd7c-2697-442e-afae-9e7aa4c5045b) <br>
![agg](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/d91b2117-334d-4734-a342-853d0e90308d) <br>
![Hm](https://github.com/mahdihammi/Vivianet_AI-Internship_Technical_Assessment/assets/89527502/fa5b576d-56d7-4a9a-90bf-5dce06ca7a04) <br>






