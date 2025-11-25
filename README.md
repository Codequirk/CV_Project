🌟 MirrorCloneFX – AI-Powered Gesture-Controlled Visual Effects

MirrorCloneFX is a real-time, gesture-controlled visual effects system built using OpenCV, MediaPipe, and Python.
It transforms your webcam feed into multiple artistic visual styles and lets you switch between modes using hand gestures — no keyboard required.

🚀 Features
🎥 Real-Time Webcam Processing

Captures your webcam feed and applies visual effects live

Mirror view for natural movement

✋ Hand Gesture Recognition (MediaPipe)

Gesture → Mode mapping:

Gesture	Description	Mode
✌️ Two fingers (Index + Middle)	V-sign	Dots Mode
☝️ One finger (Index only)	Pointing	Lines Mode
🤙 Thumb + Pinky (Shaka)	Hang loose	ASCII Mode
✋ Open palm (4+ fingers)	Big hand	Particles Mode

Includes:

Angle-based finger detection

Gesture history smoothing

Cooldown to avoid rapid switching

🎨 Visual Effects
1️⃣ Dots Mode

Converts the frame into glowing dots based on brightness.

2️⃣ Lines Mode

Edge-based neon line effect using Canny + dilation.

3️⃣ ASCII Mode

Turns your webcam feed into ASCII art characters with color coding.

4️⃣ Particles Mode

Generates colorful floating particles from detected hand landmarks
(with physics: gravity, velocity, lifespan).

📦 Project Structure
MirrorCloneFX/
│── main.py               # Main entry point
│── requirements.txt      # Required libraries
│── MirrorCloneFX.py      # Class containing all effects & logic
│── README.md             # Documentation

🛠️ Requirements

Install dependencies:

pip install opencv-python mediapipe numpy

▶️ How to Run
python main.py


Press Q to exit the application.

🧠 How Gesture Detection Works

MirrorCloneFX uses:

Finger joint angles (MCP–PIP–TIP)

Threshold-based classification

Majority voting history buffer

Gesture cooldown timer

This reduces noise and ensures stable gesture recognition.

💡 Tech Used

Python

OpenCV – video capture + image processing

MediaPipe Hands – gesture detection

NumPy – vector math

Deque – gesture history smoothing
