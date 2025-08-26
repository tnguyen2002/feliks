# 🧩 Felix

**Felix** is a Rubik’s Cube solver that uses a **built-in webcam** to scan each face of the cube and then returns a **human-readable solution** to the scramble.  

It leverages **OpenCV** for:
- **Canny edge detection** (to extract cube facelets)  
- **CIEDE2000 color classification** (for robust color detection)  
- **Contour detection** (to reconstruct the cube state)  

---

## ✨ Features

- 📷 **Automatic Cube Scanning** — Uses your webcam to capture all six faces.  
- 🎨 **Robust Color Recognition** — CIEDE2000 color difference algorithm improves accuracy.  
- 🔍 **State Reconstruction** — Extracts cube configuration with contour detection.  
- 🧠 **Solver Engine** — Computes a step-by-step, human-readable solution.  


## 📱 [Demo](https://drive.google.com/file/d/1PxF8HhCKiHEwGDhYK_wz2IJOvZ7hBClJ/view?usp=sharing)


## 🚀 Getting Started

### Prerequisites
Make sure you have the following installed:
- [Python 3.9+](https://www.python.org/)  
- [OpenCV](https://opencv.org/)  
- [NumPy](https://numpy.org/)  

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/felix.git
   cd felix
