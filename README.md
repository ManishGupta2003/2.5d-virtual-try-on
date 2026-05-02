# 👕 2.5D Virtual Try-On

A real-time virtual clothing try-on application using advanced 2.5D perspective warping and pose estimation. Try different clothes in real-time through your webcam with AI-powered pose detection and intelligent cloth overlay blending.

## ✨ Features

- 🎥 **Real-Time Overlay** - See clothes rendered on your body instantly
- 🤖 **Pose Detection** - Uses MediaPipe for accurate body keypoint detection
- 📐 **2.5D Perspective Warping** - Advanced cloth deformation matching body shape:
  - Torso warping with perspective transform (4-point mapping)
  - Sleeve warping with triangular affine patches
  - Natural cloth draping and fitting
- 🚀 **Performance Optimization**:
  - Frame skipping for stable processing
  - Exponential smoothing for jitter reduction
  - Clamped jump detection to prevent sudden movements
  - Adaptive fast fallback when FPS drops
  - In-memory cache for resized overlays
- 🎨 **Multiple Clothing Options** - Pre-loaded assets and custom uploads
- 📸 **Snapshot Capture** - Save your virtual try-on results
- 🎯 **Landmark Visualization** - Optional pose landmark display
- 💨 **Lightweight Model** - Uses MediaPipe's efficiency-optimized pose model

## 🛠️ Tech Stack

- **Python 3.7+**
- **OpenCV (cv2)** - Image processing and video capture
- **MediaPipe** - Real-time pose estimation
- **NumPy** - Numerical computations
- **Pillow** - Image manipulation
- **Matplotlib** - Visualization (optional)

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam/Camera
- 2GB+ RAM (recommended)
- CPU with reasonable performance (GPU optional)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ManishGupta2003/2.5d-virtual-try-on.git
cd 2.5d-virtual-try-on
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r require.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy pillow matplotlib
```

### 4. Prepare Asset Directory
Create an `assets` folder and add clothing images:
```bash
mkdir assets
# Add your clothing PNG/JPG files here:
# - tshirt.png
# - jacket.png
# - hoodie.png
# etc.
```

## 📖 Usage

### Basic Launch
```bash
python main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **q** | Quit application |
| **c** | Switch to next clothing item |
| **x** | Switch to previous clothing item |
| **u** | Upload custom clothing image |
| **s** | Save current frame as snapshot |
| **d** | Toggle pose landmark visualization |
| **h** | Show help menu |

### Example Workflow
```bash
python main.py
# Camera opens with default clothing
# Press 'c' to cycle through available clothes
# Press 'u' to upload your own clothing image
# Press 's' to capture a screenshot
# Press 'q' to exit
```

## 📁 Project Structure

```
2.5d-virtual-try-on/
├── main.py                 # Main application (pose detection & UI)
├── utils/
│   └── overlay_utils.py   # 2.5D warping & blending algorithms
├── assets/                 # Clothing image directory
│   ├── tshirt.png
│   ├── jacket.png
│   ├── hoodie.png
│   └── ...
├── require.txt            # Dependencies list
└── README.md              # This file
```

## 🔧 Core Components

### **main.py**
Main application entry point with:
- Video capture and processing pipeline
- MediaPipe pose detection integration
- Smoothing and jump clamping for stability
- UI rendering and keyboard input handling
- Clothing asset management
- Snapshot saving functionality

**Key Configuration:**
```python
CAM_INDEX = 0              # Webcam index (0 = default)
TARGET_W, TARGET_H = 640, 480  # Resolution
POSE_SCALE = 0.5           # Downscale factor for pose detection
PROCESS_EVERY_N = 2        # Process every N frames
MIN_FPS_FOR_FULL_WARP = 14 # Use fast mode below this FPS
ALPHA = 0.35              # Smoothing factor (0-1)
MAX_JUMP = 0.10           # Max allowed normalized jump
```

### **utils/overlay_utils.py**
Advanced 2.5D warping and blending utilities:

#### `overlay_clothes()` - 2.5D Full Warp
- Warps clothing using perspective and affine transforms
- Maps torso region to shoulders→hips area
- Applies bilateral sleeve warping using triangular patches
- Handles alpha blending with proper masking

#### `overlay_clothes_fast()` - Fallback Mode
- Simple bounding box extraction
- Fast scaling and alpha blending
- Used when FPS drops below threshold
- Maintains visual quality at lower performance cost

#### Helper Functions
- `_ensure_alpha()` - Converts images to BGRA format
- `_warp_perspective()` - Perspective transform with masking
- `_warp_affine_patch()` - Triangular affine warping
- `_alpha_blend_region()` - Smooth blending with alpha mask
- `clear_overlay_cache()` - Memory optimization

## 🎯 How It Works

### 1. **Pose Detection**
- Captures video frames at 640x480
- Processes every 2nd frame with MediaPipe pose estimation
- Extracts 33 body landmarks (shoulders, elbows, wrists, hips, etc.)

### 2. **Smoothing & Stability**
- Applies exponential smoothing to detected landmarks
- Clamps sudden jumps to prevent jitter
- Maintains history for temporal coherence

### 3. **Cloth Warping**
- **Torso**: Maps clothing corners to shoulder and hip points using perspective transform
- **Sleeves**: Warps left/right sleeve regions using two triangular patches per arm
- Creates natural cloth deformation matching body pose

### 4. **Alpha Blending**
- Extracts alpha channel from clothing image
- Blends warped cloth with video frame
- Maintains proper transparency and compositing

### 5. **Adaptive Performance**
- Monitors FPS in real-time
- Switches to fast mode (simple scaling) if FPS < 14
- Caches resized overlays to reduce memory allocations

## 🎨 Adding Custom Clothing

### Method 1: Asset Directory
1. Create PNG/JPG images of clothing (transparent background recommended)
2. Save to `assets/` folder
3. Update `DEFAULT_ASSETS` list in main.py
4. Restart application

### Method 2: Runtime Upload
1. Press **'u'** while application is running
2. Select image file via file dialog
3. Clothing is instantly loaded and applied

**Image Tips:**
- Use PNG with transparency for best results
- Recommended size: 500x800 pixels
- Place clothing centered in image
- White or light backgrounds work well

## 📊 Configuration & Tuning

### Improve Stability
```python
ALPHA = 0.2          # Lower = smoother (less responsive)
MAX_JUMP = 0.05      # Tighter clamping (less jittery)
PROCESS_EVERY_N = 3  # Skip more frames (faster, less accurate)
```

### Improve Responsiveness
```python
ALPHA = 0.7          # Higher = more responsive
MAX_JUMP = 0.15      # Looser clamping (more movement)
PROCESS_EVERY_N = 1  # Process every frame (slower)
```

### Adaptive Performance
```python
MIN_FPS_FOR_FULL_WARP = 20  # Use full warp at higher FPS
# or
MIN_FPS_FOR_FULL_WARP = 10  # Be more aggressive with fast mode
```

## 🚨 Troubleshooting

### Webcam Not Opening
```bash
# Check available cameras
python -c "import cv2; print(cv2.getBuildInformation())"

# Try different camera index
CAM_INDEX = 1  # Change in main.py
```

### Poor Pose Detection
- Ensure good lighting
- Face camera directly
- Keep full body in frame
- Wear contrasting clothing for better skeleton detection

### Laggy Performance
- Lower resolution: Change `TARGET_W, TARGET_H` to 480x360
- Skip more frames: Increase `PROCESS_EVERY_N`
- Reduce `MIN_FPS_FOR_FULL_WARP` threshold
- Close other applications

### Clothing Not Fitting Properly
- Adjust source triangles in `overlay_clothes()` function
- Modify clothing asset dimensions
- Use simpler, less wrinkled clothing images

### Memory Issues
- Clear cache periodically: `clear_overlay_cache()`
- Reduce number of loaded assets
- Restart application

## 📈 Performance Benchmarks

| Resolution | FPS | Mode | Notes |
|-----------|-----|------|-------|
| 640x480 | 18-24 | Full 2.5D | Good quality |
| 480x360 | 25-30 | Full 2.5D | Faster processing |
| 640x480 | 8-12 | Fast Fallback | Lower quality |
| 320x240 | 30+ | Fast Fallback | Real-time, low detail |

*Performance varies based on hardware and background complexity.*

## 🔬 Mathematical Foundation

### Perspective Transform (Torso)
```
4-point source → 4-point destination
Uses cv2.getPerspectiveTransform() for homography matrix
```

### Affine Transform (Sleeves)
```
3-point source → 3-point destination
Uses cv2.getAffineTransform() for linear mapping
```

### Alpha Blending Formula
```
Output = (α × Overlay) + (1 - α) × Background
Where α = normalized alpha channel (0-1)
```

## 📚 Resources

- [MediaPipe Pose Documentation](https://mediapipe.dev/solutions/pose)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Perspective Transform Theory](https://docs.opencv.org/master/d9/d0c/group__imgproc__shape__transform.html)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Multi-person support
- [ ] 3D body model integration
- [ ] Real fabric physics simulation
- [ ] GPU acceleration with CUDA
- [ ] Mobile app port
- [ ] Better sleeve fitting algorithm
- [ ] Dynamic clothing animation
- [ ] Color matching features

### How to Contribute
1. Fork the repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Commit changes: `git commit -m 'Add improvement'`
4. Push branch: `git push origin feature/improvement`
5. Submit Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Manish Gupta**
- GitHub: [@ManishGupta2003](https://github.com/ManishGupta2003)
- Project: [2.5D Virtual Try-On](https://github.com/ManishGupta2003/2.5d-virtual-try-on)

## 🎯 Roadmap

### v1.1 (Upcoming)
- [ ] Multi-person detection
- [ ] Improved sleeve fitting
- [ ] Custom color adjustment

### v1.2
- [ ] 3D mesh body model
- [ ] Physics-based cloth simulation
- [ ] Real-time fabric texture mapping

### v2.0
- [ ] Mobile app (iOS/Android)
- [ ] Cloud API for integration
- [ ] AR glasses support

## 💡 Tips & Tricks

### Better Results
- Stand 1-2 meters from camera
- Ensure full body is visible
- Wear fitting clothes for better pose detection
- Use good lighting (avoid backlighting)

### Custom Clothing Dataset
- Collect high-quality clothing images
- Use consistent background (white/transparent)
- Maintain similar aspect ratios
- Create variants (front/back views)

### Advanced Tweaking
- Edit source triangle coordinates in `overlay_clothes()`
- Adjust cloth positioning percentages
- Modify blending alpha values
- Experiment with interpolation methods

## ⚠️ Known Limitations

- Single person per frame
- Requires visible body landmarks
- Clothing must be roughly garment-shaped
- Limited to upper body (shoulders to hips)
- No physics-based cloth simulation
- Accuracy depends on pose detection quality

## 📞 Support & Issues

Found a bug? Have a suggestion?
- Open an [Issue](https://github.com/ManishGupta2003/2.5d-virtual-try-on/issues)
- Check existing issues first
- Provide clear reproduction steps
- Include system info and error logs

---

**Made with ❤️ by Manish Gupta**

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
