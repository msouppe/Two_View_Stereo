# Plane Sweeping 2-View Stereo

## Prerequisites
- Python 3 
- Python packages: OpenCV, numpy, matplotlib.pyplot, scipy, os, time
- MAC OS or Windows 10 (Ubuntu Windows' Bash)

## Installation 
* Clone the repo:
``` 
git clone https://github.com/msouppe/Two_View_Stereo.git
```

* Navigate to `main.py`

* Run the program:
```bash
python3 main.py 
```

## Project Structure
* `main.py`: Main program
  
## Output
**Part 1**  
For the images that were taken to calibrate the camera we obtain the following:
* Intrinsic matrix K
* Radial distortion coefficients
* Reprojection mean square error
  
**Part 2**    
  
**Part 3**    
Computation for finding the camera relative pose:
* Few epipolar lines on the images.
* The matrix R<sup>R</sup><sub>R</sub> and **r**<sup>R</sup>
* Re-projected feature points on the first image

**Part 4**   
Creating a plane-sweeping stereo  
* Values for *d<sub>min</sub>* and *d<sub>max</sub>*
* N=20 warped second images
* Resulting depth image in grayscale
