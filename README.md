# dense-flow
Dense optical flow and RGB extraction, rewritten for my purposes (extract all RGB, flow_x, flow_y frames from a video, resized appropriately, and write to a specified output folder)

Most code taken from https://github.com/agethen/dense-flow, with changes to remove functionality I didn't need and add functionality I did. All credit goes to https://github.com/agethen

## Prerequisites
You need to build opencv 4 with CUDA.

Refer to my notes on installing opencv 4 with CUDA support on Ubuntu 18.04: https://github.com/daveboat/OpenCV_Ubuntu_Instructions

## Building
Edit Makefile with your opencv include and library directories, then run

```
mkdir build
make
```

## Running
The executable takes the following arguments:
```
-f --vidFile    filename of video
-o --outFolder  output folder - rbg saved in outFolder/rgb/, flow saved in outFolder/flow_x and outFolder/flow_y
-r --resize     resize video so that smaller of width and height is this number of pixels, with bilinear interpolation and preserving aspect ratio. set <=0 for no resize (Default = 256)
-b --bound      optical flow value upper and lower limit: values outside of (-bound, bound) are truncated. (Default = 20)
-t --type       optical flow algorithm (0 = Farneback, 1 = TVL1, 2 = Brox). (Default = 1)
-d --device_id  gpu id to use (Default = 0)
-s --step         number of frames to skip when saving optical flow and rgb frames (Default = 1)
```

Typical usage example:
```
time build/denseFlow_gpu --vidFile="video.avi" --outFolder="video/"

Processing file: video.avi (123 frames)

real	0m8.246s
user	0m3.772s
sys 0m4.428s
```

## New
Typical usage example:
```
mkdir build
make
python main.py --datasetpath=videos/ --outputpath=output
```