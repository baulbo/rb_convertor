# Rosbag Convertor

An extendible Python script with easy to install dependencies (no ROS required) to convert some ROS 1 and ROS 2 data to .bin and .png KITTI like format, or HDF5 format. It currently supports PointCloud2 and CompressedImage with H264 encoding (i.e a video) message formats. It was not thoroughly tested and therefore might not work as intended. The development was used to convert rather specific data formats.
  
**Workflow.**

1. Refer to the [Setup](#setup) section for installation instructions.
2. Change `main.py` to match your setup, which should be intuitive.
3. Run the script as described in the [Usage](#usage) section.
4. (for developers) Add support for the desired file format by implementing the `File` abstract base class using the [developer guide](#developer-guide).
5. (for developers) Add support for the desired ROS message format by implementing the `Converter` abstract base class using the [developer guide](#developer-guide).

## Usage

Run the command below.

```txt
# first, put your rosbag in data/rosbag.bag
#     writes to KITTI like format by default
python main.py
```

## Setup

Install [ffmpeg](https://www.ffmpeg.org) and the python packages. The setup has not been tested on Windows but it should work.

```sh
# (optional) create a virtual env
python -m venv venv

# install py packages
pip install -r requirements.txt
```

## Developer Guide

Just look at the code :) 

