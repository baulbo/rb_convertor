"""

Script to use the convertor. You can uncomment the part you want to use and update the rosbag topics and output locations.

"""

import argparse
import os
import numpy as np
from pathlib import Path
import h5py

from h5py._hl.files import File 
from utils.converter import PointCloud2Converter, ImageConverter, CompressedImageConverter
from utils.converter import HDF5File, KittiFile
from utils.converter import DataType, VideoCoding

def main():
  bag_path_str = './data/rosbag.bag'

  ###############################################
  # SYNCED EXAMPLE writing to KITTI like format #
  ###############################################

  # write point cloud data as KITTI .bin format 
  kitti_out_dir = "output/data"
  point_cloud_topic = '/ouster/points'
  # TODO: update ROS topic here ^^
  kitti_file = KittiFile(dtype=DataType.POINTCLOUD, out_dir=kitti_out_dir)
  point_cloud_converter = PointCloud2Converter(topic=point_cloud_topic, out_file=kitti_file)
  lidar_timestamps = point_cloud_converter.convert_messages(bag_path=Path(bag_path_str), start_offset=5, duration=2, accumulate_timestamps=True)

  # write camera (video) image data to h5py (.h264 coding)
  image_topic = '/interfacea/link1/image/h264'
  # TODO: update ROS topic here ^^
  kitti_file = KittiFile(dtype=DataType.COMPRESSEDIMAGE, out_dir=kitti_out_dir)
  compressed_img_converter = CompressedImageConverter(topic=image_topic, out_file=kitti_file, coding=VideoCoding.H264)
  compressed_img_converter.convert_messages(bag_path=Path(bag_path_str), ref_timestamps=lidar_timestamps)

  ###############################################
  #   SYNCED EXAMPLE writing HDF5 file format   #
  ###############################################
  """
  OUT_DIR = 'output/'
  out_path = os.path.join(OUT_DIR, "synced_data.h5")

  # write point cloud data to hdf5 file format
  point_cloud_topic = '/ouster/points'
  # TODO: update ROS topic here ^^
  hdf5_file = HDF5File(file_path=out_path, dtype=DataType.POINTCLOUD, dset_name='/lidar/point_cloud', mode='w')
  point_cloud_converter = PointCloud2Converter(topic=point_cloud_topic, out_file=hdf5_file)
  lidar_timestamps = point_cloud_converter.convert_messages(bag_path=Path(bag_path_str), start_offset=5, duration=2, accumulate_timestamps=True)

  # write lidar signal images to hdf5 file (synced with PointCloud2 timestamps accumulated from previous conversion)
  lidar_image_topics = ['/ouster/signal_image', '/ouster/reflec_image', '/ouster/range_image', '/ouster/nearir_image']
  # TODO: update ROS topics here ^^
  hdf5_dset_names = ['/lidar/signal_image', '/lidar/reflec_image', '/lidar/range_image', '/lidar/nearir_image']
  # TODO: update HDF5 datasets here ^^
  for topic, dset_name in zip(lidar_image_topics, hdf5_dset_names):
    hdf5_file = HDF5File(file_path=out_path, dtype=DataType.IMAGE, dset_name=dset_name, mode='a')
    img_converter = ImageConverter(topic=topic, out_file=hdf5_file)
    img_converter.convert_messages(bag_path=Path(bag_path_str), ref_timestamps=lidar_timestamps, desired_dtype=np.uint8)

  # write camera (video) image data to h5py (.h264 coding)
  image_topic = '/interfacea/link2/image/h264'
  # TODO: update ROS topic here ^^
  # use mode 'a' to not overwrite the hdf5 file
  hdf5_file = HDF5File(file_path=out_path, dtype=DataType.COMPRESSEDIMAGE, dset_name='/camera/link1', mode='a')
  compressed_img_converter = CompressedImageConverter(topic=image_topic, out_file=hdf5_file, coding = VideoCoding.H264)
  compressed_img_converter.convert_messages(bag_path=Path(bag_path_str), ref_timestamps=lidar_timestamps)

  """
  ##################################################
  # NON-SYNCED EXAMPLE writing to HDF5 file format #
  ##################################################
  """
  OUT_DIR = 'output/'
  out_path = os.path.join(OUT_DIR, "data.h5")

  # write point cloud data to hdf5 file format
  point_cloud_topic = '/ouster/points'
  # TODO: update ROS topic here ^^
  hdf5_file = HDF5File(file_path=out_path, dtype=DataType.POINTCLOUD, dset_name='/lidar/point_cloud', mode='w')
  point_cloud_converter = PointCloud2Converter(topic=point_cloud_topic, out_file=hdf5_file)
  point_cloud_converter.convert_messages(bag_path=Path(bag_path_str), start_offset=5, duration=5)

  # write camera (video) image data to h5py (.h264 coding)
  image_topic = '/interfacea/link1/image/h264'
  # TODO: update ROS topic here ^^
  # use mode 'a' to not overwrite the hdf5 file
  hdf5_file = HDF5File(file_path=out_path, dtype=DataType.COMPRESSEDIMAGE, dset_name='/camera/link1', mode='a')
  img_converter = CompressedImageConverter(topic=image_topic, out_file=hdf5_file, coding = VideoCoding.H264)
  img_converter.convert_messages(bag_path=Path(bag_path_str), start_offset=5, duration=5)
  """

if __name__ == "__main__":
  main()

