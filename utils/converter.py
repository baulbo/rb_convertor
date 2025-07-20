from typing import NewType, TypeVar, Any 

import h264decoder # https://github.com/DaWelter/h264decoder
from tqdm import tqdm
import h5py
from abc import ABC, abstractmethod
from PIL import Image as im
from rosbags.highlevel import AnyReader
import os

from pathlib import Path
import rosbags 
from enum import Enum
import numpy as np

from utils.ros_numpy.point_cloud2 import pointcloud2_to_array, get_xyz_points
from utils.ros_numpy.image import image_to_numpy

class DataType(Enum):
  """Enum for converted data types."""
  POINTCLOUD      = 1
  IMAGE           = 2
  COMPRESSEDIMAGE = 3

class File(ABC):
  """Abstract base class for writing converted data to specific file formats."""
  def __init__(self, file_path: str, dtype : DataType):
    self._file_path = file_path
    self._dtype = dtype
    self._file = None
    self.is_open = False

  def write(self, **kwargs) -> None:
    """Write data to file and open file if not opened yet."""
    # open file if not open yet
    if not self.is_open:
      self.open(**kwargs)
    # call writer corresponding to given data type
    match self._dtype:
      case DataType.POINTCLOUD:
        self._write_point_cloud(data=kwargs['data'])
      case DataType.IMAGE:
        self._write_image(**kwargs)
      case DataType.COMPRESSEDIMAGE:
        self._write_image(data=kwargs['data'])
      case _:
        raise TypeError(f"The given data type '{self._dtype}' is not supported by used file writer.")
    

  @classmethod
  def _write_image_to_disk(cls, image_path: str, data: np.ndarray):
    if image_path:
      pil_img= im.fromarray(data) 
      pil_img.save(image_path)

  @abstractmethod
  def open(self, **kwargs) -> None:
    """Opens the file as the self._file property in write mode."""
    # general opening procedure
    self.is_open = True

  @abstractmethod
  def close(self) -> None:
    """Handles closing of open file."""
    # general closing procedure
    self.is_open = False
    self._file = None

  @abstractmethod
  def _write_point_cloud(self, data: np.ndarray, **kwargs) -> None:
    raise NotImplementedError

  @abstractmethod
  def _write_image(self, data: np.ndarray, **kwargs) -> None:
    raise NotImplementedError 


class KittiFile(File):
  """File implementation to write PointCloud2 as .bin (KITTI format) and CompressedImages as .PNG to disk.
  """
  def __init__(self, dtype: DataType, out_dir: str, file_path: str = None) -> None:
    super().__init__(file_path, dtype)
    self.__frame_count = 0
    # create output file structure
    self._out_dir = out_dir
    self._lidar_out_dir = os.path.join(self._out_dir, 'lidar_points', 'data')
    self._images_out_dir = os.path.join(self._out_dir, 'images', 'data')
    if not os.path.exists(self._images_out_dir):
      os.makedirs(self._images_out_dir)
    if not os.path.exists(self._lidar_out_dir):
      os.makedirs(self._lidar_out_dir)

  def open(self, **kwargs) -> None:
    # NOTE: KITTI just writes each incoming frame to disk, so no files to open.
    super(KittiFile, self).open()
    pass

  def close(self) -> None:
    super(KittiFile, self).close()
    pass

  def _write_point_cloud(self, data: np.ndarray, **kwargs) -> None:
    # write as KITTI .bin, where data is shape (w, h, 3, 1)
    #pc = pypcd.PointCloud.from_msg(msg)
    p = np.squeeze(data)
    # source: https://github.com/PRBonn/lidar-bonnetal/issues/78 (@kosmastsk)
    p_shape = p.shape
    p = p.reshape(p_shape[0]*p_shape[1], p_shape[2]) # (w, h, 3) --> (w*h, 3)
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    intensity = np.zeros((p.shape[0]))
    arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
    arr[::4] = x
    arr[1::4] = y
    arr[2::4] = z
    arr[3::4] = intensity
    arr.astype('float32').tofile(os.path.join(self._lidar_out_dir, f'{self.__frame_count:010d}'+'.bin'))
    self.__frame_count+=1

  def _write_image(self, data: np.ndarray, desired_dtype: np.dtype = None, **kwargs) -> None:
    """Adds image to HDF5 dataset in memory but does not immediately write to disk.

    Args:
      data: image data of shape (h, w, c).
    """
    if data is not None: # NOTE: None as data will skip frame / image
      # write image to disk
      if self._images_out_dir:
        File._write_image_to_disk(os.path.join(self._images_out_dir, f"{self.__frame_count:010d}"+'.png'), data=data)
      self.__frame_count+=1

class HDF5File(File):
  def __init__(self, file_path: str, dtype: DataType, dset_name: str, mode : str = None, images_dir : str = None) -> None:
    """
    Args:
      file_path: path to output HDF5 file.
      dtype: the DataType to write.
      mode: 'w' to overwrite existing file and 'a' to append to existing HDF5 file.
      images_dir: directory to write CompressedImages to (NOTE: the directory is assumed to exist already).
    """
    super().__init__(file_path, dtype)

    # general HDF5 init
    self.__dset_name = dset_name
    self._mode  = mode
    self.__images_dir = images_dir

    # to be initialized later
    self._dset = None

  def __init_point_cloud_dataset(self, shape: tuple, chunk_dim: int = 1) -> None:
    w, h, c, _ = shape
    self._dset = self._file.create_dataset(name=self.__dset_name, 
                             shape=(w, h, c, 0),       # (..., num point clouds)
                             maxshape=(w, h, c, None), # (..., 2**64), None makes dset resizable
                             chunks=(w, h, c, chunk_dim),
                             compression="gzip", compression_opts=9
                           )

  def __init_image_dataset(self, shape: tuple, dtype: np.dtype = None) -> None:
    h, w, c = shape
    self._dset = self._file.create_dataset(name=self.__dset_name, 
                             shape=(h, w, c, 0),      # (height, width, channels e.g. rgb, num images)
                             maxshape=(h, w, c, None), # (..., 2**64), None makes dset resizable
                             dtype=dtype
                           )

  @property 
  def dset(self):
    return self._dset
  @dset.setter
  def dset(self, shape: tuple) -> None:
    match self._dtype:
      case DataType.COMPRESSEDIMAGE:
        self.__init_image_dataset(shape=shape, dtype=np.uint8)
      case _:
        raise TypeError(f"Manually setting dataset for given data type {self._dtype} is not allowed.")

  def open(self, **kwargs) -> None:
    self._file = h5py.File(self._file_path, self._mode)
    match self._dtype:
      case DataType.POINTCLOUD:
        self.__init_point_cloud_dataset(shape=kwargs['data'].shape)
      case DataType.IMAGE:
        self.__init_image_dataset(shape=kwargs['data'].shape, dtype=kwargs['data'].dtype)
      case DataType.COMPRESSEDIMAGE:
        # NOTE: CompressedImage msgs H264 encoded cannot be initialized before a 
        #         couple of msgs are decoded. Therefore, the dset.setter is used 
        #         once a frame is available.
        pass
      case _:
        raise TypeError(f"File writer was not able to open file for given data type {self._dtype}.")
    super(HDF5File, self).open()

  def close(self) -> None:
    self._file.flush()
    self._file.close()
    super(HDF5File, self).close()

  def _write_point_cloud(self, data: np.ndarray, **kwargs) -> None:
    # increase dataset size for the next chunk of point clouds
    num_pc_per_chunk = data.shape[-1]
    self._dset.resize(self._dset.shape[-1]+num_pc_per_chunk, axis=len(self._dset.shape)-1)
    # write directly to hdf5 output file
    self._dset.write_direct(data, 
                     dest_sel=np.s_[:,:,:, 
                     self._dset.shape[-1]-num_pc_per_chunk:]
                     )

  def _write_image(self, data: np.ndarray, desired_dtype: np.dtype = None, **kwargs) -> None:
    """Adds image to HDF5 dataset in memory but does not immediately write to disk.

    Args:
      data: image data of shape (h, w, c).
    """
    if data is not None: # NOTE: None as data will skip frame / image
      # increase dataset size by 1 (NOTE: assumes index increases by 1 every write call)
      num_frames_saved = self._dset.shape[-1]
      self._dset.resize(num_frames_saved+1, axis=len(self._dset.shape)-1)
      if desired_dtype is not None:
        match desired_dtype:
          case np.uint8:
            ma, mi = data.max(), data.min()
            data = ((data - ma) / (ma - mi) * 255).astype(np.uint8)
          case _:
            raise TypeError(f'The desired image dtype {desired_dtype} is not supported. Only np.uint8 is supported.')
        
      self._dset[:,:,:, num_frames_saved-1] = data

      # write image to disk
      if self.__images_dir:
        File._write_image_to_disk(os.path.join(self.__images_dir, str(num_frames_saved)+'.png'), data=data)

  def _write_image_to_disk(self, data: np.ndarray, index : int = 0):
    if self.__images_dir:
      save_path = os.path.join(self.__images_dir, str(index)+'.png')
      pil_img= im.fromarray(data) 
      pil_img.save(save_path)


class MsgConverter(ABC):
  def __init__(self, topic: str, out_file: File) -> None:
    self.topic: str = topic
    self.out_file: File = out_file
    
  @staticmethod
  def read_messages(bag_path: Path, topic: str, start_offset: int = None, duration: int = None) -> tuple[Any, int]:
    """Generator that reads messages from a rosbag's given topic.

    Args:
      bag_path: path to rosbag file (with '.bag' filename extension)
      topic: the ros topic.
      start_offset: time offset from first timestamp in bag (in seconds)
      duration: duration to convert and write mesages for (in seconds)
    """
    with AnyReader([bag_path], default_typestore=None) as reader:
      selected_conn = [[ x for x in reader.connections if x.topic == topic ][0]] # NOTE: only selects a connection topic

      # calc start and stop timestamps (in ns)
      start_t = reader.start_time if start_offset is None else reader.start_time + int( start_offset * 1e9 )
      stop_t = reader.end_time
      if duration is not None: stop_t = start_t + int( duration * 1e9 )
      total_t = stop_t - start_t
      print("Progression of timestamp between configured start and end time, might not reach 100%.")
      pbar    = tqdm(total=100)
      prev_t  = start_t 
      # start reading rosbag msgs
      for conn, t, rawdata in reader.messages(connections=selected_conn, start=start_t, stop=stop_t):
        msg = reader.deserialize(rawdata, conn.msgtype)
        pbar.update( ((t - prev_t) / total_t ) * 100 )
        prev_t = t
        yield (msg, t)
      pbar.close()

  def convert_messages(self, bag_path: Path, start_offset: int = None, duration: int = None, accumulate_timestamps: bool = False, ref_timestamps: tuple[int] = None, **kwargs) -> tuple[int] | None:
    """Converts rosbag data to other data repr. message 
        per message and writes it to output file."""
    # init variables for timestamp-based synchronisation
    timestamp_sync: bool = False if ref_timestamps is None else True
    if timestamp_sync:
      if ( ref_timestamps is None ) or ( len(ref_timestamps) == 0 ):
        raise Exception("No reference timestamps found. Reference timestamps should have been passed to constructor when timestamp syncing is enabled.")
      num_rt = len(ref_timestamps)
      rti = 0
      rt = ref_timestamps[rti]
      if ( start_offset is not None ) or ( duration is not None ):
        start_offset = None
        duration = None
        print("Configured start offset and duration are not used when timestamp syncing is enabled.")

    # convert data in messages one by one
    messages = MsgConverter.read_messages(bag_path, self.topic, start_offset, duration)
    prev_data = None
    prev_t = None
    timestamps = None if not accumulate_timestamps else list()
    for msg, t in messages:
      data = self._convert_data(msg)

      # accumulate timestamps 
      if accumulate_timestamps:
        timestamps.append(t)

      # sync 
      if timestamp_sync:
        selected_data = data
        if rt <= t: # msg t has passed reference t
          # select previous frame if its t was closer to reference t
          if ( prev_data is not None ) and ( ( rt - prev_t ) < ( t - rt ) ):
            selected_data = prev_data

          # only write data that is closest to reference timestamp
          self.out_file.write(data=selected_data, **kwargs)
          if rti+1 < num_rt:
            rti+=1
            rt = ref_timestamps[rti]
          else: break

        prev_t = t
        prev_data = data

        # all ref. samples have been processed (or no time overlap with reference timestamps)
        #if ( rti > num_rt - 1 ) or ( t > ref_timestamps[num_rt - 1] ):
        #  break

      else:
        # write all converted data when not syncing samples
        self.out_file.write(data=data, **kwargs)

    # converted data has been written to file, close it
    self.out_file.close()
    return timestamps

  @abstractmethod
  def _convert_data(self, msg: Any) -> Any:
    """Convert a single message's data to another representation."""
    raise NotImplementedError

class PointCloud2Converter(MsgConverter):
  @staticmethod
  def xyz_to_points_chunk(xyz_points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Reshapes point clouds shunk from (N, 3) -> (w, h, 3, num point clouds per chunk)"""
    points_chunk = np.reshape(xyz_points, (width, height, 3, -1))
    return points_chunk

  def _convert_data(self, msg: Any) -> np.ndarray: 
    cloud_array = pointcloud2_to_array(msg)
    xyz_points = get_xyz_points(cloud_array)                # shape (N, 3)
    reshaped_points = PointCloud2Converter.xyz_to_points_chunk(xyz_points, 
                                          msg.width, 
                                          msg.height)       # shape (w, h, 3, num pcs)
    return reshaped_points

class VideoCoding(Enum):
  """Enum for video encoding formats."""
  H264 = 1

class CompressedImageConverter(MsgConverter):
  def __init__(self, coding: VideoCoding = VideoCoding.H264, **kwargs):
    super().__init__(**kwargs)
    self.__coding = coding
    match coding:
      case VideoCoding.H264:
        self.__decoder = h264decoder.H264Decoder()
        self.__is_frame_available = False
      case _:
        raise TypeError(f"The given video coding {coding} is not supported by the used converter class.")

  def _convert_data(self, msg: Any) -> np.ndarray: 
    converted_frame = None
    frame_bytes = msg.data.tobytes()
    frames = self.__decoder.decode(frame_bytes)
    if ( not self.__is_frame_available and len(frames) > 0 ):
      _ , w, h, ls = frames[0]
      self.__is_frame_available = True
      # NOTE: implementation assumes that after ^^ is True, every subsequent chunk contains a frame

    if self.__is_frame_available:
      frame, w, h, ls = frames[0] # initially, set to current msg's frame
      if ( not self.out_file.is_open ):
        # open file earlier to create hdf5 dataset from first available frame
        self.out_file.open()
      # FIXME: setting the dataset below only for HDF5 file format is wrong since it's not MsgConverter specific...
      if ( isinstance(self.out_file, HDF5File) and not self.out_file.dset ):
        self.out_file.dset = (h, w, 3)
      if frame is not None:
        frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
        frame = frame.reshape((h, ls//3, 3))
        converted_frame = frame[:,:w,:]
    # NOTE: returns None if frame is not to be written to file (e.g. when no frames available or to skip specific frames)
    return converted_frame 

class ImageConverter(MsgConverter):
  def _convert_data(self, msg: Any) -> np.ndarray: 
    """Converts rgb, rgba, bgr, bgra, mono {8 or 16} Image messages to numpy array.
    Returns:
      Numpy ndarray containing image data of shape (height, width, channels)
    """
    return image_to_numpy(msg)
