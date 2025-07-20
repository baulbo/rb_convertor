# NOTE: this code is copied from https://github.com/eric-wieser/ros_numpy
#   with minor modifications to prevent having to install the whole repository

import numpy as np

name_to_dtypes = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1)
  }

def image_to_numpy(msg):
	if not msg.encoding in name_to_dtypes:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	
	dtype_class, channels = name_to_dtypes[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

  # NOTE: grayscale (single channel) images will have shape (h, w, 1)
	return data

