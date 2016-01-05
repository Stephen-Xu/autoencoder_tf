import numpy as np
import gzip



def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def get_data_from_minst():

    num_images = 60000
    rows = 28
    cols = 28
    max_value = 0xFF

    filename = '../datasets/train-images-idx3-ubyte.gz'


    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
          raise ValueError(
              'Invalid magic number %d in MNIST image file: %s' %
              (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows* cols)
        data = data.astype(float)
        data -= max_value / 2.0
        data /= max_value
    
    return data

