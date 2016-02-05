import numpy as np
import gzip



def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

def get_data_from_minst():

    num_images = 60000
    rows = 28
    cols = 28
    max_value = 0xFF

    filename = '../datasets/train-images-idx3-ubyte.gz'
    filename_lab ='../datasets/train-labels-idx1-ubyte.gz'

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
    
    lab = extract_labels(filename_lab)
    
    return data,lab

def get_test_from_mnist():
    num_images = 10000
    rows = 28
    cols = 28
    max_value = 0xFF

    filename = '../datasets/t10k-images-idx3-ubyte.gz'
    filename_lab ='../datasets/t10k-labels-idx1-ubyte.gz'

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

    lab = extract_labels(filename_lab)

    return data,lab
