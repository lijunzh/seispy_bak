"""Read/write data into HDF5 format"""
import os
import warnings

import h5py
import numpy as np


def write(file, data, overwrite=False):
    """Save data to HDF5"""
    # Check if file exists and overwrite
    if os.path.exists(file):
        if not overwrite:
            response = input("File {} exists. \n".format(file) +
                             "Overwrite existing file? [y/n]: ")
            if response.lower() == 'n':
                print("Keep original file. Saving process is aborted.")
                return None
            elif (response.lower() == 'y') or (response == ""):
                pass
            else:
                raise ValueError("Unknown input.")
        else:
            warnings.warn("Overwrite original file {}".format(file))

    # save data to hdf5 file
    with h5py.File(file, 'w') as f:
        _recursive_write_(f, 'data', data)
    print("Data saved in {}.".format(file))
    return None


def _recursive_write_(obj, name, data):
    """Recursively write data to obj(hf/group)
    Warning: only use it with write function. Not intend for other usage.
    """

    # print("The current data is ({})".format(data))
    if isinstance(data, dict):
        # print("Create group named ({})".format(name))
        g = obj.create_group(name)
        for key, item in data.items():
            # print(key, item)
            # print("The current key is ({})".format(key))
            _recursive_write_(g, key, item)
    else:
        # print("Save ({}) with value of ({})".format(name, data))
        if isinstance(data, np.ndarray):
            obj.create_dataset(name, data=data, compression="gzip", compression_opts=9)
        else:
            obj.create_dataset(name, data=data)


def read(file):
    """Read data from HDF5
    Warning: read function returns only a (dict of) narray(s), you have to convert to original format manually.
    """

    warnings.warn("Read function only returns a (dict of) list(s), you have to convert to original format manually.")
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            data = _recursive_read_({}, "data", f)
    else:
        raise ValueError("File {} does not exists!".format(file))

    if isinstance(data, dict):
        if data:
            return data
        else:
            raise ValueError("Empty dataset.")
    else:
        return data


def _recursive_read_(dic, name, obj):
    """Recursively read data from obj(hf/group)
    Warning: only use it with read function. Not intend for other usage.
    """

    # print("Current object is {}".format(obj))
    if isinstance(obj, h5py.File):
        return _recursive_read_(dic, name, obj[name])
    elif isinstance(obj, h5py.Group):
        for key, item in obj.items():
            dic[key] = _recursive_read_({}, key, item)
        return dic
    elif isinstance(obj, h5py.Dataset):
        return obj.value



if __name__ == '__main__':
    dic = {'int': 1,
           'str': "s",
           'boolean': True,
           'list': [1, 2],
           'tuple': (1, 2),
           'dict': {'dic_key': "dic_item"},
           'array': np.array([1, 2]),
           }

    print("Testing dict:")
    for key, item in dic.items():
        print("\t{: <8}\t{}".format(key, item))

    write("tmp/test1.h5", dic, overwrite=True)
    write("tmp/test2.h5", (True, False), overwrite=True)
    data1 = read("tmp/test1.h5")
    for key, item in data1.items():
        print("\t{: <8}\t{}".format(key, item))

    data2 = read("tmp/test2.h5")
    print(data2)
