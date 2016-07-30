'''Read SAC file into Numpy array'''
import numpy as np

# Read SAC
from future.utils import native_str
from . import header as HD
import os
import sys


def read(file_name, byteorder=None, checksize=False):
    '''Customed version of obspy sac reader

    '''

    # Check byte order
    is_byteorder_specified = byteorder is not None
    if not is_byteorder_specified:
        byteorder = sys.byteorder

    if byteorder == 'little':
        endian_str = '<'
    elif byteorder == 'big':
        endian_str = '>'
    else:
        raise ValueError("Unrecognized byteorder. Use {'little', 'big'}")

    with open(file_name, 'rb') as fh:
        # --------------------------------------------------------------
        # READ HEADER
        # The sac header has 70 floats, 40 integers, then 192 bytes
        #    in strings. Store them in array (and convert the char to a
        #    list). That's a total of 632 bytes.
        # --------------------------------------------------------------

        hf = np.array(memoryview(fh.read(4 * 70))
                      ).view(native_str(endian_str + 'f4')).copy()
        hi = np.array(memoryview(fh.read(4 * 40))
                      ).view(native_str(endian_str + 'i4')).copy()
        hs = np.array(memoryview(fh.read(24 * 8))
                      ).view(native_str('|S8')).copy()

        if not is_valid_byteorder(hi):
            if is_byteorder_specified:
                # specified but not valid. you dun messed up.
                raise ValueError("Incorrect byteorder {}".format(byteorder))
            else:
                # not valid, but not specified.
                # swap the dtype interpretation (dtype.byteorder), but keep the
                # bytes, so the arrays in memory reflect the bytes on disk
                hf = hf.newbyteorder('S')
                hi = hi.newbyteorder('S')

        # check header lengths
        if len(hf) != 70 or len(hi) != 40 or len(hs) != 24:
            hf = hi = hs = None
            raise ValueError("Cannot read all header values")

        npts = hi[HD.INTHDRS.index('npts')]

        # check file size
        if checksize:
            cur_pos = fh.tell()
            fh.seek(0, os.SEEK_END)
            length = fh.tell()
            fh.seek(cur_pos, os.SEEK_SET)
            th_length = (632 + 4 * int(npts))
            if length != th_length:
                msg = "Actual and theoretical file size are inconsistent.\n" \
                      "Actual/Theoretical: {}/{}\n" \
                      "Check that headers are consistent with time series."
                raise ValueError(msg.format(length, th_length))

        # --------------------------------------------------------------
        # READ DATA
        # --------------------------------------------------------------
        data = np.array(memoryview(fh.read(int(npts) * 4))
                        ).view(native_str(endian_str + 'f4')).copy()

        if len(data) != npts:
            fh.close()
            print("Cannot read all data points")

    fh.close()

    return data


def is_valid_byteorder(hi):
    nvhdr = hi[HD.INTHDRS.index('nvhdr')]
    return (0 < nvhdr < 20)

if __name__ == '__main__':
    pass
