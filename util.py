import numpy as np
import binvox
import gzip

def read_binvox(filename):
    with open(filename, 'rb') as f:
        model = binvox.read_as_3d_array(f)
        data = model.data.astype(np.float32)
        return np.expand_dims(data, -1)

def save_binvox(filename, data):
    dims = data.shape
    translate = [0.0, 0.0, 0.0]
    model = binvox.Voxels(data, dims, translate, 1.0, 'xyz')
    with open(filename, 'wb') as f:
        model.write(f)

def read_schematics(filename):
    with open(filename, 'rb') as f:
        if ("binvox" not in filename):
            model = read_schematic_as_3d_array(filename)
        else:
            model = binvox.read_as_3d_array(f)
        data = model.data.astype(np.float32)
        return np.expand_dims(data, -1)

def read_schematic_as_3d_array(filename):
    import binvox_rw as binvox
    import struct
    import binascii
    import codecs
    import numpy as np

    def getSizeBytes(ba, tagName):
        blocksAmountOfBytes = ba[ba.find(tagName):]
        blocksAmountOfBytes = blocksAmountOfBytes[len(tagName):len(tagName) + 4]
        return int(codecs.encode(blocksAmountOfBytes, 'hex'), 16);

    def getBytesDataForTagName(ba, tagName):
        blocksBytes = ba[ba.find(tagName):]
        blocksBytes = blocksBytes[len(tagName) + 4:(len(tagName) + 4) + getSizeBytes(ba, tagName)]
        return blocksBytes

    def getIntValue(ba, tagName):
        blocksAmountOfBytes = ba[ba.find(tagName):]
        blocksAmountOfBytes = blocksAmountOfBytes[len(tagName):len(tagName) + 2]
        return int(codecs.encode(blocksAmountOfBytes, 'hex'), 16);

    data = gzip.open(filename, 'rb')
    ba = bytearray(data.read())

    maxHeight = getIntValue(ba, "Height") - 1
    maxLength = getIntValue(ba, "Length") - 1
    maxWidth = getIntValue(ba, "Width") - 1

    currentHeight = 0
    currentLength = 0
    currentWidth = 0

    voxArray = np.ndarray(shape=(32, 32, 32), dtype=bool)
    currHeightArr = np.ndarray(shape=(maxHeight + 1), dtype=bool)
    currLengthArr = np.zeros(shape=(maxLength + 1))

    for (byte) in getBytesDataForTagName(ba, "Blocks"):
        if (byte == 0):
            voxArray[currentHeight, currentLength, currentWidth] = False
        else:
            voxArray[currentHeight, currentLength, currentWidth] = True

        if (currentHeight == maxHeight):
            if (currentLength == maxLength):
                currentLength = 0
                currentWidth = currentWidth + 1
            else:
                currentLength = currentLength + 1

            currentHeight = 0
        else:
            currentHeight = currentHeight + 1    

    vox = binvox.Voxels(voxArray, [32, 32, 32], [0.0, 0.0, 0.0], 41.133, 'xyz')
    return vox
