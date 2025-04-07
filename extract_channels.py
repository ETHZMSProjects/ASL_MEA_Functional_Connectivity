import numpy as np

def extract_channel_data(data, chan_list, channel_axis=1):
    """
    Selects specific channels along the specified axis while keeping all other dimensions unchanged.

    Parameters:
        data (numpy array): Multi-dimensional array containing neural data.
        chan_list (list or array): List of channel indices to select.
        channel_axis (int): The axis corresponding to channels.

    Returns:
        numpy array: A subset of `data` containing only the selected channels.
    """
    # Create a tuple of slices, with `:` for all axes except `channel_axis`
    slices = tuple(slice(None) if i != channel_axis else np.array(chan_list) for i in range(data.ndim))
    
    return data[slices]

def extract_original_channels(indices,chan_list):
    return [chan_list[idx] for idx in indices]

def list_to_8x8_array(lst):
    """
    Converts a list of length 64 into an 8x8 NumPy array.
    
    Parameters:
        lst (list): A list containing 64 elements.
    
    Returns:
        np.ndarray: An 8x8 NumPy array with the same values.
    
    Raises:
        ValueError: If the input list is not of length 64.
    """
    if len(lst) != 64:
        raise ValueError("Input list must have exactly 64 elements.")
    
    return np.array(lst).reshape(8, 8)

# Area 44 Superior
area_44_superior = [
    192, 193, 208, 216, 160, 165, 178, 185,
    194, 195, 209, 217, 162, 167, 180, 184,
    196, 197, 211, 218, 164, 170, 177, 189,
    198, 199, 210, 219, 166, 174, 173, 187,
    200, 201, 213, 220, 168, 176, 183, 186,
    202, 203, 212, 221, 172, 175, 182, 191,
    204, 205, 214, 223, 161, 169, 181, 188,
    206, 207, 215, 222, 163, 171, 179, 190
]

# Area 44 Inferior
area_44_inferior = [
    129, 144, 150, 158, 224, 232, 239, 255,
    128, 142, 152, 145, 226, 233, 242, 241,
    130, 135, 148, 149, 225, 234, 244, 243,
    131, 138, 141, 151, 227, 235, 246, 245,
    134, 140, 143, 153, 228, 236, 248, 247,
    132, 146, 147, 155, 229, 237, 250, 249,
    133, 137, 154, 157, 230, 238, 252, 251,
    136, 139, 156, 159, 231, 240, 254, 253
]

area_44 = area_44_superior + area_44_inferior

# Area 6v Superior
area_6v_superior = [
    62, 51, 43, 35, 94, 87, 79, 78,
    60, 53, 41, 33, 95, 86, 77, 76,
    63, 54, 47, 44, 93, 84, 75, 74,
    58, 55, 48, 40, 92, 85, 73, 72,
    59, 45, 46, 38, 91, 82, 71, 70,
    61, 49, 42, 36, 90, 83, 69, 68,
    56, 52, 39, 34, 89, 81, 67, 66,
    57, 50, 37, 32, 88, 80, 65, 64
]


# Area 6v Inferior
area_6v_inferior = [
    125, 126, 112, 103, 31, 28, 11, 8,
    123, 124, 110, 102, 29, 26, 9, 5,
    121, 122, 109, 101, 27, 19, 18, 4,
    119, 120, 108, 100, 25, 15, 12, 6,
    117, 118, 107, 99, 23, 13, 10, 3,
    115, 116, 106, 97, 21, 20, 7, 2,
    113, 114, 105, 98, 17, 24, 14, 0,
    127, 111, 104, 96, 30, 22, 16, 1
]

area_6v = area_6v_superior + area_6v_inferior

