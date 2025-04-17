import numpy
def transfer_numpy_to_float(data):
    if isinstance(data, dict):
        return {k: transfer_numpy_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [transfer_numpy_to_float(item) for item in data]
    elif isinstance(data, (numpy.integer, )):
        return int(data)
    elif isinstance(data, (numpy.floating, )):
        return float(data)
    elif isinstance(data, numpy.ndarray):
        return data.tolist()
    else:
        return data