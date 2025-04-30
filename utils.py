import torch


def get_device():
    """
    使用するデバイスを取得

    Returns:
        device (mps or cpu)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device is available and being used.")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, using CPU instead.")
    
    return device
