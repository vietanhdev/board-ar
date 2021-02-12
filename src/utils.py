import numpy as np

def overlay_png(background, rgb_channels, alpha_channel):
    """Place an image with alpha channel (transparent) on a background image

    Args:
        background (np.array): Background image
        rgb_channels (np.array): BGR channels of the transparent image. Shape: (height, width, 3)
        alpha_channel (np.array): The alpha channel of the transparent image. Shape: (height, width)

    Returns:
        np.array: Output image
    """

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = background.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)
