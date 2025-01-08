import pytest
import cv2
import io
import numpy as np

@pytest.fixture
def image_fixture(width=100, height=100, channel=3, fill_color=[255, 0, 0]):
    # Create an example image using NumPy defined by WxHxC
    img = np.zeros((width, height, channel), dtype=np.uint8)
    img[:] = fill_color  # Fill image with red color

    # Encode the image as PNG to bytes
    _, buffer = cv2.imencode('.png', img)
    image_bytes = io.BytesIO(buffer)

    # Set a filename to simulate a file upload
    image_bytes.name = 'test_image.png'
    image_bytes.seek(0)

    return image_bytes