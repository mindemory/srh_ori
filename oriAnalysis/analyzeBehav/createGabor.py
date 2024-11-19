import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
import svgwrite

def create_gabor_grating(size=100, frequency=5, orientation=45, sigma=0.3, output_file="gabor_grating.eps"):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    grating =  np.cos(2 * np.pi  * frequency * X)  # Sine grating
    rotated_grating = rotate(grating, -orientation, reshape=False)
    gaussian_envelope = np.exp(-((X**2 + Y**2) / (2 * sigma**2)))
    # rotated_grating = rotated_grating * gaussian_envelope
    rotated_grating = rotated_grating * gaussian_envelope

    scaled_grating = (rotated_grating - rotated_grating.min()) / (rotated_grating.max() - rotated_grating.min())

    # Create an RGB image from the scaled grating
    rgb_image = np.zeros((size, size, 3))  # Create a 3-channel (RGB) image
    rgb_image[..., 0] = scaled_grating  # Red channel
    rgb_image[..., 1] = scaled_grating  # Green channel
    rgb_image[..., 2] = scaled_grating  # Blue channel

    # Plot the rotated sinusoidal grating
    plt.imshow(rgb_image, extent=(0, 1, 0, 1))
    plt.axis('off') 
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    plt.show()

    # # Change color map to gray
    # min_gray = 0.5  # Medium gray
    # max_white = 1.0  # White
    # scaled_grating = (rotated_grating - np.min(rotated_grating)) / (np.max(rotated_grating) - np.min(rotated_grating))
    # scaled_grating = min_gray + (max_white - min_gray) * scaled_grating
    # # grating = (grating - np.min(grating)) / (np.max(grating) - np.min(grating))

    # # Apply Gaussian envelope for blurring
    # # blurred_grating = gaussian_filter(grating, sigma=sigma)
    # # blurred_grating = grating  

    # # cmap = plt.cm.gray
    # # cmap.set_under('gray')
    # plt.imshow(scaled_grating, cmap='gray', vmin=min_gray, vmax=max_white)
    # # plt.imshow(blurred_grating, cmap='gray', vmin=0.5, vmax=1)
    # plt.axis('off')
    # plt.show()

    # Save as SVG
    # dwg = svgwrite.Drawing(output_file, size=(size, size))
    # for i in range(size):
    #     for j in range(size):
    #         gray_value = int(blurred_grating[i, j] * 255)
    #         color = svgwrite.rgb(gray_value, gray_value, gray_value, '%')
    #         dwg.add(dwg.rect((j, i), (1, 1), fill=color))
    # dwg.save()

# Run the function
create_gabor_grating()