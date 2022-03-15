import cv2
from cv2 import dnn_superres


def super_res(name, dest, path_to_weights):
    # Create an SR object
    image = cv2.imread(name)
    size = image.shape
    if size[0] < 700 and size[1] < 700:
        print("yay")
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read image

        # Read the desired model
        sr.readModel(path_to_weights)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 4)

        # Upscale the image
        result = sr.upsample(image)

        # Save the image
        cv2.imwrite(dest, result)
    else:
        cv2.imwrite(dest, image)
