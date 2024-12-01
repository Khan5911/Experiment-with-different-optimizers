
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import cv2

def visualize_activation_maps(img_path, model, layer_name):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if required

    # Select a convolutional layer
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get activations
    activations = intermediate_layer_model.predict(img_array)

    # Visualize activations
    num_filters = activations.shape[-1]
    plt.figure(figsize=(15, 15))
    for i in range(num_filters):
        plt.subplot(8, 8, i + 1)  # Adjust grid size based on number of filters
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.show()

    # Generate heatmap
    heatmap = np.mean(activations[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # Resize to match original image size

    # Convert heatmap to RGB and overlay
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + np.uint8(img)
    
    # Show the superimposed image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    from tensorflow.keras.applications import VGG16
    model = VGG16(weights='imagenet', include_top=False)

    # Replace 'path_to_image.jpg' with the path of your image
    img_path = 'path_to_image.jpg'
    layer_name = 'block5_conv3'  # Replace with your desired layer
    visualize_activation_maps(img_path, model, layer_name)
