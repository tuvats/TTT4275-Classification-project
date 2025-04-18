import numpy as np
import matplotlib.pyplot as plt
import random

def plot_correct_images(image_preds, labels, max_images=6, save_path=None):
    plt.figure(figsize=(14, 3))
    for i in range(max_images):
        idx = random.randint(0, len(image_preds) - 1)
        img = image_preds[idx].reshape(28, 28)
        label = labels[idx]

        plt.subplot(1, max_images, i + 1)
        plt.imshow(img)
        plt.title(f"Predicted: {label}")
        plt.axis('off')

    plt.suptitle("Correctly Classified Images")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)

    plt.show()


def plot_wrong_images(image_preds, pred_labels, true_labels, max_images=6, save_path=None):
    plt.figure(figsize=(14, 3))
    for i in range(max_images):
        idx = random.randint(0, len(image_preds) - 1)
        img = image_preds[idx].reshape(28, 28)
        pred_label = pred_labels[idx]
        true_label = true_labels[idx]

        plt.subplot(1, max_images, i + 1)
        plt.imshow(img)
        plt.title(f"Predicted: {pred_label}")
        plt.xlabel(f"True: {true_label}")
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Wrongly Classified Images")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)

    plt.show()
