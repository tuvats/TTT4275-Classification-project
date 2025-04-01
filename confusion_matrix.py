import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(y_true, y_pred, num_classes = 3):
    conf_matrix = np.zeros((num_classes, num_classes))

    for pred, true in zip(y_pred, y_true):
        conf_matrix[int(true) -1, int(pred) -1] += 1

    return conf_matrix

def plot_cm(cm, class_names, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, ax=ax)

    num_classes = len(class_names)
    #Set ticks and labels
     # Set axis labels and ticks
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Add cell values
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.grid(False)
    plt.show()