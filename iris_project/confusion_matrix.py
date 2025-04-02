import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(y_true, y_pred, num_classes = 3):
    conf_matrix = np.zeros((num_classes, num_classes))

    for pred, true in zip(y_pred, y_true):
        conf_matrix[int(true) -1, int(pred) -1] += 1

    return conf_matrix

def plot_cm(cm, class_names, title="Confusion Matrix", ax=None, error_rate=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        show_plot = True
    else:
        show_plot = False

    # Plot the heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Construct title with optional error rate
    if error_rate is not None:
        full_title = f"{title}\nError Rate: {error_rate:.2%}"
    else:
        full_title = title
    ax.set_title(full_title)

    plt.colorbar(im, ax=ax)

    # Set axis labels and ticks
    num_classes = len(class_names)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.grid(False)
    if show_plot:
        plt.tight_layout()
        plt.show()
