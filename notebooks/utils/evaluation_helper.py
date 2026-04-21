import cupy as cp
import matplotlib.pyplot as plt

class EvaluationHelper:
    """
    Helper functions for evaluating model predictions and visualizing results.
    
    Provides methods to collect prediction results, build confusion matrices, and plot visualizations
    such as confusion matrices and per-class accuracy charts.
    """

    @staticmethod
    def collect_prediction_results(model, data_loader, batch_size):
        """Collects predictions, true labels, and confidence scores for a model on a dataset."""
        all_indices = []
        all_true_labels = []
        all_pred_labels = []
        all_confidences = []

        dataset_indices = list(range(len(data_loader)))

        for start in range(0, len(dataset_indices), batch_size):
            batch_indices = dataset_indices[start:start + batch_size]
            x_batch, y_batch = data_loader.load_batch(
                indices=batch_indices,
                normalize=True,
                one_hot=True
            )

            outputs = model.forward(input=x_batch)
            y_pred = outputs[-1]

            predicted_labels = cp.argmax(y_pred, axis=1)
            true_labels = cp.argmax(y_batch, axis=1)
            confidences = cp.max(y_pred, axis=1)

            all_indices.extend(batch_indices)
            all_true_labels.extend(cp.asnumpy(true_labels).tolist())
            all_pred_labels.extend(cp.asnumpy(predicted_labels).tolist())
            all_confidences.extend(cp.asnumpy(confidences).tolist())

        return {
            "indices": all_indices,
            "true_labels": all_true_labels,
            "pred_labels": all_pred_labels,
            "confidences": all_confidences,
        }

    @staticmethod
    def build_confusion_matrix(true_labels, pred_labels, num_classes):
        """Builds a confusion matrix from true and predicted labels."""
        confusion = cp.zeros((num_classes, num_classes), dtype=cp.int32)
        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion[true_label, pred_label] += 1
        return cp.asnumpy(confusion)

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, class_names):
        """Plots a confusion matrix with class labels and counts."""
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(confusion_matrix, cmap="Blues")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        for row in range(len(class_names)):
            for col in range(len(class_names)):
                value = confusion_matrix[row, col]
                ax.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > confusion_matrix.max() / 2 else "black"
                )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_per_class_accuracy(confusion_matrix, class_names):
        """Plots per-class accuracy based on the confusion matrix."""
        class_totals = confusion_matrix.sum(axis=1)
        class_correct = confusion_matrix.diagonal()
        per_class_accuracy = [
            (correct / total) if total > 0 else 0.0
            for correct, total in zip(class_correct, class_totals)
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(class_names, per_class_accuracy, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Class Accuracy")
        ax.tick_params(axis="x", rotation=45)

        for bar, accuracy in zip(bars, per_class_accuracy):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                accuracy + 0.02,
                f"{accuracy:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prediction_gallery(data_loader, prediction_results, mask, title, num_images=12, cols=4):
        """Plots a gallery of predictions based on a mask function."""
        selected = [
            (index, true_label, pred_label, confidence)
            for index, true_label, pred_label, confidence in zip(
                prediction_results["indices"],
                prediction_results["true_labels"],
                prediction_results["pred_labels"],
                prediction_results["confidences"],
            )
            if mask(true_label, pred_label)
        ]

        selected = sorted(selected, key=lambda item: item[3], reverse=True)[:num_images]

        rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for axis, (index, true_label, pred_label, confidence) in zip(axes, selected):
            image = data_loader.load_image(
                image_path=data_loader.image_paths[index],
                normalize=False
            )
            image = cp.asnumpy(cp.transpose(image, (1, 2, 0))).clip(0.0, 1.0)

            axis.imshow(image)
            axis.set_title(
                f"pred={data_loader.classes[pred_label]}\ntrue={data_loader.classes[true_label]}\nconf={confidence:.3f}",
                fontsize=10
            )
            axis.axis("off")

        for axis in axes[len(selected):]:
            axis.axis("off")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()