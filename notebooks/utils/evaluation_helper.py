import cupy as cp
import matplotlib.pyplot as plt

class EvaluationHelper:
    """
    Stateful helper for evaluating one model on one dataset and visualizing the results.
    """

    def __init__(self, model, data_loader, batch_size):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.class_names = data_loader.get_class_names()
        self.num_classes = len(data_loader.get_class_ids())
        self._prediction_results = None
        self._confusion_matrix = None

    def refresh(self):
        """Clear cached evaluation artifacts so they are recomputed on next access."""
        self._prediction_results = None
        self._confusion_matrix = None

    def collect_prediction_results(self, force=False):
        """Collect predictions, true labels, and confidence scores for the configured dataset."""
        if self._prediction_results is not None and not force:
            return self._prediction_results

        all_indices = []
        all_true_labels = []
        all_pred_labels = []
        all_confidences = []

        dataset_indices = list(range(len(self.data_loader)))

        for start in range(0, len(dataset_indices), self.batch_size):
            batch_indices = dataset_indices[start:start + self.batch_size]
            x_batch, y_batch = self.data_loader.load_batch(
                indices=batch_indices,
                normalize=True,
                one_hot=True
            )

            outputs = self.model.forward(input=x_batch)
            y_pred = outputs[-1]

            predicted_labels = cp.argmax(y_pred, axis=1)
            true_labels = cp.argmax(y_batch, axis=1)
            confidences = cp.max(y_pred, axis=1)

            all_indices.extend(batch_indices)
            all_true_labels.extend(cp.asnumpy(true_labels).tolist())
            all_pred_labels.extend(cp.asnumpy(predicted_labels).tolist())
            all_confidences.extend(cp.asnumpy(confidences).tolist())

        self._prediction_results = {
            "indices": all_indices,
            "true_labels": all_true_labels,
            "pred_labels": all_pred_labels,
            "confidences": all_confidences,
        }
        return self._prediction_results

    def collect_evaluation_artifacts(self, force=False):
        """Collect cached prediction results and confusion matrix for the configured dataset."""
        return {
            "prediction_results": self.collect_prediction_results(force=force),
            "confusion_matrix": self.build_confusion_matrix(force=force),
        }

    def get_selected_predictions(self, mask):
        """Return predictions filtered by a boolean mask function."""
        prediction_results = self.collect_prediction_results()
        return [
            (index, true_label, pred_label, confidence)
            for index, true_label, pred_label, confidence in zip(
                prediction_results["indices"],
                prediction_results["true_labels"],
                prediction_results["pred_labels"],
                prediction_results["confidences"],
            )
            if mask(true_label, pred_label)
        ]

    def build_confusion_matrix(self, force=False):
        """Build or return the cached confusion matrix."""
        if self._confusion_matrix is not None and not force:
            return self._confusion_matrix

        prediction_results = self.collect_prediction_results(force=force)
        confusion = cp.zeros((self.num_classes, self.num_classes), dtype=cp.int32)
        for true_label, pred_label in zip(prediction_results["true_labels"], prediction_results["pred_labels"]):
            confusion[true_label, pred_label] += 1
        self._confusion_matrix = cp.asnumpy(confusion)
        return self._confusion_matrix

    def plot_confusion_matrix(self):
        """Plot the cached confusion matrix with class labels and counts."""
        confusion_matrix = self.build_confusion_matrix()
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(confusion_matrix, cmap="Blues")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(self.class_names)))
        ax.set_yticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        for row in range(len(self.class_names)):
            for col in range(len(self.class_names)):
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

    def plot_per_class_accuracy(self):
        """Plot per-class accuracy based on the cached confusion matrix."""
        confusion_matrix = self.build_confusion_matrix()
        class_totals = confusion_matrix.sum(axis=1)
        class_correct = confusion_matrix.diagonal()
        per_class_accuracy = [
            (correct / total) if total > 0 else 0.0
            for correct, total in zip(class_correct, class_totals)
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(self.class_names, per_class_accuracy, color="steelblue")
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

    def plot_prediction_gallery(self, mask, title, num_images=12, cols=4):
        """Plot a gallery of dataset examples selected by a prediction mask."""
        selected = self.get_selected_predictions(mask=mask)

        selected = sorted(selected, key=lambda item: item[3], reverse=True)[:num_images]

        rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for axis, (index, true_label, pred_label, confidence) in zip(axes, selected):
            image = self.data_loader.load_image(
                image_path=self.data_loader.image_paths[index],
                normalize=False
            )
            image = cp.asnumpy(cp.transpose(image, (1, 2, 0))).clip(0.0, 1.0)

            axis.imshow(image)
            axis.set_title(
                f"pred={self.data_loader.classes[pred_label]}\ntrue={self.data_loader.classes[true_label]}\nconf={confidence:.3f}",
                fontsize=10
            )
            axis.axis("off")

        for axis in axes[len(selected):]:
            axis.axis("off")

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def select_feature_map_layers(self, outputs, max_layers=4):
        """Select representative spatial feature-map layers from a forward pass."""
        spatial_layer_indices = [
            layer_index
            for layer_index, output in enumerate(outputs)
            if getattr(output, "ndim", 0) == 4 and output.shape[2] > 1 and output.shape[3] > 1
        ]

        if len(spatial_layer_indices) <= max_layers:
            return spatial_layer_indices

        selected = []
        last_position = -1
        for step in range(max_layers):
            position = round(step * (len(spatial_layer_indices) - 1) / (max_layers - 1))
            if position == last_position:
                position = min(position + 1, len(spatial_layer_indices) - 1)
            selected.append(spatial_layer_indices[position])
            last_position = position

        return selected

    def plot_feature_map_progression(
        self,
        indices,
        max_layers=4,
        figsize_scale=2.4,
        colormap="viridis"
    ):
        """Plot one row per image with the input and mean activations for selected layers."""
        plot_examples = []
        for index in indices:
            image_path = self.data_loader.image_paths[index]
            image_tensor = self.data_loader.load_image(
                image_path=image_path,
                normalize=False
            )
            model_input = self.data_loader.load_image(
                image_path=image_path,
                normalize=True
            )[cp.newaxis, ...]
            outputs = self.model.forward(input=model_input)
            prediction = outputs[-1][0]

            plot_examples.append({
                "index": index,
                "image_tensor": image_tensor,
                "outputs": outputs,
                "predicted_label": int(cp.argmax(prediction).item()),
                "confidence": float(cp.max(prediction).item()),
                "true_label": self.data_loader.labels[index],
            })

        selected_layers = self.select_feature_map_layers(
            outputs=plot_examples[0]["outputs"],
            max_layers=max_layers
        )

        rows = len(indices)
        cols = len(selected_layers) + 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize_scale * cols, figsize_scale * rows)
        )

        if rows == 1:
            axes = [axes]

        for row, example in enumerate(plot_examples):
            original_image = cp.asnumpy(cp.transpose(example["image_tensor"], (1, 2, 0))).clip(0.0, 1.0)
            axes[row][0].imshow(original_image)
            axes[row][0].set_title(
                f"Input\ntrue={self.data_loader.classes[example['true_label']]}\npred={self.data_loader.classes[example['predicted_label']]} ({example['confidence']:.3f})",
                fontsize=10
            )
            axes[row][0].axis("off")

            for col, layer_index in enumerate(selected_layers, start=1):
                feature_maps = example["outputs"][layer_index][0]
                mean_map = cp.asnumpy(cp.mean(feature_maps, axis=0))
                axes[row][col].imshow(mean_map, cmap=colormap)
                axes[row][col].set_title(
                    f"Layer {layer_index + 1}\nmean activation",
                    fontsize=10
                )
                axes[row][col].axis("off")

        plt.suptitle("Feature Map Progression Across Layers", fontsize=16)
        plt.tight_layout()
        plt.show()