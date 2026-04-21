import os
import random
import cupy as cp
from PIL import Image, ImageEnhance
from typing import Iterator, Tuple, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt

class ImagenetteDataLoader:
    """
    Data loader for the Imagenette dataset.

    Loads images, optionally applies lightweight augmentation and
    normalization, and returns CuPy tensors shaped as
    (channels, height, width).
    """

    IMAGENETTE_LABELS = {
        'n01440764': 'Tench',
        'n02102040': 'English Springer',
        'n02979186': 'Cassette Player',
        'n03000684': 'Chain Saw',
        'n03028079': 'Church',
        'n03394916': 'French Horn',
        'n03417042': 'Garbage Truck',
        'n03425413': 'Gas Pump',
        'n03445777': 'Golf Ball',
        'n03888257': 'Parachute',
    }
    
    def __init__(self, root_path: str, split: str = 'train', target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImagenetteDataLoader.
        
        Args:
            root_path: Path to the imagenette2 directory
            split: Either 'train' or 'val'
            target_size: Optional tuple (width, height) used by PIL when resizing.
                If None, the original image size is preserved.
        """
        self.root_path = Path(root_path)
        self.split = split
        self.target_size = target_size
        self.split_path = self.root_path / split
        
        if not self.split_path.exists():
            raise ValueError(f"Split path does not exist: {self.split_path}")
        
        self.class_ids = sorted([
            directory for directory in os.listdir(self.split_path)
            if os.path.isdir(self.split_path / directory)
        ])
        self.class_to_idx = {
            class_id: idx for idx, class_id in enumerate(self.class_ids)
        }
        self.classes = [
            self.IMAGENETTE_LABELS.get(class_id, class_id)
            for class_id in self.class_ids
        ]
        self.num_classes = len(self.class_ids)
        self.channel_mean: cp.ndarray = cp.asarray(
            [0.485, 0.456, 0.406],
            dtype=cp.float32
        ).reshape(3, 1, 1)
        self.channel_std: cp.ndarray = cp.asarray(
            [0.229, 0.224, 0.225],
            dtype=cp.float32
        ).reshape(3, 1, 1)
        
        self.image_paths = []
        self.labels = []
        self.collect_images()

    def apply_random_crop(self, image: Image.Image):
        """Apply a random crop to a PIL image."""
        width, height = image.size
        crop_scale = random.uniform(0.75, 0.9)
        crop_width = max(1, int(width * crop_scale))
        crop_height = max(1, int(height * crop_scale))
        max_left = max(0, width - crop_width)
        max_upper = max(0, height - crop_height)
        left = random.randint(0, max_left) if max_left > 0 else 0
        upper = random.randint(0, max_upper) if max_upper > 0 else 0
        right = left + crop_width
        lower = upper + crop_height
        return image.crop((left, upper, right, lower))

    def apply_color_jitter(self, image: Image.Image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2) -> Image.Image:
        """Apply random brightness and contrast adjustments to a PIL image."""
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(image)
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = enhancer.enhance(factor)

        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(image)
            factor = 1.0 + random.uniform(-contrast, contrast)
            image = enhancer.enhance(factor)

        if random.random() < 0.5:
            enhancer = ImageEnhance.Color(image)
            factor = 1.0 + random.uniform(-saturation, saturation)
            image = enhancer.enhance(factor)
        
        return image

    def apply_augmentation(
        self,
        image: Image.Image,
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> Image.Image:
        """Apply lightweight augmentation to a PIL image."""
        if random.random() < random_crop_chance:
            image = self.apply_random_crop(image)

        if random.random() < flip_chance:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < color_jitter_chance:
            image = self.apply_color_jitter(image)        

        return image

    def apply_cut_mix(
        self,
        images: cp.ndarray,
        labels: cp.ndarray,
        alpha: float = 1.0
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Apply CutMix augmentation to a batch of images and one-hot labels."""
        batch_size, _, image_height, image_width = images.shape
        cut_mix_lambda = random.betavariate(alpha, alpha) if alpha > 0 else 1.0
        cut_ratio = (1.0 - cut_mix_lambda) ** 0.5
        cut_width = max(1, int(image_width * cut_ratio))
        cut_height = max(1, int(image_height * cut_ratio))

        center_x = random.randint(0, image_width - 1)
        center_y = random.randint(0, image_height - 1)

        left = max(0, center_x - cut_width // 2)
        upper = max(0, center_y - cut_height // 2)
        right = min(image_width, left + cut_width)
        lower = min(image_height, upper + cut_height)
        left = max(0, right - cut_width)
        upper = max(0, lower - cut_height)

        shuffled_indices = cp.random.permutation(batch_size)
        mixed_images = images.copy()
        mixed_images[:, :, upper:lower, left:right] = images[shuffled_indices, :, upper:lower, left:right]

        patch_area = max(0, right - left) * max(0, lower - upper)
        adjusted_lambda = 1.0 - (patch_area / float(image_width * image_height))
        mixed_labels = (adjusted_lambda * labels) + ((1.0 - adjusted_lambda) * labels[shuffled_indices])

        return mixed_images, mixed_labels

    def format_label_mix(self, label_vector: cp.ndarray, top_k: int = 2) -> str:
        """Format a one-hot or mixed label vector for display in plots."""
        flattened_labels = cp.ravel(label_vector)
        top_indices = cp.asnumpy(cp.argsort(flattened_labels)[-top_k:][::-1])
        parts: List[str] = []

        for class_index in top_indices:
            weight = float(flattened_labels[class_index].item())
            if weight <= 0:
                continue
            class_name = self.classes[int(class_index)]
            parts.append(f"{class_name} ({weight:.2f})")

        return "\n".join(parts) if parts else "Unknown"
    
    def collect_images(self) -> None:
        """Collect all image paths and their corresponding integer labels."""
        for class_id in self.class_ids:
            class_dir = self.split_path / class_id
            image_files = sorted([file for file in os.listdir(class_dir) 
                                 if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for image_file in image_files:
                image_path = class_dir / image_file
                self.image_paths.append(str(image_path))
                self.labels.append(self.class_to_idx[class_id])

    def load_image(
        self,
        image_path: str,
        normalize: bool = False,
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> cp.ndarray:
        """
        Load a single image into GPU memory.

        Args:
            image_path: Path to the image file
            normalize: Whether to scale pixel values into the [0, 1] range and
                apply per-channel mean/std normalization
            random_crop_chance: Probability that augmentation uses a random crop
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter

        Returns:
            Image tensor of shape (channels, height, width)
        """
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.apply_augmentation(
            image=image,
            random_crop_chance=random_crop_chance,
            flip_chance=flip_chance,
            color_jitter_chance=color_jitter_chance
        )

        if self.target_size is not None:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        image_array = cp.asarray(image, dtype=cp.float32)
        image_array = cp.transpose(image_array, (2, 0, 1))
        image_array /= 255.0

        if normalize:
            image_array = (image_array - self.channel_mean) / self.channel_std

        return image_array

    def encode_labels(self, labels: List[int], one_hot: bool = True) -> cp.ndarray:
        """
        Encode label indices as either one-hot vectors or integer indices.

        Args:
            labels: Integer class labels
            one_hot: Whether to return one-hot encoded labels

        Returns:
            Encoded labels tensor
        """
        label_array: cp.ndarray = cp.asarray(labels, dtype=cp.int32)

        if not one_hot:
            return label_array

        encoded: cp.ndarray = cp.zeros((len(labels), self.num_classes), dtype=cp.float32)
        encoded[cp.arange(len(labels)), label_array] = 1.0
        return encoded
    
    def load_images(
        self,
        normalize: bool = False,
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0,
        cut_mix_chance: float = 0
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load all images from the dataset.

        Args:
            normalize: Whether to apply per-channel normalization after scaling
            random_crop_chance: Probability that augmentation uses a random crop
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
            cut_mix_chance: Probability that CutMix is applied to the loaded batch

        Returns:
            Tuple of (images, labels) where:
            - images: CuPy array of shape (num_samples, channels, height, width)
            - labels: CuPy array of one-hot encoded labels of shape (num_samples, num_classes)
        """
        images = []
        
        for image_path in self.image_paths:
            try:
                images.append(
                    self.load_image(
                        image_path=image_path,
                        normalize=normalize,
                        random_crop_chance=random_crop_chance,
                        flip_chance=flip_chance,
                        color_jitter_chance=color_jitter_chance
                    )
                )
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        images = cp.stack(images, axis=0)
        labels = self.encode_labels(self.labels[:len(images)], one_hot=True)

        if random.random() < cut_mix_chance:
            images, labels = self.apply_cut_mix(images=images, labels=labels)

        return images, labels
    
    def load_batch(
        self,
        indices: List[int],
        normalize: bool = False,
        one_hot: bool = True,
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0,
        cut_mix_chance: float = 0
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load a batch of images by their indices.
        
        Args:
            indices: List of image indices to load
            normalize: Whether to apply per-channel normalization after scaling
            one_hot: Whether to one-hot encode labels
            random_crop_chance: Probability of applying a random crop during augmentation
            flip_chance: Probability of applying horizontal flip during augmentation
            color_jitter_chance: Probability of applying color jitter during augmentation
            cut_mix_chance: Probability that CutMix is applied to the loaded batch

        Returns:
            Tuple of (batch_images, batch_labels) where:
            - batch_images: CuPy array of shape (batch_size, channels, height, width)
            - batch_labels: CuPy array of class indices or one-hot encoded labels
        """
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            try:
                image_path = self.image_paths[idx]
                image_array = self.load_image(
                    image_path=image_path,
                    normalize=normalize,
                    random_crop_chance=random_crop_chance,
                    flip_chance=flip_chance,
                    color_jitter_chance=color_jitter_chance
                )
                batch_images.append(image_array)
                batch_labels.append(self.labels[idx])
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
        
        batch_images = cp.stack(batch_images, axis=0)
        batch_labels = self.encode_labels(batch_labels, one_hot=one_hot)

        if random.random() < cut_mix_chance:
            batch_images, batch_labels = self.apply_cut_mix(images=batch_images, labels=batch_labels)
        
        return batch_images, batch_labels

    def iter_batches(
        self,
        batch_size: int,
        normalize: bool = False,
        one_hot: bool = True,
        shuffle: bool = False,
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0,
        cut_mix_chance: float = 0
    ) -> Iterator[Tuple[cp.ndarray, cp.ndarray]]:
        """
        Iterate over the dataset one batch at a time.

        Args:
            batch_size: Number of samples per batch
            normalize: Whether to apply per-channel normalization after scalingW
            one_hot: Whether to one-hot encode labels
            shuffle: Whether to shuffle sample order before iteration
            random_crop_chance: Probability that augmentation uses a random crop
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
            cut_mix_chance: Probability that CutMix is applied to each yielded batch

        Yields:
            Tuples of (batch_images, batch_labels)
        """
        indices: List[int] = list(range(len(self)))

        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if not batch_indices:
                continue

            yield self.load_batch(
                indices=batch_indices,
                normalize=normalize,
                one_hot=one_hot,
                random_crop_chance=random_crop_chance,
                flip_chance=flip_chance,
                color_jitter_chance=color_jitter_chance,
                cut_mix_chance=cut_mix_chance
            )

    def get_normalization_stats(self) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Return the per-channel normalization statistics used by the loader.

        Returns:
            Tuple of (mean, std), each shaped as (3, 1, 1)
        """
        return self.channel_mean, self.channel_std
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def get_class_names(self) -> List[str]:
        """Return the list of human-readable class names."""
        return self.classes

    def get_class_ids(self) -> List[str]:
        """Return the original Imagenette synset folder names."""
        return self.class_ids
    
    def get_image_shape(self, index: int = 0) -> Optional[Tuple[int, int, int]]:
        """
        Get the shape of an image (channels, height, width).
        
        Args:
            index: Image index (default: first image)
            
        Returns:
            Tuple of (channels, height, width), or None if the image cannot be loaded
        """
        try:
            image = Image.open(self.image_paths[index])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.target_size is not None:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            image_array = cp.asarray(image, dtype=cp.float32)
            return tuple(cp.transpose(image_array, (2, 0, 1)).shape)
        except Exception as e:
            print(f"Error getting image shape: {e}")
            return None
    
    def plot_image(
        self,
        index: int,
        figsize: Tuple[int, int] = (6, 6),
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0,
        cut_mix_chance: float = 0
    ) -> None:
        """
        Plot a single image by its index.
        
        Args:
            index: Image index to plot
            figsize: Tuple of (width, height) for the figure size
            random_crop_chance: Probability that augmentation uses a random crop
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
            cut_mix_chance: Probability that CutMix is applied before plotting
        """
        try:
            plot_indices = [index]
            if cut_mix_chance > 0 and len(self) > 1:
                partner_index = random.randrange(len(self) - 1)
                if partner_index >= index:
                    partner_index += 1
                plot_indices.append(partner_index)

            batch_images, batch_labels = self.load_batch(
                indices=plot_indices,
                normalize=False,
                one_hot=True,
                random_crop_chance=random_crop_chance,
                flip_chance=flip_chance,
                color_jitter_chance=color_jitter_chance,
                cut_mix_chance=cut_mix_chance
            )

            image = cp.asnumpy(cp.transpose(batch_images[0], (1, 2, 0)))
            image = image.clip(0.0, 1.0)
            label_description = self.format_label_mix(batch_labels[0])
            
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.title(f"Label: {label_description} (Index: {index})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting image at index {index}: {e}")
    
    def plot_batch(
        self,
        indices: List[int],
        figsize: Tuple[int, int] = (12, 10),
        random_crop_chance: float = 0,
        flip_chance: float = 0,
        color_jitter_chance: float = 0,
        cut_mix_chance: float = 0
    ) -> None:
        """
        Plot multiple images in a grid.
        
        Args:
            indices: List of image indices to plot
            figsize: Tuple of (width, height) for the figure size
            random_crop_chance: Probability that augmentation uses a random crop
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
            cut_mix_chance: Probability that CutMix is applied before plotting
        """
        num_images = len(indices)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        batch_images, batch_labels = self.load_batch(
            indices=indices,
            normalize=False,
            one_hot=True,
            random_crop_chance=random_crop_chance,
            flip_chance=flip_chance,
            color_jitter_chance=color_jitter_chance,
            cut_mix_chance=cut_mix_chance
        )

        plotted_count = int(batch_images.shape[0])
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(plotted_count):
            try:
                idx = indices[i]
                image = cp.asnumpy(cp.transpose(batch_images[i], (1, 2, 0)))
                image = image.clip(0.0, 1.0)
                label_description = self.format_label_mix(batch_labels[i])
                axes[i].imshow(image)
                axes[i].set_title(f"{label_description}\n(Index: {idx})")
                axes[i].axis('off')
            except Exception as e:
                print(f"Error plotting image at index {idx}: {e}")
        
        for j in range(plotted_count, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
