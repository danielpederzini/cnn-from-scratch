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

    def apply_augmentation(self, image: Image.Image, flip_chance: float = 0, color_jitter_chance: float = 0) -> Image.Image:
        """Apply lightweight augmentation to a PIL image."""
        image = self.apply_random_crop(image)

        if random.random() < flip_chance:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < color_jitter_chance:
            image = self.apply_color_jitter(image)        

        return image
    
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

    def load_image(self, image_path: str, normalize: bool = False, flip_chance: float = 0, color_jitter_chance: float = 0) -> cp.ndarray:
        """
        Load a single image into GPU memory.

        Args:
            image_path: Path to the image file
            normalize: Whether to scale pixel values into the [0, 1] range and
                apply per-channel mean/std normalization
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
    
    def load_images(self, normalize: bool = False, flip_chance: float = 0, color_jitter_chance: float = 0) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load all images from the dataset.

        Args:
            normalize: Whether to apply per-channel normalization after scaling
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter

        Returns:
            Tuple of (images, labels) where:
            - images: CuPy array of shape (num_samples, channels, height, width)
            - labels: CuPy array of one-hot encoded labels of shape (num_samples, num_classes)
        """
        images = []
        
        for image_path in self.image_paths:
            try:
                images.append(self.load_image(image_path=image_path, normalize=normalize, flip_chance=flip_chance, color_jitter_chance=color_jitter_chance))
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        images = cp.stack(images, axis=0)
        labels = self.encode_labels(self.labels[:len(images)], one_hot=True)

        return images, labels
    
    def load_batch(
        self,
        indices: List[int],
        normalize: bool = False,
        one_hot: bool = True,
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load a batch of images by their indices.
        
        Args:
            indices: List of image indices to load
            normalize: Whether to apply per-channel normalization after scaling
            one_hot: Whether to one-hot encode labels
            flip_chance: Probability of applying horizontal flip during augmentation
            color_jitter_chance: Probability of applying color jitter during augmentation

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
                image_array = self.load_image(image_path=image_path, normalize=normalize, flip_chance=flip_chance, color_jitter_chance=color_jitter_chance)
                batch_images.append(image_array)
                batch_labels.append(self.labels[idx])
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
        
        batch_images = cp.stack(batch_images, axis=0)
        batch_labels = self.encode_labels(batch_labels, one_hot=one_hot)
        
        return batch_images, batch_labels

    def iter_batches(
        self,
        batch_size: int,
        normalize: bool = False,
        one_hot: bool = True,
        shuffle: bool = False,
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> Iterator[Tuple[cp.ndarray, cp.ndarray]]:
        """
        Iterate over the dataset one batch at a time.

        Args:
            batch_size: Number of samples per batch
            normalize: Whether to apply per-channel normalization after scaling
            one_hot: Whether to one-hot encode labels
            shuffle: Whether to shuffle sample order before iteration
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter

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
                flip_chance=flip_chance,
                color_jitter_chance=color_jitter_chance
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
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> None:
        """
        Plot a single image by its index.
        
        Args:
            index: Image index to plot
            figsize: Tuple of (width, height) for the figure size
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
        """
        try:
            image = Image.open(self.image_paths[index])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.apply_augmentation(image=image, flip_chance=flip_chance, color_jitter_chance=color_jitter_chance)
            if self.target_size is not None:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            class_name = self.classes[self.labels[index]]
            class_id = self.class_ids[self.labels[index]]
            
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.title(f"Class: {class_name} [{class_id}] (Index: {index})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting image at index {index}: {e}")
    
    def plot_batch(
        self,
        indices: List[int],
        figsize: Tuple[int, int] = (12, 10),
        flip_chance: float = 0,
        color_jitter_chance: float = 0
    ) -> None:
        """
        Plot multiple images in a grid.
        
        Args:
            indices: List of image indices to plot
            figsize: Tuple of (width, height) for the figure size
            flip_chance: Probability that augmentation uses a horizontal flip
            color_jitter_chance: Probability that augmentation uses color jitter
        """
        num_images = len(indices)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            try:
                image = Image.open(self.image_paths[idx])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = self.apply_augmentation(image=image, flip_chance=flip_chance, color_jitter_chance=color_jitter_chance)
                if self.target_size is not None:
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
                
                class_name = self.classes[self.labels[idx]]
                class_id = self.class_ids[self.labels[idx]]
                axes[i].imshow(image)
                axes[i].set_title(f"{class_name}\n[{class_id}]")
                axes[i].axis('off')
            except Exception as e:
                print(f"Error plotting image at index {idx}: {e}")
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
