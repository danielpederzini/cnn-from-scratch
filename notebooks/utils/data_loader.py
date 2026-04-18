import os
import random
import cupy as cp
from PIL import Image
from typing import Iterator, Tuple, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt

class ImagenetteDataLoader:
    """
    A data loader class for the imagenette dataset.
    
    Loads images and converts them to numpy arrays with shape (channels, height, width).
    """
    
    def __init__(self, root_path: str, split: str = 'train', target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize the ImagenetteDataLoader.
        
        Args:
            root_path: Path to the imagenette2 directory
            split: Either 'train' or 'val'
            target_size: Optional tuple (height, width) to resize images. If None, uses original size.
        """
        self.root_path = Path(root_path)
        self.split = split
        self.target_size = target_size
        self.split_path = self.root_path / split
        
        if not self.split_path.exists():
            raise ValueError(f"Split path does not exist: {self.split_path}")
        
        self.classes = sorted([d for d in os.listdir(self.split_path) 
                               if os.path.isdir(self.split_path / d)])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
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
    
    def collect_images(self):
        """Collect all image paths and their corresponding labels."""
        for class_name in self.classes:
            class_dir = self.split_path / class_name
            image_files = sorted([file for file in os.listdir(class_dir) 
                                 if file.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for image_file in image_files:
                image_path = class_dir / image_file
                self.image_paths.append(str(image_path))
                self.labels.append(self.class_to_idx[class_name])

    def load_image(self, image_path: str, normalize: bool = False, aug_chance: float = 0, flip_chance: float = 0) -> cp.ndarray:
        """
        Load a single image into GPU memory.

        Args:
            image_path: Path to the image file
            normalize: Whether to scale pixel values into the [0, 1] range and
                apply per-channel mean/std normalization

        Returns:
            Image tensor of shape (channels, height, width)
        """
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if random.random() < aug_chance:
            if random.random() < flip_chance:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                width, height = image.size
                left, upper, right, lower = width/8, height/8, 3*width/8, 3*height/8
                image = image.crop((left, upper, right, lower))

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
    
    def load_images(self, normalize: bool = False, aug_chance: float = 0, flip_chance: float = 0) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load all images from the dataset.
        
        Returns:
            Tuple of (images, labels) where:
            - images: numpy array of shape (num_samples, channels, height, width)
            - labels: numpy array of class indices of shape (num_samples,)
        """
        images = []
        
        for image_path in self.image_paths:
            try:
                images.append(self.load_image(image_path=image_path, normalize=normalize, aug_chance=aug_chance, flip_chance=flip_chance))
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
        aug_chance: float = 0,
        flip_chance: float = 0
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Load a batch of images by their indices.
        
        Args:
            indices: List of image indices to load
            normalize: Whether to scale image pixels to the [0, 1] range
            one_hot: Whether to one-hot encode labels
            aug_chance: Probability of applying augmentation
            flip_chance: Probability of applying horizontal flip during augmentation
        Returns:
            Tuple of (batch_images, batch_labels) where:
            - batch_images: numpy array of shape (batch_size, channels, height, width)
            - batch_labels: numpy array of class indices
        """
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            try:
                image_path = self.image_paths[idx]
                image_array = self.load_image(image_path=image_path, normalize=normalize, aug_chance=aug_chance, flip_chance=flip_chance)
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
        aug_chance: float = 0,
        flip_chance: float = 0
    ) -> Iterator[Tuple[cp.ndarray, cp.ndarray]]:
        """
        Iterate over the dataset one batch at a time.

        Args:
            batch_size: Number of samples per batch
            normalize: Whether to scale image pixels to the [0, 1] range
            one_hot: Whether to one-hot encode labels
            shuffle: Whether to shuffle sample order before iteration

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
                aug_chance=aug_chance,
                flip_chance=flip_chance
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
        """Return the list of class names."""
        return self.classes
    
    def get_image_shape(self, index: int = 0) -> Tuple[int, int, int]:
        """
        Get the shape of an image (channels, height, width).
        
        Args:
            index: Image index (default: first image)
            
        Returns:
            Tuple of (channels, height, width)
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
    
    def plot_image(self, index: int, figsize: Tuple[int, int] = (6, 6)):
        """
        Plot a single image by its index.
        
        Args:
            index: Image index to plot
            figsize: Tuple of (width, height) for the figure size
        """
        try:
            image = Image.open(self.image_paths[index])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.target_size is not None:
                image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            class_name = self.classes[self.labels[index]]
            
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.title(f"Class: {class_name} (Index: {index})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting image at index {index}: {e}")
    
    def plot_batch(self, indices: List[int], figsize: Tuple[int, int] = (12, 10)):
        """
        Plot multiple images in a grid.
        
        Args:
            indices: List of image indices to plot
            figsize: Tuple of (width, height) for the figure size
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
                if self.target_size is not None:
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
                
                class_name = self.classes[self.labels[idx]]
                axes[i].imshow(image)
                axes[i].set_title(f"{class_name}")
                axes[i].axis('off')
            except Exception as e:
                print(f"Error plotting image at index {idx}: {e}")
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
