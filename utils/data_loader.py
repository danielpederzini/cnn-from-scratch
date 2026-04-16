import os
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
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
        
        self.image_paths = []
        self.labels = []
        self._collect_images()
    
    def _collect_images(self):
        """Collect all image paths and their corresponding labels."""
        for class_name in self.classes:
            class_dir = self.split_path / class_name
            image_files = sorted([f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for image_file in image_files:
                image_path = class_dir / image_file
                self.image_paths.append(str(image_path))
                self.labels.append(self.class_to_idx[class_name])
    
    def load_images(self) -> Tuple[np.ndarray, np.ndarray]:
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
                image = Image.open(image_path)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                if self.target_size is not None:
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
                
                image_array = np.array(image)
                image_array = np.transpose(image_array, (2, 0, 1))
                
                images.append(image_array)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
        
        images = np.stack(images, axis=0)
        labels = np.array(self.labels[:len(images)])
        
        return images, labels
    
    def load_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of images by their indices.
        
        Args:
            indices: List of image indices to load
            
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
                image = Image.open(image_path)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                if self.target_size is not None:
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
                
                image_array = np.array(image)
                image_array = np.transpose(image_array, (2, 0, 1))
                
                batch_images.append(image_array)
                batch_labels.append(self.labels[idx])
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
        
        batch_images = np.stack(batch_images, axis=0)
        batch_labels = np.array(batch_labels)
        
        return batch_images, batch_labels
    
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
            
            image_array = np.array(image)
            return tuple(np.transpose(image_array, (2, 0, 1)).shape)
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
