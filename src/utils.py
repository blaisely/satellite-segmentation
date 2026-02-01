import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Utils:
    @staticmethod
    def visualize(title: str, save: bool = False, class_rgb_values=None, class_names=None, **images):
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 6))

        if n_images == 1:
            axes = [axes]

        for idx, (name, image) in enumerate(images.items()):
            axes[idx].imshow(image)
            axes[idx].set_title(name.replace('_', ' ').title(), fontsize=18)
            axes[idx].axis('off')

        fig.suptitle(title, fontsize=22)

        if class_rgb_values is not None and class_names is not None:
            legend_patches = Utils.create_segmentation_legend(class_rgb_values, class_names)
            fig.legend(
                handles=legend_patches,
                loc='lower center',
                ncol=min(len(class_names), 5),
                fontsize=10,
                frameon=False
            )

        plt.tight_layout()
        plt.show()

        if save:
            plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')

    @staticmethod
    def create_segmentation_legend(class_rgb_values, class_names):
        patches = []
        for rgb, name in zip(class_rgb_values, class_names):
            color = [c / 255 for c in rgb]
            patches.append(mpatches.Patch(color=color, label=name))
        return patches
    
    @staticmethod
    def visualize_batch(
        images,
        masks,
        preds,
        class_rgb_values,
        class_names,
        title="Predictions",
        save=False,
        save_path="predictions.png"
    ):
        n = len(images)
        fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))

        if n == 1:
            axes = axes.reshape(1, 3)

        for i in range(n):
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title("Original Image", fontsize=14)

            axes[i, 1].imshow(
                Utils.colour_code_segmentation(masks[i], class_rgb_values)
            )
            axes[i, 1].set_title("Ground Truth", fontsize=14)

            axes[i, 2].imshow(
                Utils.colour_code_segmentation(preds[i], class_rgb_values)
            )
            axes[i, 2].set_title("Prediction", fontsize=14)

            for j in range(3):
                axes[i, j].axis("off")

        # Legend (single, shared)
        legend_patches = Utils.create_segmentation_legend(
            class_rgb_values, class_names
        )
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=len(class_names),
            fontsize=16
        )

        fig.suptitle(title, fontsize = 20)
        plt.tight_layout(pad = 1.5)

        if save:
            plt.savefig(save_path, dpi=300)

        plt.show()
    
    @staticmethod
    # Perform one hot encoding on label
    def one_hot_encode(label, label_values):
        """
        Convert a segmentation image label array to one-hot format
        by replacing each pixel value with a vector of length num_classes
        # Arguments
            label: The 2D array segmentation image label
            label_values
            
        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of num_classes
        """
        semantic_map = []
        for colour in label_values:
            equality = np.equal(label, colour)
            class_map = np.all(equality, axis = -1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)

        return semantic_map
    
    @staticmethod    
    # Perform reverse one-hot-encoding on labels / preds
    def reverse_one_hot(image, label_values):
        """
        Converts an RGB mask [H, W, 3] into a 2D label map [H, W]
        where each pixel is the index of the matching color in label_values.
        """
        # Create an empty 2D array
        semantic_map = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Map each RGB color to its corresponding index
        for i, color in enumerate(label_values):
            # Find pixels that match the [R, G, B] triplet exactly
            equality = np.all(image == color, axis=-1)
            semantic_map[equality] = i
            
        return semantic_map
    
    @staticmethod
    # Perform colour coding on the reverse-one-hot outputs
    def colour_code_segmentation(image, label_values):
        colour_codes = np.array(label_values)
        x = colour_codes[image.astype(int)]
        
        return x
    
    @staticmethod
    def denormalize(image: np.array):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (image * std + mean).clip(0, 1)