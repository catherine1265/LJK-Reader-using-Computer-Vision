import cv2
import matplotlib.pyplot as plt


def print_step(step, msg):
    print(f"  [{step}] {msg}")


def show(img, title='', cmap=None, size=(10, 8)):
    plt.figure(figsize=size)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_row(images, titles, size=(18, 5)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=size)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
