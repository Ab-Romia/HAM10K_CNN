import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from sklearn.metrics import jaccard_score

def evaluate_model(model, val_loader_class, history, class_names):
    device = next(model.parameters()).device
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader_class:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, task='class')
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Plot Accuracy Curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Classification Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # --- Correlation Matrix ---
    corr = np.corrcoef(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Prediction vs True Correlation')
    plt.show()


def dice_coefficient(pred, target, epsilon=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

def evaluate_segmentation_model(model, val_loader_seg, threshold=0.5, max_visualize=5):
    device = next(model.parameters()).device
    model.eval()

    dice_scores = []
    ious = []
    accuracies = []

    visualized = 0

    with torch.no_grad():
        for images, masks in val_loader_seg:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, task='seg')  # shape: (B, 1, H, W)
            preds = (outputs > threshold).float()

            for i in range(images.size(0)):
                pred = preds[i]
                target = masks[i]

                dice = dice_coefficient(pred, target)
                iou = jaccard_score(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), zero_division=0)
                acc = (pred == target).float().mean().item()

                dice_scores.append(dice.item())
                ious.append(iou)
                accuracies.append(acc)

                # --- Visualization ---
                if visualized < max_visualize:
                    input_img = images[i].cpu().permute(1, 2, 0).numpy()
                    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                    gt_mask = target.squeeze().cpu().numpy()
                    pred_mask = pred.squeeze().cpu().numpy()

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(input_img)
                    axs[0].set_title("Input Image")
                    axs[1].imshow(gt_mask, cmap='gray')
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(pred_mask, cmap='gray')
                    axs[2].set_title("Prediction")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                    visualized += 1

    print(f"Avg Dice Coefficient: {np.mean(dice_scores):.4f}")
    print(f"Avg IoU:              {np.mean(ious):.4f}")
    print(f"Avg Pixel Accuracy:   {np.mean(accuracies):.4f}")