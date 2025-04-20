import copy


def train_model(model, train_loader_class, val_loader_class,
                train_loader_seg, val_loader_seg,
                criterion_class, criterion_seg, optimizer, scheduler,
                epochs=10):
    device = next(model.parameters()).device
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'train_seg_loss': [],
        'val_seg_loss': []
    }

    for epoch in range(epochs):
        model.train()

        # --- Classification Training ---
        correct = 0
        total = 0
        running_loss = 0
        for images, labels in train_loader_class:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, task='class')
            loss = criterion_class(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader_class)

        # --- Segmentation Training ---
        seg_running_loss = 0
        for images, masks in train_loader_seg:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images, task='seg')
            loss = criterion_seg(outputs, masks)
            loss.backward()
            optimizer.step()
            seg_running_loss += loss.item()
        train_seg_loss = seg_running_loss / len(train_loader_seg)

        # --- Classification Validation ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader_class:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task='class')
                loss = criterion_class(outputs, labels)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_loss += loss.item()
        val_acc = correct / total
        val_loss /= len(val_loader_class)

        # --- Segmentation Validation ---
        val_seg_loss = 0
        with torch.no_grad():
            for images, masks in val_loader_seg:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images, task='seg')
                loss = criterion_seg(outputs, masks)
                val_seg_loss += loss.item()
        val_seg_loss /= len(val_loader_seg)

        # Store metrics
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_seg_loss'].append(train_seg_loss)
        history['val_seg_loss'].append(val_seg_loss)

        scheduler.step()

        # Save best model
        if val_acc == max(history['val_acc']):
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"  Class | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Seg   | Train Loss: {train_seg_loss:.4f} | Val Loss: {val_seg_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


