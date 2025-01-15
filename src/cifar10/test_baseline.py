import logging
import torch


def test_model(model, dataloader, criterion, device, run_id):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():  # No need to compute gradients during testing
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate final metrics
    total_loss = running_loss / len(dataloader.dataset)
    accuracy = running_corrects.double() / len(dataloader.dataset)
    logger = logging.getLogger(run_id)
    logger.info('==================<TEST>==================')
    logger.info(f'Test Loss: {total_loss:.4f} Accuracy: {accuracy:.4f}')