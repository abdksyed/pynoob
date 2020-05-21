import torch
import torch.nn as nn
import torch.nn.functional as F

test_loss = []

test_acc = []
criterion = nn.CrossEntropyLoss()

max_val = 0

def test(model, device, test_loader):
    model.eval()
    correct = 0
    tloss = 0
    global max_val

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            tloss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    tloss /= len(test_loader.dataset)
    test_loss.append(tloss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          tloss, correct, len(test_loader.dataset),
          100 * correct/len(test_loader.dataset)))

    test_acc.append(100 * correct/len(test_loader.dataset))

    if test_acc[-1] > max_val:
        max_val = test_acc[-1]
        path = '/classifier.pt'
        torch.save(model.state_dict(), path)