import matplotlib.pyplot as plt
def plot_loss_vs_epochs(train_losses, val_losses):
    assert(len(train_losses) == len(val_losses))
    plt.grid()
    plt.plot(list(range(len(train_losses))), train_losses, "o-", label="Train")
    plt.plot(list(range(len(train_losses))), val_losses, "o-", label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()