import matplotlib.pyplot as plt


def plot(scores):
    # TODO: Update to proper plotting
    x = scores[:, 0]
    y = scores[:, 1]
    plt.xlabel('Accuracy objective')
    plt.ylabel('Fairness objective')

    plt.scatter(x, y)
    plt.show()
