import torch
import torch.nn as nn


class HammingLoss(nn.Module):
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, y_pred, y_true):
        if y_pred.size() != y_true.size():
            raise ValueError("Size of predicted and true labels must be same")

        # Calculate Hamming distance
        return (y_pred != y_true).float().sum(dim=-1).mean()


class HammingDistanceWithHistogram(nn.Module):
    def __init__(self, bins=None):
        super(HammingDistanceWithHistogram, self).__init__()
        self.bins = bins

    def forward(self, y_pred, y_true):
        if y_pred.size() != y_true.size():
            raise ValueError("Size of predicted and true labels must be same")

        # Calculate Hamming distance for each code
        distances = (y_pred != y_true).float().sum(dim=-1)

        # Compute the histogram of Hamming distances
        if self.bins is None:
            self.bins = int(distances.max().item()) + 1
        histogram = torch.histc(distances, bins=self.bins, min=0, max=self.bins)

        return distances.mean(), histogram.detach()


class HistogramWeightedBCELoss(nn.Module):
    def __init__(self):
        super(HistogramWeightedBCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.histogram = None

    def hamming_distance_histogram(self, y_pred, y_true):
        # Calculate Hamming distance for each code
        distances = (y_pred != y_true).float().sum(dim=-1)
        # Compute the histogram of Hamming distances
        # bins = int(distances.max().item()) + 1
        bins = y_pred.size(-1)
        histogram = torch.histc(distances, bins=bins, min=0, max=bins)
        return distances.mean(), histogram.detach()

    def forward(self, pred_binary_code, groundtruth_code):
        # Compute the Hamming distance and histogram
        _, histogram_new = self.hamming_distance_histogram(
            pred_binary_code, groundtruth_code
        )

        pred_binary_code = pred_binary_code.round()
        # Update the moving average of the histogram
        if self.histogram is None:
            self.histogram = histogram_new
        else:
            self.histogram = histogram_new * 0.05 + self.histogram * 0.95

        hist_soft = torch.minimum(self.histogram, 0.51 - self.histogram)
        bin_weights = torch.exp(hist_soft * 3)

        loss = self.loss_fn(pred_binary_code, groundtruth_code)
        weighted_loss = (loss * bin_weights).mean()

        return weighted_loss


# Testing the loss
# if __name__ == "__main__":
#   loss_fn = HammingLoss()
#
##    y_pred = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])
##    y_true = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 1]])
##    loss = loss_fn(y_pred, y_true)
##    print(f"Hamming Loss: {loss.item()}")
#    y_pred= torch.randn((8, 4)) # Batch of 8 samples with 4x4 shape each
#    y_true= torch.randint(0, 2, (8, 4)).float()
#    loss_fn = HistogramWeightedBCELoss()
#    loss = loss_fn(y_pred, y_true)
#    print(f"Hamming Loss: {loss.item()}")
#
#    y_pred = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
#    y_true = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]])
#    loss_fn = HammingDistanceWithHistogram()
#    average_distance, histogram = loss_fn(y_pred, y_true)
#    print(f"Average Hamming Distance: {average_distance.item()}")
#    print(f"Histogram: {histogram.numpy()}")
#    print("halt")
#    # Testing the function
