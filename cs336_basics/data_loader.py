import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    start_indices = torch.randint(
        low=0, high=len(dataset) - context_length, size=(batch_size,)
    )

    # For each starting index, create a sequence of length context_length for the input and labels.
    input_sequences = []
    label_sequences = []
    for start_idx in start_indices:
        input_seq = dataset[start_idx : start_idx + context_length]
        label_seq = dataset[start_idx + 1 : start_idx + context_length + 1]
        input_sequences.append(torch.LongTensor(input_seq))
        label_sequences.append(torch.LongTensor(label_seq))

    device = torch.device(device)

    input_tensor = torch.stack(input_sequences).to(device)
    label_tensor = torch.stack(label_sequences).to(device)

    return input_tensor, label_tensor
