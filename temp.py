import torch


class SignalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # PyTorch STFT operation
        window = torch.hann_window(window_length=400).to(x.device)
        stft_out = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=window, return_complex=False)
        return stft_out


model = SignalModel().eval()
dummy_signal = torch.randn(1, 16000)  # Example 1 second audio snippet at 16kHz

torch.onnx.export(
    model,
    dummy_signal,
    "signal_model.onnx",
    opset_version=17,  # STFT requires newer opset standards
    input_names=['signal'],
    output_names=['spectrogram']
)