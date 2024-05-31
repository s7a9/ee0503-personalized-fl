import io
import torch


def model_to_bytes(model) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()

def bytes_to_model(bytes_: bytes, model: torch.nn.Module) -> torch.nn.Module:
    buffer = io.BytesIO(bytes_)
    model.load_state_dict(torch.load(buffer))
    return model
