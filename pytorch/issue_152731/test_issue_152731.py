import torch
import pytest

def test_addition_cpu_overflow():
    device = torch.device("cpu")
    tensor_shape = (8, 1, 1, 1)
    tensor_dtype = torch.float16
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype, device=device) * 1e-5
    tensor2 = torch.randn(tensor_shape, dtype=tensor_dtype, device=device) * 1e-5
    alpha = 1.04119e+32
    f16_max = 65504.0

    print(f"CPU test: Alpha={alpha}, Max float16={f16_max}")

    # Expecting an overflow error
    with pytest.raises(RuntimeError):
        torch.add(tensor1, tensor2, alpha=alpha)
        
def test_addition_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    tensor_shape = (8, 1, 1, 1)
    tensor_dtype = torch.float16
    tensor1 = torch.randn(tensor_shape, dtype=tensor_dtype, device=device) * 1e-5
    tensor2 = torch.randn(tensor_shape, dtype=tensor_dtype, device=device) * 1e-5
    alpha = 1.04119e+32

    result = torch.add(tensor1, tensor2, alpha=alpha)
    assert result.shape == tensor1.shape
    print("CUDA addition successful!")
