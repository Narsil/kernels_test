from hf_kernels import load_kernel
import torch


def run():
    activation = load_kernel("kernels-community/activation")
    x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

    # Run the kernel
    y = torch.empty_like(x)
    activation.gelu_fast(y, x)
    print(y)


if __name__ == "__main__":
    run()
