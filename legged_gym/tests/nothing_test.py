import torch

def smooth_approximation(x, a):
    # Ensure x and a are tensors
    x = torch.tensor(x)
    a = torch.tensor(a)

    # Apply the function element-wise
    f = torch.where(x < 0, torch.tensor(0.0), 
                    torch.where(x > a, torch.tensor(1.0),
                                x / a))
    return f

num_envs = 5
x = torch.tensor([0.5, 1.5, 2.0, -0.5, 1.0])  # Example tensor x
a = torch.tensor([1.0, 1.5, 1.0, 0.5, 1.0])   # Example tensor a corresponding to x

f = smooth_approximation(x, a)

print(f)
