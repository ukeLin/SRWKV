import torch


def horizontal_forward_scan(input_tensor):
    b, c, h, w = input_tensor.shape
    return input_tensor.permute(0, 2, 3, 1).reshape(b, h * w, c)


def horizontal_forward_scan_inv(transformed_tensor, original_shape):
    b, c, h, w = original_shape
    return transformed_tensor.view(b, h, w, c).permute(0, 3, 1, 2)


def horizontal_backward_scan(input_tensor):
    b, c, h, w = input_tensor.shape
    input_tensor = torch.flip(input_tensor, dims=[-1])
    return input_tensor.permute(0, 2, 3, 1).reshape(b, h * w, c)


def horizontal_backward_scan_inv(transformed_tensor, original_shape):
    b, c, h, w = original_shape
    recovered = transformed_tensor.view(b, h, w, c).permute(0, 3, 1, 2)
    return torch.flip(recovered, dims=[-1])


def vertical_forward_scan(input_tensor):
    return input_tensor.permute(0, 1, 3, 2).flatten(2).permute(0, 2, 1)


def vertical_forward_scan_inv(transformed_tensor, original_shape):
    b, c, h, w = original_shape
    recovered = transformed_tensor.permute(0, 2, 1).view(b, c, w, h)
    return recovered.permute(0, 1, 3, 2)


def vertical_backward_scan(input_tensor):
    input_tensor = torch.flip(input_tensor, dims=[-2]).contiguous()
    return input_tensor.permute(0, 1, 3, 2).flatten(2).permute(0, 2, 1)


def vertical_backward_scan_inv(transformed_tensor, original_shape):
    b, c, h, w = original_shape
    recovered = transformed_tensor.permute(0, 2, 1).view(b, c, w, h)
    recovered = recovered.permute(0, 1, 3, 2)
    return torch.flip(recovered, dims=[-2])
