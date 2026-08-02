def build_model(in_channels, num_classes, img_size=224, encoder_pretrained_path=None):
    from .SRWKV import SRWKV

    return SRWKV(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        encoder_pretrained_path=encoder_pretrained_path,
    )
