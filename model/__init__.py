from .SRWKV import SRWKV

def build_model(in_channels, num_classes, encoder_pretrained_path=None):
    model = SRWKV(
        in_channels=in_channels, 
        num_classes=num_classes,
        encoder_pretrained_path=encoder_pretrained_path
    )
    return model
