import timm

def get_model(model_name: str, num_classes=6, pretrained=True):
    model_name = model_name.lower()
    
    if model_name == 'efficientnetv2_s':
        model = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'swin_tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'convnext_tiny':
        model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
        
    return model