import timm
import config

def build_model(model_name=None, num_classes=None, pretrained=True):
    model_name = model_name or config.MODEL_NAME
    num_classes = num_classes or config.NUM_CLASSES

    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=config.DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE
    )
    
    return model

def freeze_backbone(model, unfreeze_last_n_stages=None):
    """Freeze the backbone, then unfreeze the final ConvNeXt stages and head."""
    unfreeze_last_n_stages = (
        config.UNFREEZE_LAST_N_STAGES
        if unfreeze_last_n_stages is None
        else unfreeze_last_n_stages
    )

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_last_n_stages > 0:
        for stage in model.stages[-unfreeze_last_n_stages:]:
            for param in stage.parameters():
                param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

    return model

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return{
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params /1e6,
        "trainable_params_m": trainable_params /1e6
    }
