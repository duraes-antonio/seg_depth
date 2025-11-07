from typing import Optional, Dict

import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel
from torch.nn import ELU

from model.trans_unet.vit_seg_modeling import TransUnetConfigType, CONFIGS, VisionTransformer
from options.dataset_resolution import Resolutions, shape_by_resolution
from options.model import Models, Encoders


def load_model(
        model: Models,
        resolution=Resolutions.Half,
        trans_unet_config: TransUnetConfigType = TransUnetConfigType.r50_vit_b16,
        num_classes=1
):
    is_coatnet = any(
        model.value.endswith(encoder_name.value) for encoder_name in {
            Encoders.CoatNet2_224,
            Encoders.CoatNet2_384,
            Encoders.CoatNet3_224,
            Encoders.CoatSmall224,
            Encoders.CoatLiteMedium224,
            Encoders.CoatLiteMedium384,
        })
    model_instance = get_pytorch_segmentation_models(model, num_classes, load_weight=not is_coatnet, resolution=resolution)

    if model == Models.TransUnet:
        config_vit = CONFIGS[trans_unet_config.value]
        img_size = max(shape_by_resolution[resolution])
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_patches_size = 16
        if trans_unet_config.value.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model_instance = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        model_instance.load_from(weights=np.load(config_vit.pretrained_path))

    return model_instance


def get_pytorch_segmentation_encoder(
        model: Models,
) -> str:
    smp_encoder_by_name: Dict[Encoders, str] = {
        Encoders.InceptionResNetV2: 'inceptionresnetv2',
        Encoders.VGG19BN: 'vgg19_bn',
        Encoders.Xception: 'xception',
        Encoders.MixedTransformerB2: 'mit_b2',

        Encoders.CoatSmall224: 'coat-small_224',
        Encoders.CoatLiteMedium224: 'coat-lite-medium_224',
        Encoders.CoatLiteMedium384: 'coat-lite-medium_384',
        Encoders.CoatNet2_224: 'coatnet-2_224',
        Encoders.CoatNet2_384: 'coatnet-2_384',
        Encoders.CoatNet3_224: 'coatnet-3_224',
    }
    for encoder_name in Encoders:
        if model.value.endswith(encoder_name.value):
            return smp_encoder_by_name[encoder_name]


def get_pytorch_segmentation_models(
        model: Models,
        num_classes=1,
        load_weight=True,
        resolution=Resolutions.Mini
) -> Optional[SegmentationModel]:
    sm_default_args = {
        "encoder_weights": "imagenet" if load_weight else None,
        "classes": num_classes,
        "activation": ELU
    }
    is_coatnet_3 = model in {
        Models.UNetCoatNet3_224,
        Models.UNetPlusPlusCoatNet3_224,
    }
    smp_encoder_name = get_pytorch_segmentation_encoder(model)

    # if is_coatnet_3:
    #     accepted_resolutions = {
    #         Resolutions.Square224,
    #         Resolutions.Mini,
    #     }
    #     error_message = f"{model} only supports {accepted_resolutions} size"
    #     assert resolution in accepted_resolutions, error_message

    is_coatnet_2_or_coat = model in {
        Models.UNetCoatNet2_224,
        Models.UNetCoatNet2_384,
        Models.UNetCoatSmall_224,
        Models.UNetCoatLiteMedium_224,
        Models.UNetCoatLiteMedium_384,

        Models.UNetPlusPlusCoatSmall_224,
        Models.UNetPlusPlusCoatNet2_224,
        Models.UNetPlusPlusCoatNet2_384,
        Models.UNetPlusPlusCoatLiteMedium_224,
        Models.UNetPlusPlusCoatLiteMedium_384,
    }

    # if is_coatnet_2_or_coat:
    #     accepted_resolutions = {
    #         Resolutions.Mini,
    #         Resolutions.Square224,
    #         Resolutions.Square384,
    #         Resolutions.Half2,
    #     }
    #     error_message = f"{model} only supports {accepted_resolutions} size"
    #     assert resolution in accepted_resolutions, error_message

    if model.value.startswith('unet++'):
        return smp.UnetPlusPlus(encoder_name=smp_encoder_name, **sm_default_args)

    if model.value.startswith('unet'):
        return smp.Unet(encoder_name=smp_encoder_name, **sm_default_args)

    return None
