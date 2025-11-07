import ssl
from enum import Enum

ssl._create_default_https_context = ssl._create_unverified_context


class Encoders(Enum):
    InceptionResNetV2 = 'inception-resnet-v2'
    VGG19BN = 'vgg-19-bn'
    Xception = 'xception'
    MixedTransformerB2 = 'mt-b2'
    MixedTransformerB3 = 'mt-b3'
    MixedTransformerB4 = 'mt-b4'

    CoatSmall224 = 'coat-small_224'
    CoatLiteMedium224 = 'coat-lite-medium_224'
    CoatLiteMedium384 = 'coat-lite-medium_384'
    CoatNet2_224 = 'coatnet-2_224'
    CoatNet2_384 = 'coatnet-2_384'
    CoatNet3_224 = 'coatnet-3_224'

    def __str__(self):
        return self.value


class Models(Enum):
    TransUnet = 'trans-unet'

    UNetPlusPlusInceptionResNetv2 = f'unet++_{Encoders.InceptionResNetV2}'
    UNetPlusPlusVGG19BN = f'unet++_{Encoders.VGG19BN}'
    UNetPlusPlusXception = f'unet++_{Encoders.Xception}'

    UNetPlusPlusCoatSmall_224 = f'unet++_{Encoders.CoatSmall224}'
    UNetPlusPlusCoatLiteMedium_224 = f'unet++_{Encoders.CoatLiteMedium224}'
    UNetPlusPlusCoatLiteMedium_384 = f'unet++_{Encoders.CoatLiteMedium384}'

    UNetPlusPlusCoatNet2_224 = f'unet++_{Encoders.CoatNet2_224}'
    UNetPlusPlusCoatNet2_384 = f'unet++_{Encoders.CoatNet2_384}'
    UNetPlusPlusCoatNet3_224 = f'unet++_{Encoders.CoatNet3_224}'

    UNetInceptionResNetv2 = f'unet_{Encoders.InceptionResNetV2}'
    UNetVGG19BN = f'unet_{Encoders.VGG19BN}'
    UNetXception = f'unet_{Encoders.Xception}'
    UNetMixedTransformerB2 = f'unet_{Encoders.MixedTransformerB2}'
    UNetMixedTransformerB3 = f'unet_{Encoders.MixedTransformerB3}'
    UNetMixedTransformerB4 = f'unet_{Encoders.MixedTransformerB4}'

    UNetCoatSmall_224 = f'unet_{Encoders.CoatSmall224}'
    UNetCoatLiteMedium_224 = f'unet_{Encoders.CoatLiteMedium224}'
    UNetCoatLiteMedium_384 = f'unet_{Encoders.CoatLiteMedium384}'

    UNetCoatNet2_224 = f'unet_{Encoders.CoatNet2_224}'
    UNetCoatNet2_384 = f'unet_{Encoders.CoatNet2_384}'
    UNetCoatNet3_224 = f'unet_{Encoders.CoatNet3_224}'

    def __str__(self):
        return self.value
