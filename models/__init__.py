from .cnn6 import CNN6
from .resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101
from .efficientnet import EfficientNet_B0, EfficientNet_B1, EfficientNet_B2
from .ast import ASTModel
from .ast_freq import ASTModel as ASTModelFBE  # AST + 病理频段 FBE (4 band)
from .ast_freq_uniform import ASTModel as ASTModelUniform
from .ast_cnn_dual import ASTModelDualCNN  # AST + 等分频段 FBE (12 band)
from .ssast import SSASTModel
from .projector import Projector

_backbone_class_map = {
    'cnn6': CNN6,
    'resnet10': ResNet10,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'efficientnet_b0': EfficientNet_B0,
    'efficientnet_b1': EfficientNet_B1,
    'efficientnet_b2': EfficientNet_B2,
    'ast': ASTModel,
    'ast_fbe': ASTModelFBE,  # AST + 病理频段 FBE (4 band)
    'ast_fbe_patho': ASTModelFBE,  # 别名，更明确
    'ast_fbe_uniform': ASTModelUniform,  # AST + 等分频段 FBE (12 band)
    'ast_cnn_dual': ASTModelDualCNN,  # AST + CNN dual channel
    'ssast': SSASTModel
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))