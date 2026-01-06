import torch.nn.functional as F

def calculate_losses(features_transformed, features_content, features_style, gram_style, config):
    """
    Calculates Content and Style losses.
    """
    # 1. Content Loss (Usually from relu2_2)
    content_loss = config['loss_weights']['content'] * F.mse_loss(features_transformed.relu2_2, features_content.relu2_2)

    # 2. Style Loss (Gram Matrix comparison over all 4 layers)
    style_loss = 0.
    for ft_y, gm_s in zip(features_transformed, gram_style):
        # Calculate gram matrix of the output
        from src.utils.helpers import gram_matrix
        gm_y = gram_matrix(ft_y)
        style_loss += F.mse_loss(gm_y, gm_s)
    
    style_loss *= config['loss_weights']['style']

    return content_loss, style_loss