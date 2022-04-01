from keras2bert.backend import K
from keras2bert.layers import Loss


class BinaryDiceLoss(Loss):
    """二分类Dice Loss
    """
    def __init__(self,
                 output_dims=None,
                 alpha=0.1,
                 smooth=1,
                 square_denominator=True):
        super(BinaryDiceLoss, self).__init__(output_dims)
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    def compute_loss(self, inputs, mask=None):
        if mask[1] is None:
            mask = 1.0
        else:
            mask = mask[1]
            mask = K.cast(mask, K.floatx())

        y_true, y_pred = inputs
        y_true = y_true * mask
        y_pred = y_pred * mask

        y_pred = ((1 - y_pred) ** self.alpha) * y_pred
        intersection = K.sum(y_pred * y_true, axis=1)

        if not self.square_denominator:
            dice_eff = ((2 * intersection + self.smooth) /
                        (K.sum(y_pred, axis=1) + K.sum(y_true, axis=1) + self.smooth))
        else:
            dice_eff = ((2 * intersection + self.smooth) /
                        (K.sum(K.square(y_pred), axis=1) + K.sum(K.square(y_true), axis=1) + self.smooth))

        return 1 - K.mean(dice_eff)
    
    def get_config(self):
        config = {
            "alpha": self.alpha,
            "smooth": self.smooth,
            "square_denominator": self.denominator
        }
        base_config = super(BinaryDiceLoss, self).get_config()
        config.update(base_config)
        return config


class MultiClassDiceLoss(Loss):
    """多分类Dice Loss
    """
    def __init__(self,
                 output_dims=None,
                 alpha=0.1,
                 smooth=1,
                 square_denominator=False):
        super(MultiClassDiceLoss, self).__init__(output_dims)
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_true = K.cast(y_true, 'int32')
        y_true = K.one_hot(y_true, K.int_shape(y_pred)[-1])
        assert K.int_shape(y_true) == K.int_shape(y_pred), "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss(alpha=self.alpha, smooth=self.smooth)
        total_loss = 0
        N = K.int_shape(y_pred)[-1]
        for i in range(N):
            total_loss += binaryDiceLoss.compute_loss([y_true[:, :, i], y_pred[:, :, i]], mask)
        return total_loss / N
    
    def get_config(self):
        config = {
            "alpha": self.alpha,
            "smooth": self.smooth,
            "square_denominator": self.denominator
        }
        base_config = super(MultiClassDiceLoss, self).get_config()
        config.update(base_config)
        return config


class DiceLoss(Loss):
    """Data-Imbalanced Dice Loss
    Reference：
      [Dice Loss for Data-imbalanced NLP Tasks]
      (https://arxiv.org/pdf/1911.02855.pdf?ref=https://githubhelp.com)
    """
    def __init__(self,
                 output_dims=None,
                 alpha=0.1,
                 smooth=1,
                 square_denominator=True):
        super(DiceLoss, self).__init__(output_dims)
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    def compute_loss(self, inputs, mask=None):
        if mask[1] is None:
            mask = 1.0
        else:
            mask = mask[1][:, :, None]
            mask = K.cast(mask, K.floatx())

        y_true, y_pred = inputs
        y_true = K.cast(y_true, 'int32')
        y_true = K.one_hot(y_true, K.int_shape(y_pred)[-1])
        assert K.int_shape(y_true) == K.int_shape(y_pred), "predict & target shape do not match"

        y_true = y_true * mask
        y_pred = y_pred * mask

        y_pred = ((1 - y_pred) ** self.alpha) * y_pred
        intersection = K.sum(y_pred * y_true, axis=1)

        if not self.square_denominator:
            dice_eff = ((2 * intersection + self.smooth) /
                        (K.sum(y_pred, axis=1) + K.sum(y_true, axis=1) + self.smooth))
        else:
            dice_eff = ((2 * intersection + self.smooth) /
                        (K.sum(K.square(y_pred), axis=1) + K.sum(K.square(y_true), axis=1) + self.smooth))

        return 1 - K.mean(dice_eff)
    
    def get_config(self):
        config = {
            "alpha": self.alpha,
            "smooth": self.smooth,
            "square_denominator": self.denominator
        }
        base_config = super(DiceLoss, self).get_config()
        config.update(base_config)
        return config
