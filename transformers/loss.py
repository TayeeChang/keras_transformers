from transformers.backend import keras, K, tf
from transformers.layers import Loss


class BinaryDiceLoss(Loss):
    """二分类Dice Loss
    """
    def __init__(self,
                 output_dims=None,
                 alpha=0.1,
                 smooth=1,
                 square_denominator=True,
                 **kwargs):
        super(BinaryDiceLoss, self).__init__(output_dims, **kwargs)
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
            "square_denominator": self.square_denominator
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
                 square_denominator=False,
                 **kwargs):
        super(MultiClassDiceLoss, self).__init__(output_dims, **kwargs)
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
            "square_denominator": self.square_denominator
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
                 square_denominator=True,
                 **kwargs):
        super(DiceLoss, self).__init__(output_dims, **kwargs)
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
            "square_denominator": self.square_denominator
        }
        base_config = super(DiceLoss, self).get_config()
        config.update(base_config)
        return config


class FocalLoss(Loss):
    """适用于二分类情形
    Reference:
      [Focal Loss for Dense Object Detection]
      (https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
    """
    def __init__(self,
                 output_dims,
                 alpha=0.25,
                 gamma=2,
                 **kwargs):
        super(FocalLoss, self).__init__(output_dims, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, inputs, mask=None):
        if mask[1] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1], K.floatx())

        y_true, y_pred = inputs
        y_true = K.cast(y_true, 'int32')
        y_pred = tf.batch_gather(y_pred, y_true[:, :, None])[:, :, 0]
        loss = -self.alpha * (1 - y_pred) ** self.gamma * K.log(y_pred)
        loss = K.sum(loss) / K.sum(mask)
        return loss

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "gamma": self.gamma
        }
        base_config = super(FocalLoss, self).get_config()
        config.update(base_config)
        return config


custom_objects = {
    "BinaryDiceLoss": BinaryDiceLoss,
    "MultiClassDiceLoss": MultiClassDiceLoss,
    "DiceLoss": DiceLoss,
    "FocalLoss": FocalLoss,
}

keras.utils.get_custom_objects().update(custom_objects)