from transformers.backend import K, keras
from keras.optimizers import Optimizer
import re


class RAdam(Optimizer):
    """RAdam optimizer
    Default parameters follow those provided in the original paper.
    # References
        - [ON THE VARIANCE OF THE ADAPTIVE LEARNING RATE AND BEYOND]
          (https://arxiv.org/pdf/1908.03265.pdf)
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = kwargs.pop('lr', learning_rate)
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        super(RAdam, self).__init__(**kwargs)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        if self.initial_decay:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        t = K.cast(self.iterations, K.floatx()) + 1
        sma_inf = 2. / (1. - self.beta_2) - 1.
        sma_t = sma_inf - 2. * t * K.pow(self.beta_2, t) / (1. - K.pow(self.beta_2, t))
        r_t = K.sqrt(((sma_t - 4.) * (sma_t - 2.) * sma_inf) /
                     ((sma_inf - 4.) * (sma_inf - 2.) * sma_t))
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1. - K.pow(self.beta_2, t)))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1. - K.pow(self.beta_2, t)))
            m_corr_t = m_t / (1. - K.pow(self.beta_1, t))
            update_t = K.switch(sma_t > 4., r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)
            new_p = p - lr * update_t
            # apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(RAdam, self).get_config()
        config.update(base_config)
        return config


class Lookahead(Optimizer):
    """Lookahead optimizer
    Default parameters follow those provided in the original paper.
    # Arguments
        optimizer: optimizer identifier.
        sync_period: int. k > 0. Fast weights first look ahead every k updates.
        slow_step: float. 0< alpha < 1. Slow weights step size .
    # References
        - [Lookahead Optimizer: k steps forward, 1 step back]
          (https://arxiv.org/pdf/1907.08610.pdf)
    """
    def __init__(self, optimizer, sync_period=5, slow_step=0.5, **kwargs):
        super(Lookahead, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.optimizer = optimizer
            self.sync_period = K.variable(sync_period, dtype='int64', name='sync_period')
            self.slow_step = K.variable(slow_step, name='slow_step')

    @property
    def iterations(self):
        return self.optimizer.iterations

    @K.symbolic
    def get_updates(self, loss, params):
        sync_cond = K.equal((self.iterations + 1) % self.sync_period, 0)
        slow_params = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        fast_updates = self.optimizer.get_updates(loss, params)
        with K.control_dependencies(fast_updates):
            slow_updates = [
                K.update(slow_param,
                         K.switch(sync_cond,
                                  slow_param + self.slow_step * (param - slow_param),
                                  slow_param))
                for slow_param, param in zip(slow_params, params)]
            with K.control_dependencies(slow_updates):
                new_update = [
                    K.update(param, K.switch(sync_cond, slow_param, param))
                    for param, slow_param in zip(params, slow_params)]
        return new_update

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
            'sync_period': int(K.get_value(self.sync_period)),
            'slow_step': float(K.get_value(self.slow_step)),
        }
        base_config = super(Lookahead, self).get_config()
        config.update(base_config)
        return config


def wrap_optimizer_with_warmup(optimizer):
    class WarmupOptimizer(optimizer):
        """warmup优化器
        """
        def __init__(self, *args, **kwargs):
            self.warmup_steps = kwargs.pop('warmup_steps')
            self.total_steps = kwargs.pop('total_steps')
            self.min_lr = kwargs.pop('min_lr', 0.0)
            super(WarmupOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):
            lr = self.learning_rate
            t = K.cast(self.iterations, K.floatx())
            self.learning_rate = K.switch(
                t <= self.warmup_steps,
                lr * t / self.warmup_steps,
                self.min_lr + (lr - self.min_lr) *
                    (1.0 - (t - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            )
            updates = super(WarmupOptimizer, self).get_updates(loss, params)
            return updates

        def get_config(self):
            config = {
                'warmup_steps': self.warmup_steps,
                'total_steps': self.total_steps,
                'min_lr': self.min_lr,
            }
            base_config = super(WarmupOptimizer, self).get_config()
            config.update(base_config)
            return config
    return WarmupOptimizer


def wrap_optimizer_with_weight_decay(optimizer):
    class WeightDecayOptimizer(optimizer):
        """权重衰减优化器
        """
        def __init__(self, *args, **kwargs):
            self.weight_decay = kwargs.pop('weight_decay', 0.02)
            self.exclude_weights = kwargs.pop('exclude_weights', [])
            super(WeightDecayOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):

            def new_update(x, new_x):
                if self._do_weight_decay(x):
                    new_x = new_x - self.learning_rate * self.weight_decay * x
                return old_update(x, new_x)

            old_update = K.update
            K.update = new_update
            updates = super(WeightDecayOptimizer, self).get_updates(loss, params)
            K.update = old_update
            return updates

        def _do_weight_decay(self, x):
            return not any(re.search(p, x.name) for p in self.exclude_weights)

        def get_config(self):
            config = {
                "weight_decay": self.weight_decay
            }
            base_config = super(WeightDecayOptimizer, self).get_config()
            config.update(base_config)
            return config
    return WeightDecayOptimizer


def wrap_optimizer_with_accumulate_grads(optimizer):
    class AccumulateGradsOptimizer(optimizer):
        """梯度累积优化器
        """
        def __init__(self, *args, **kwargs):
            self.acc_grad_steps = kwargs.pop('acc_grad_steps', 2)
            super(AccumulateGradsOptimizer, self).__init__(*args, **kwargs)

        @K.symbolic
        def get_updates(self, loss, params):

            def get_gradients_acc(loss, params):
                return [g / self.acc_grad_steps for g in grads_cache]

            grads_cache = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            iterations = K.cast(self.iterations, K.floatx())
            cond = K.cast(K.equal(iterations % self.acc_grad_steps, 0), K.floatx())

            def new_update(x, new_x):
                new_x = cond * new_x + (1 - cond) * x
                return old_update(x, new_x)

            grads = self.get_gradients(loss, params)
            self.get_gradients = get_gradients_acc

            old_update = K.update
            K.update = new_update
            updates = super(AccumulateGradsOptimizer, self).get_updates(loss, params)
            K.update = old_update

            new_updates = []
            with K.control_dependencies(updates):
                for grad, grad_cache in zip(grads, grads_cache):
                    new_grad_cache = grad + (1 - cond) * grad_cache
                    new_updates.append(K.update(grad_cache, new_grad_cache))
            return new_updates

        def get_config(self):
            config = {
               'acc_grad_steps': self.acc_grad_steps
            }
            base_config = super(AccumulateGradsOptimizer, self).get_config()
            config.update(base_config)
            return config
    return AccumulateGradsOptimizer
