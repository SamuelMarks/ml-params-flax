""" Implementation of ml_params API """

import jax.numpy as jnp
from flax.metrics import tensorboard
from jax import random
from ml_params.base import BaseTrainer

from ml_params_flax import get_logger
from ml_params_flax.train import create_model, create_optimizer, train_epoch, eval_model

logging = get_logger('ml_params_flax')


class FlaxTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Trax """
    data = None
    K = jnp

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(FlaxTrainer, self).load_data(dataset_name=dataset_name,
                                                       data_loader=data_loader,
                                                       data_loader_kwargs=data_loader_kwargs,
                                                       data_type=data_type,
                                                       output_type=output_type,
                                                       K=jnp)
        assert self.data is not None and len(self.data) >= 2

    def train(self, epochs=10, batch_size=32, train_ds=None, test_ds=None, model_dir=None, learning_rate=0.1,
              momentum=0.9, *args,
              **kwargs):
        super(FlaxTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None and len(self.data) >= 2
        assert batch_size is not None
        assert model_dir is not None

        if train_ds is None or test_ds is None:
            train_ds, test_ds = self.data[0], self.data[1]

        rng = random.PRNGKey(0)

        summary_writer = tensorboard.SummaryWriter(model_dir)

        rng, init_rng = random.split(rng)
        model = create_model(init_rng)
        optimizer = create_optimizer(model, learning_rate, momentum)

        for epoch in range(1, epochs + 1):
            rng, input_rng = random.split(rng)
            optimizer, train_metrics = train_epoch(
                optimizer, train_ds, batch_size, epoch, input_rng)
            loss, accuracy = eval_model(optimizer.target, test_ds)
            logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                         epoch, loss, accuracy * 100)
            summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
            summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
            summary_writer.scalar('eval_loss', loss, epoch)
            summary_writer.scalar('eval_accuracy', accuracy, epoch)
        summary_writer.flush()
        return optimizer


del BaseTrainer, get_logger

__all__ = ['FlaxTrainer']
