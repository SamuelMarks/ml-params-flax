""" Implementation of ml_params API """
from sys import stdout

import jax.numpy as jnp
from flax import optim
from flax.metrics import tensorboard
from jax import random
from ml_params.base import BaseTrainer

from ml_params_flax import get_logger
from ml_params_flax.train import train_epoch, eval_model

logging = get_logger('ml_params_flax')


class FlaxTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Flax """

    rng = None  # type: None or jnp.ndarray

    def load_data(self, dataset_name, data_loader=None,
                  data_type='infer', output_type=None, K=jnp,
                  as_numpy=False, **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :param as_numpy: Convert output to numpy
        :type as_numpy: ```bool```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        data_loader_kwargs['as_numpy'] = as_numpy
        self.data = super(FlaxTrainer, self).load_data(dataset_name=dataset_name,
                                                       data_loader=data_loader,
                                                       data_type=data_type,
                                                       output_type=output_type,
                                                       K=K,
                                                       **data_loader_kwargs)
        assert self.data is not None and len(self.data) >= 2

    def load_model(self, model, call=False, rng=random.PRNGKey(0), **model_kwargs):
        """
        Load the model. Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance

        :param call: call `model()` even if `len(model_kwargs) == 0`
        :type call: ```bool```

        :param rng: random number generator
        :type rng: ```jnp.ndarray```

        :param \**model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
           to be passed into the model. If empty, doesn't call, unless call=True.

        :return self.model, e.g., the result of applying `model_kwargs` on model

        :Keyword Arguments:
            * *num_classes* (``int``) --
              Number of classes
        """
        super(FlaxTrainer, self).load_model(model=model, call=call, **model_kwargs)
        self.rng = rng

    def train(self, callbacks, epochs, loss, metrics, metric_emit_freq, optimizer,
              save_directory, output_type='infer', writer=stdout,
              batch_size=128, learning_rate=0.1, momentum=0.9,
              *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param batch_size:
        :type batch_size: ```int```

        :param learning_rate:
        :type learning_rate: ```float```

        :param momentum:
        :type momentum: ```float```

        :param args:
        :param kwargs:
        :return:
        """
        super(FlaxTrainer, self).train(callbacks=callbacks,
                                       epochs=epochs,
                                       loss=loss,
                                       metrics=metrics,
                                       metric_emit_freq=metric_emit_freq,
                                       optimizer=optimizer,
                                       save_directory=save_directory,
                                       output_type=output_type,
                                       writer=writer,
                                       *args, **kwargs)
        assert self.data is not None and len(self.data) >= 2
        assert self.model is not None
        assert self.rng is not None

        train_ds, test_ds = self.data[0], self.data[1]

        summary_writer = tensorboard.SummaryWriter(save_directory)

        if optimizer is None:
            optimizer = optim.Momentum(learning_rate=learning_rate, beta=momentum).create(self.model)

        for epoch in range(1, epochs + 1):
            rng, input_rng = random.split(self.rng)
            optimizer, train_metrics = train_epoch(
                optimizer, train_ds, batch_size, epoch, input_rng)
            loss, accuracy = eval_model(optimizer.target, test_ds)
            if metric_emit_freq(epoch):
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
