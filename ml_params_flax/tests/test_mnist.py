from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from ml_params_flax.ml_params_impl import FlaxTrainer


class TestMnist(TestCase):
    tfds_dir = None
    model_dir = None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.tfds_dir = path.join(path.expanduser('~'), 'tensorflow_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.tfds_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        trainer = FlaxTrainer()
        trainer.load_data('mnist', tfds_dir=TestMnist.tfds_dir)
        trainer.load_model('TODO')
        trainer.train(epochs=3, model_dir=TestMnist.model_dir, loss=None, callbacks=None, optimizer=None,
                      metrics=None, metric_emit_freq=lambda epoch: True, save_directory=TestMnist.model_dir)


if __name__ == '__main__':
    unittest_main()
