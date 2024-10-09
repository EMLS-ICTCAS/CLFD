import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('dataset', ['seq-cifar10', 'seq-mnist'])
@pytest.mark.parametrize('imprint_weights', [0, 1])
def test_lucir(dataset, imprint_weights):
    sys.argv = ['mammoth',
                '--model',
                'lucir',
                '--dataset',
                dataset,
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--imprint_weights',
                str(imprint_weights),
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1',
                '--fitting_epochs',
                '2']

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_lucir.{dataset}.{imprint_weights}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
