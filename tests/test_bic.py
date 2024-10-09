import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.main import main
import pytest


@pytest.mark.parametrize('distill_after_bic', [0, 1])
def test_bic(distill_after_bic):
    sys.argv = ['mammoth',
                '--model',
                'bic',
                '--dataset',
                'seq-cifar10',
                '--bic_epochs',
                '5',
                '--distill_after_bic',
                str(distill_after_bic),
                '--buffer_size',
                '50',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--batch_size',
                '4',
                '--non_verbose',
                '1',
                '--num_workers',
                '0',
                '--seed',
                '0',
                '--debug_mode',
                '1']

    # log all outputs to file
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
    sys.stdout = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'test_bic.{distill_after_bic}.log'), 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    main()
