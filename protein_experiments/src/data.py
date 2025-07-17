"""Unified data loading for protein tasks."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging
from Bio.Seq import Seq

logger = logging.getLogger(__name__)


TASK_INFO = {
    "fluorescence": {
        "type": "regression",
        "multilabel": False,
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "fluorescence",
        "needs_translation": True,
        "sequence_col": "sequence",
        "label_col": "label",
    },
    "stability": {
        "type": "regression",
        "multilabel": False,
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "stability",
        "needs_translation": True,
        "sequence_col": "sequence",
        "label_col": "label",
    },
    "mpp": {
        "type": "regression",
        "multilabel": False,
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "melting_point",
        "needs_translation": True,
        "sequence_col": "sequence",
        "label_col": "label",
    },
    "beta_lactamase_complete": {
        "type": "regression",
        "multilabel": False,
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "beta_lactamase_complete",
        "needs_translation": True,
        "sequence_col": "sequence",
        "label_col": "label",
    },
    "ssp": {
        "type": "classification",
        "multilabel": False,
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "ssp",
        "num_classes": 8,
        "needs_translation": True,
        "sequence_col": "sequence",
        "label_col": "label",
    },
    "deeploc": {
        "type": "classification",
        "multilabel": True,
        "dataset": "bloyal/deeploc",
        "name": "default",
        "num_classes": 11,
        "needs_translation": False,
        "sequence_col": "Sequence",
        "label_col": [
            "Membrane", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
            "Mitochondrion", "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole",
            "Golgi apparatus", "Peroxisome"
        ],
    }
}


def load_protein_data(config: Dict) -> Dict[str, any]:
    """Load protein task data.
    
    Args:
        config: Data configuration with keys:
            - task: Task name (fluorescence, stability, etc.)
            - train_size: Fraction of training data to use (default: 1.0)
            - val_split: Fraction of train to use for validation (default: 0.1)
            - max_length: Maximum sequence length (optional)
            - seed: Random seed (default: 42)
    
    Returns:
        Dictionary with:
            - train_sequences: List of training sequences
            - train_labels: Training labels
            - val_sequences: Validation sequences (if val_split > 0)
            - val_labels: Validation labels
            - test_sequences: Test sequences
            - test_labels: Test labels
            - task_info: Task metadata
    """
    task = config['task']
    if task not in TASK_INFO:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_INFO.keys())}")
    
    task_info = TASK_INFO[task]
    dataset_name = task_info['dataset']
    dataset_conf = task_info.get('name')
    needs_translation = task_info.get('needs_translation', True)
    sequence_col = task_info.get('sequence_col', 'sequence')
    label_col = task_info.get('label_col', 'label')

    train_size = config.get('train_size', 1.0)
    val_split = config.get('val_split', 0.1)
    max_length = config.get('max_length')
    seed = config.get('seed', 42)
    
    logger.info(f"Loading {task} dataset from {dataset_name}...")
    
    dataset = load_dataset(
        dataset_name, 
        name=dataset_conf if dataset_conf != 'default' else None,
        trust_remote_code=True
    )
    
    result = {'task_info': task_info}
    
    # Process train split
    if 'train' in dataset:
        train_data = dataset['train']
        sequences = train_data[sequence_col]
        
        if isinstance(label_col, list):
            labels = train_data.select_columns(label_col).to_pandas().values.astype(np.float32)
        else:
            labels = train_data[label_col]

        # Convert to numpy array for consistent handling
        if not isinstance(labels, np.ndarray):
            if task_info.get('multilabel', False) or (task_info.get('type') == 'classification' and task == 'ssp'):
                 labels = np.array(labels, dtype=object)
            else:
                labels = np.array(labels)

        if needs_translation:
            logger.info(f"Translating {len(sequences)} DNA sequences to protein...")
            sequences = [str(Seq(seq).translate(to_stop=True)) for seq in sequences]
        
        # Subsample if requested
        if train_size < 1.0:
            n_samples = int(len(sequences) * train_size)
            indices = np.random.permutation(len(sequences))[:n_samples]
            sequences = [sequences[i] for i in indices]
            labels = labels[indices]
        
        # Use existing validation split or create one
        if 'validation' in dataset:
            logger.info("Using existing validation split.")
            result['train_sequences'] = sequences
            result['train_labels'] = labels
            val_data = dataset['validation']
            result['validation_sequences'] = val_data[sequence_col]
            
            # Correctly handle multi-column labels for validation set
            if isinstance(label_col, list):
                result['validation_labels'] = val_data.select_columns(label_col).to_pandas().values.astype(np.float32)
            else:
                result['validation_labels'] = np.array(val_data[label_col])

            if needs_translation:
                result['validation_sequences'] = [str(Seq(s).translate(to_stop=True)) for s in result['validation_sequences']]

        elif val_split > 0:
            logger.info(f"Creating validation split ({val_split*100}% of training data).")
            train_seqs, val_seqs, train_labels, val_labels = train_test_split(
                sequences, labels, test_size=val_split, random_state=config.get("seed")
            )
            result['train_sequences'] = train_seqs
            result['train_labels'] = train_labels
            result['validation_sequences'] = val_seqs
            result['validation_labels'] = val_labels
        else:
            result['train_sequences'] = sequences
            result['train_labels'] = labels

    # Process other splits (test, etc.)
    for split_name in ['test', 'CASP12', 'CB513', 'TS115']:
        if split_name in dataset:
            split_data = dataset[split_name]
            sequences = split_data[sequence_col]

            if isinstance(label_col, list):
                labels = split_data.select_columns(label_col).to_pandas().values.astype(np.float32)
            else:
                labels = split_data[label_col]

            if not isinstance(labels, np.ndarray):
                if task_info.get('multilabel', False) or (task_info.get('type') == 'classification' and task == 'ssp'):
                    labels = np.array(labels, dtype=object)
                else:
                    labels = np.array(labels)
            
            if needs_translation:
                logger.info(f"Translating {len(sequences)} {split_name} DNA sequences to protein...")
                sequences = [str(Seq(seq).translate(to_stop=True)) for seq in sequences]

            result[f'{split_name}_sequences'] = sequences
            result[f'{split_name}_labels'] = labels

    # Apply max length filter to all splits
    if max_length:
        for split in ['train', 'validation', 'test', 'CASP12', 'CB513', 'TS115']:
            if f'{split}_sequences' in result:
                seqs = result[f'{split}_sequences']
                lbls = result[f'{split}_labels']
                filtered = [(s, l) for s, l in zip(seqs, lbls) if len(s) <= max_length]
                if filtered:
                    seqs_f, lbls_f = zip(*filtered)
                    result[f'{split}_sequences'] = list(seqs_f)
                    # Preserve correct label type
                    if task_info.get('multilabel', False) or (task_info.get('type') == 'classification' and task == 'ssp'):
                         result[f'{split}_labels'] = np.array(lbls_f, dtype=object)
                    else:
                        result[f'{split}_labels'] = np.array(lbls_f)


    # Log statistics
    logger.info(f"Loaded {task} data:")
    for split in ['train', 'validation', 'test', 'CASP12', 'CB513', 'TS115']:
        if f'{split}_sequences' in result:
            logger.info(f"  {split}: {len(result[f'{split}_sequences'])} sequences")
    
    return result