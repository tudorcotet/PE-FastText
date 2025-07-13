"""Unified data loading for protein tasks."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


TASK_INFO = {
    "fluorescence": {
        "type": "regression",
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "fluorescence"
    },
    "stability": {
        "type": "regression", 
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "stability"
    },
    "mpp": {
        "type": "regression",
        "dataset": "InstaDeepAI/true-cds-protein-tasks", 
        "name": "mpp"
    },
    "beta_lactamase_complete": {
        "type": "regression",
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "beta_lactamase_complete"
    },
    "ssp": {
        "type": "classification",
        "dataset": "InstaDeepAI/true-cds-protein-tasks",
        "name": "ssp",
        "num_classes": 8
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
    train_size = config.get('train_size', 1.0)
    val_split = config.get('val_split', 0.1)
    max_length = config.get('max_length')
    seed = config.get('seed', 42)
    
    logger.info(f"Loading {task} dataset...")
    
    # Load dataset
    dataset = load_dataset(task_info['dataset'], name=task_info['name'])
    
    result = {'task_info': task_info}
    
    # Process train split
    if 'train' in dataset:
        train_data = dataset['train']
        sequences = train_data['sequence']
        labels = np.array(train_data['label'])
        
        # Subsample if requested
        if train_size < 1.0:
            n_samples = int(len(sequences) * train_size)
            indices = np.random.RandomState(seed).permutation(len(sequences))[:n_samples]
            sequences = [sequences[i] for i in indices]
            labels = labels[indices]
        
        # Apply max length filter
        if max_length:
            filtered = [(s, l) for s, l in zip(sequences, labels) if len(s) <= max_length]
            if filtered:
                sequences, labels = zip(*filtered)
                sequences = list(sequences)
                labels = np.array(labels)
        
        # Create validation split
        if val_split > 0:
            train_seq, val_seq, train_labels, val_labels = train_test_split(
                sequences, labels, test_size=val_split, random_state=seed
            )
            result['train_sequences'] = train_seq
            result['train_labels'] = train_labels
            result['val_sequences'] = val_seq
            result['val_labels'] = val_labels
        else:
            result['train_sequences'] = sequences
            result['train_labels'] = labels
    
    # Process test split
    if 'test' in dataset:
        test_data = dataset['test']
        test_sequences = test_data['sequence']
        test_labels = np.array(test_data['label'])
        
        # Apply max length filter
        if max_length:
            filtered = [(s, l) for s, l in zip(test_sequences, test_labels) 
                       if len(s) <= max_length]
            if filtered:
                test_sequences, test_labels = zip(*filtered)
                test_sequences = list(test_sequences)
                test_labels = np.array(test_labels)
        
        result['test_sequences'] = test_sequences
        result['test_labels'] = test_labels
    
    # Log statistics
    logger.info(f"Loaded {task} data:")
    for split in ['train', 'val', 'test']:
        if f'{split}_sequences' in result:
            logger.info(f"  {split}: {len(result[f'{split}_sequences'])} sequences")
    
    return result