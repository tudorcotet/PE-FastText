"""Unified experiment runner for protein embedding comparisons."""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm

from src.data import load_protein_data, TASK_INFO
from src.embedders import create_embedder

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Experiment:
    """Run a single protein prediction experiment."""
    
    def __init__(self, config: Dict[str, Any], dataset_volume: Optional[Any] = None):
        """Initialize experiment with configuration.
        
        Args:
            config: Experiment configuration with keys:
                - task: Protein task name
                - embedder: Embedder configuration
                - predictor: Predictor configuration 
                - data: Data configuration
                - output_dir: Where to save results
            dataset_volume: Modal Volume object for datasets.
        """
        self.config = config
        self.task = config['task']
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_volume = dataset_volume
        
        # Initialize components
        embedder_config = config['embedder'].copy()
        # Add model_path if not specified
        if 'model_path' not in embedder_config:
            embedder_config['model_path'] = str(self.output_dir / 'fasttext_model.bin')
        self.embedder = create_embedder(embedder_config)
        self.task_info = TASK_INFO[self.task]
        self.predictor = self._create_predictor(config['predictor'])
        
    def _create_predictor(self, config: Dict[str, Any]):
        """Create predictor model based on configuration."""
        predictor_type = config.get('type', 'rf')
        
        if predictor_type == 'rf':
            if self.task_info['type'] == 'regression':
                return RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 20),
                    n_jobs=-1
                )
            else: # classification
                base_clf = RandomForestClassifier(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 20),
                    n_jobs=-1
                )
                if self.task_info['multilabel']:
                    return MultiOutputClassifier(base_clf, n_jobs=-1)
                return base_clf

        elif predictor_type == 'linear':
            if self.task_info['type'] == 'regression':
                return Ridge(alpha=config.get('alpha', 1.0))
            else: # classification
                base_clf = LogisticRegression(
                    C=config.get('C', 1.0), 
                    max_iter=1000
                )
                if self.task_info['multilabel']:
                    return MultiOutputClassifier(base_clf, n_jobs=-1)
                return base_clf

        elif predictor_type == 'mlp':
            if self.task_info['type'] == 'regression':
                return MLPRegressor(
                    hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
                    max_iter=config.get('max_iter', 500),
                    early_stopping=True
                )
            else: # classification
                return MLPClassifier(
                    hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
                    max_iter=config.get('max_iter', 500),
                    early_stopping=True
                )

        elif predictor_type == 'xgboost':
            if self.task_info['type'] == 'regression':
                return XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=config.get('n_estimators', 100),
                    n_jobs=-1,
                    random_state=config.get('seed', 42) # XGBoost requires seed directly
                )
            else: # classification
                base_clf = XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=config.get('n_estimators', 100),
                    n_jobs=-1,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=config.get('seed', 42) # XGBoost requires seed directly
                )
                if self.task_info['multilabel']:
                    return MultiOutputClassifier(base_clf, n_jobs=-1)
                return base_clf
                
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    def _get_task_type(self) -> str:
        """Get task type (regression or classification)."""
        return self.task_info['type']
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        # Set seed for reproducibility at the start of each run
        seed = self.config.get('predictor', {}).get('seed', 42)
        set_seed(seed)

        logger.info(f"Running experiment for task: {self.task}")
        start_time = time.time()
        
        # Load data, ensuring the seed is passed for reproducible splits
        data_config = self.config.get('data', {}).copy()
        data_config['task'] = self.task
        data_config['seed'] = seed
        
        data = load_protein_data(data_config)
        
        # Extract sequences and labels
        train_sequences = data['train_sequences']
        train_labels = data['train_labels']
        
        # Only process test data if it exists
        val_sequences = data.get('validation_sequences')
        val_labels = data.get('validation_labels')

        test_splits = {}
        for split_name in ['test', 'CASP12', 'CB513', 'TS115']:
            if f'{split_name}_sequences' in data:
                test_splits[split_name] = {
                    'sequences': data[f'{split_name}_sequences'],
                    'labels': data[f'{split_name}_labels']
                }
        
        # Train FastText if needed
        if self.config['embedder']['type'] == 'fasttext':
            if hasattr(self.embedder, 'train'):
                logger.info("Training FastText model...")
                self.embedder.train(train_sequences)
        
        # --- Embedding Generation ---
        logger.info("Generating embeddings...")
        embedding_start = time.time()
        
        # Define a helper to get or create embeddings
        def get_or_create_embeddings(sequences, split_name, average_sequences=True):
            # Caching logic only for ESM2 to save on computation
            if self.config['embedder']['type'] == 'esm2':
                cache_dir = Path(f"/datasets/embeddings/{self.task}/{self.embedder.model_name.replace('/', '_')}")
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Use a hash of sequences for a more robust cache key
                import hashlib
                sequences_hash = hashlib.md5("".join(sequences).encode()).hexdigest()
                cache_file = cache_dir / f"{split_name}_{sequences_hash}.npy"

                if cache_file.exists():
                    logger.info(f"Loading cached embeddings for {split_name}...")
                    return np.load(cache_file, allow_pickle=True)
                else:
                    logger.info(f"Generating embeddings for {split_name}...")
                    embeddings = self.embedder.embed(sequences, average_sequences=average_sequences)
                    np.save(cache_file, embeddings)
                    logger.info(f"Saved embeddings for {split_name} to cache.")
                    if self.dataset_volume:
                        self.dataset_volume.commit()
                    return embeddings
            else:
                 # For other models, just generate embeddings on the fly
                return self.embedder.embed(sequences, average_sequences=average_sequences)


        # Generate embeddings for all splits
        if self.task == 'ssp':
            X_train_list = get_or_create_embeddings(train_sequences, 'train', average_sequences=False)
            X_train = np.concatenate(X_train_list, axis=0)
            
            X_test_dict = {}
            for name, split in test_splits.items():
                X_test_list = get_or_create_embeddings(split['sequences'], name, average_sequences=False)
                X_test_dict[name] = np.concatenate(X_test_list, axis=0)

            if val_sequences:
                X_val_list = get_or_create_embeddings(val_sequences, 'validation', average_sequences=False)
                X_val = np.concatenate(X_val_list, axis=0)
        else:
            X_train = get_or_create_embeddings(train_sequences, 'train')
        
            X_test_dict = {}
            for name, split in test_splits.items():
                X_test_dict[name] = get_or_create_embeddings(split['sequences'], name)

            if val_sequences:
                X_val = get_or_create_embeddings(val_sequences, 'validation')

        # --- Predictor Training & Evaluation ---
        
        # Prepare labels based on task type
        if self.task_info.get('multilabel', False) and not isinstance(train_labels, np.ndarray):
            y_train = np.array(train_labels)
            y_test_dict = {name: np.array(split['labels']) for name, split in test_splits.items()}
            y_val = np.array(val_labels) if val_sequences else None
        elif self.task == 'ssp':
            y_train = np.concatenate(train_labels, axis=0)
            y_test_dict = {name: np.concatenate(split['labels'], axis=0) for name, split in test_splits.items()}
            y_val = np.concatenate(val_labels, axis=0) if val_sequences else None
        else:
            y_train = train_labels
            y_test_dict = {name: split['labels'] for name, split in test_splits.items()}
            y_val = val_labels if val_sequences else None

        embedding_time = time.time() - embedding_start
        
        # Train predictor
        logger.info("Training predictor...")
        train_start = time.time()
        self.predictor.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Evaluate on all available test sets
        results = {'test_metrics': {}}
        for name, X_test in X_test_dict.items():
            y_test = y_test_dict[name]
            logger.info(f"Evaluating on {name} split...")
            test_results = self._evaluate(X_test, y_test)
            results['test_metrics'][name] = test_results
            
            # For backward compatibility, also store top-level metrics from the 'test' split
            if name == 'test':
                results.update(test_results)

        # Add metadata
        results.update({
            'task': self.task,
            'embedder': self.config['embedder'],
            'predictor': self.config['predictor'],
            'train_size': len(train_sequences),
            'test_size': {name: len(split['sequences']) for name, split in test_splits.items()},
            'embedding_time': embedding_time,
            'train_time': train_time,
            'total_time': time.time() - start_time
        })
        
        # Evaluate on validation if available
        if val_sequences is not None and y_val is not None:
            val_results = self._evaluate(X_val, y_val)
            results['val_metrics'] = val_results
        
        # Log results for the 'test' split primarily
        if 'test' in results.get('test_metrics', {}):
            main_results = results['test_metrics']['test']
            if self.task_info['type'] == 'regression':
                logger.info(f"Test R2: {main_results.get('r2', 'N/A'):.4f}, MSE: {main_results.get('mse', 'N/A'):.4f}")
            else:
                logger.info(f"Test Accuracy: {main_results.get('accuracy', 'N/A'):.4f}, F1: {main_results.get('f1', 'N/A'):.4f}")
        
        return results
    
    def _evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate predictor on data."""
        y_pred = self.predictor.predict(X)
        
        if self.task_info['type'] == 'regression':
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        else: # classification
            f1_average = 'samples' if self.task_info['multilabel'] else 'weighted'
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1': float(f1_score(y_true, y_pred, average=f1_average, zero_division=0.0))
            }


class ExperimentRunner:
    """Run multiple experiments and compare results."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize runner with base configuration."""
        self.base_config = base_config
        self.results = []
    
    def run_comparison(self, tasks: List[str], embedder_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run experiments comparing different embedders across tasks.
        
        Args:
            tasks: List of task names
            embedder_configs: List of embedder configurations to compare
            
        Returns:
            List of experiment results
        """
        for task in tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Task: {task}")
            logger.info(f"{'='*60}")
            
            for embedder_config in embedder_configs:
                # Create experiment config
                config = self.base_config.copy()
                config.update({
                    'task': task,
                    'embedder': embedder_config,
                    'data': {
                        'task': task,
                        **self.base_config.get('data', {})
                    }
                })
                
                # Run experiment
                try:
                    experiment = Experiment(config)
                    result = experiment.run()
                    self.results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed experiment {task} with {embedder_config}: {e}")
                    self.results.append({
                        'task': task,
                        'embedder': embedder_config,
                        'error': str(e)
                    })
        
        return self.results
    
    def summarize_results(self) -> None:
        """Print summary of all results."""
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*80)
        
        for result in self.results:
            if 'error' in result:
                logger.info(f"\n{result['task']} - {result['embedder']['type']}: FAILED")
                logger.info(f"  Error: {result['error']}")
                continue
                
            task_type = 'regression' if 'r2' in result else 'classification'
            embedder_name = self._get_embedder_name(result['embedder'])
            
            logger.info(f"\n{result['task']} - {embedder_name}:")
            if task_type == 'regression':
                logger.info(f"  R2: {result['r2']:.4f}, MSE: {result['mse']:.4f}")
            else:
                logger.info(f"  Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}")
            logger.info(f"  Embedding time: {result['embedding_time']:.1f}s")
            logger.info(f"  Training time: {result['train_time']:.1f}s")
    
    def _get_embedder_name(self, embedder_config: Dict[str, Any]) -> str:
        """Get human-readable embedder name."""
        if embedder_config['type'] == 'fasttext':
            name = 'FastText'
            if embedder_config.get('pos_encoder'):
                name += f"+{embedder_config['pos_encoder']}"
            if embedder_config.get('tokenization') == 'residue':
                name += ' (residue)'
            return name
        elif embedder_config['type'] == 'esm2':
            return 'ESM2'
        else:
            return embedder_config['type']