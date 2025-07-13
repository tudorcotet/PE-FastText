"""Unified experiment runner for protein embedding comparisons."""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from tqdm import tqdm

from src.data import load_protein_data
from src.embedders import create_embedder

logger = logging.getLogger(__name__)


class Experiment:
    """Run a single protein prediction experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment with configuration.
        
        Args:
            config: Experiment configuration with keys:
                - task: Protein task name
                - embedder: Embedder configuration
                - predictor: Predictor configuration 
                - data: Data configuration
                - output_dir: Where to save results
        """
        self.config = config
        self.task = config['task']
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = create_embedder(config['embedder'])
        self.predictor = self._create_predictor(config['predictor'])
        
    def _create_predictor(self, config: Dict[str, Any]):
        """Create predictor model based on configuration."""
        predictor_type = config.get('type', 'rf')
        task_type = self._get_task_type()
        
        if predictor_type == 'rf':
            if task_type == 'regression':
                return RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 20),
                    random_state=config.get('seed', 42),
                    n_jobs=-1
                )
            else:
                return RandomForestClassifier(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 20),
                    random_state=config.get('seed', 42),
                    n_jobs=-1
                )
        elif predictor_type == 'linear':
            if task_type == 'regression':
                return Ridge(
                    alpha=config.get('alpha', 1.0),
                    random_state=config.get('seed', 42)
                )
            else:
                return LogisticRegression(
                    C=config.get('C', 1.0),
                    random_state=config.get('seed', 42),
                    max_iter=1000
                )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    def _get_task_type(self) -> str:
        """Get task type (regression or classification)."""
        from src.data import TASK_INFO
        return TASK_INFO[self.task]['type']
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        logger.info(f"Running experiment for task: {self.task}")
        start_time = time.time()
        
        # Load data
        data_config = self.config.get('data', {}).copy()
        data_config['task'] = self.task
        data = load_protein_data(data_config)
        
        # Extract sequences and labels
        train_sequences = data['train_sequences']
        train_labels = data['train_labels']
        test_sequences = data['test_sequences']
        test_labels = data['test_labels']
        
        val_sequences = data.get('val_sequences')
        val_labels = data.get('val_labels')
        
        # Train FastText if needed
        if self.config['embedder']['type'] == 'fasttext':
            if hasattr(self.embedder, 'train'):
                logger.info("Training FastText model...")
                self.embedder.train(train_sequences)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embedding_start = time.time()
        
        X_train = self.embedder.embed(train_sequences)
        X_test = self.embedder.embed(test_sequences)
        
        if val_sequences:
            X_val = self.embedder.embed(val_sequences)
        
        embedding_time = time.time() - embedding_start
        
        # Train predictor
        logger.info("Training predictor...")
        train_start = time.time()
        self.predictor.fit(X_train, train_labels)
        train_time = time.time() - train_start
        
        # Evaluate
        logger.info("Evaluating...")
        results = self._evaluate(X_test, test_labels)
        
        # Add metadata
        results.update({
            'task': self.task,
            'embedder': self.config['embedder'],
            'predictor': self.config['predictor'],
            'train_size': len(train_sequences),
            'test_size': len(test_sequences),
            'embedding_time': embedding_time,
            'train_time': train_time,
            'total_time': time.time() - start_time
        })
        
        # Evaluate on validation if available
        if val_sequences:
            val_results = self._evaluate(X_val, val_labels)
            results['val_metrics'] = val_results
        
        # Log results
        task_type = self._get_task_type()
        if task_type == 'regression':
            logger.info(f"Test R2: {results['r2']:.4f}, MSE: {results['mse']:.4f}")
        else:
            logger.info(f"Test Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        return results
    
    def _evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate predictor on data."""
        y_pred = self.predictor.predict(X)
        
        task_type = self._get_task_type()
        if task_type == 'regression':
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        else:
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1': float(f1_score(y_true, y_pred, average='weighted'))
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