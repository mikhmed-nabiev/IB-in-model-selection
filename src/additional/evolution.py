import numpy as np
from copy import deepcopy
from typing import Callable
from tqdm import tqdm

from src.model.model import Model
from src.additional.memory_pool import Pool
from src.additional.utils import is_current_model_degenerate, get_best_model

class RegularizedEvolution:
  """
  Class performing regularized evolution. Inspired by https://arxiv.org/abs/1802.01548
  """

  def __init__(
    self, 
    pool: Pool,
    population_size: int, 
    subset_size: int, 
    ncycles: int,
    max_learn_length: int = None,
    max_predict_length: int = None
  ):
    """
    Initialize parameters for regularized evolution

    Args:
      mutator: Class performing mutations of models
      evaluator: Class performing evaluation of a model
      population_size: Size of the population
      subset_size: Number of candidates selected to perform tournament
      ncycles: Number of generations to consider 
    """
    
    self._history = []
    self._population = []
    self._pool = pool
    self._psize = population_size
    self._ssize = subset_size
    self._cnum = ncycles
    self._max_learn_len = max_learn_length
    self._max_predict_len = max_predict_length

  def initialize_population(
    self, 
    tr_data: np.ndarray,
    val_data: np.ndarray,
    normalize: Callable,
    loss_func: Callable
  ):
    """Initialize population with empty models."""

    while len(self._population) < self._psize:
      model = Model(
        self._pool.nscalars, 
        self._pool.nvectors, 
        self._pool.nfeatures,
        self._max_learn_len,
        self._max_predict_len
      )
      mean_loss = model.evaluate(
        pool=self._pool,
        tr_data=tr_data,
        val_data=val_data,
        normalize=normalize,
        loss=loss_func
      )
      self._population.append({"model": model, "loss": mean_loss})

  def run_evolution(
    self, 
    tr_data: np.ndarray,
    val_data: np.ndarray,
    normalize: Callable,
    loss_func: Callable
  ):
    """Run regularized evolution"""

    for _ in tqdm(range(self._cnum)):
      candidates = np.random.choice(self._population, size=self._ssize)
      best_model = get_best_model(candidates)
      child = deepcopy(best_model)

      child.mutate(self._pool)
      mean_loss = child.evaluate(
        pool=self._pool,
        tr_data=tr_data,
        val_data=val_data,
        normalize=normalize,
        loss=loss_func
      )
      while is_current_model_degenerate(self._pool):
        child.mutate(self._pool)
        mean_loss = child.evaluate(
          pool=self._pool,
          tr_data=tr_data,
          val_data=val_data,
          normalize=normalize,
          loss=loss_func
        )

      self._population.pop(0)
      self._history.append({"model": child, "loss": mean_loss})
      self._population.append({"model": child, "loss": mean_loss})

  def get_history(self):
    return deepcopy(self._history)