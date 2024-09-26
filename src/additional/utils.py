import numpy as np
from typing import List, Callable

from src.model.model import Model
from src.additional.memory_pool import Pool


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def abs_loss(true, predicted):
  return abs(predicted - true)

def is_current_model_degenerate(pool: Pool) -> bool:
  """
  Checks pool to see whether calculations with current model(setup, predict, learn) 
  produce degenerate values
  """

  scalars = pool.get_scalars()
  vectors = pool.get_vectors().flatten()
  values = np.concatenate((scalars, vectors))
  for elem in values:
    if elem is np.nan or elem is np.inf or elem is -np.inf:
      return True 
  return False

def get_best_model(models: List[Model]) -> Model:
  best = models[0]
  for elem in models:
    if elem["loss"] < best["loss"]:
      best = elem
  return best["model"]

def get_prediction(model: Model, pool: Pool, dvalid: np.ndarray, normalize: Callable):
    pool.set_zeros()
    model.setup(pool)
    preds = []

    for row in dvalid:
        x = row[:-1]
        y = row[-1]

        pool.set_vector(index=0, value=x)
        model.predict(pool)
        s1 = pool.get_scalar(index=1)
        s1 = normalize(s1)
        preds.append((s1 >= 0.5))

    return preds    