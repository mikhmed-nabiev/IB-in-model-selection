import numpy as np
from typing import Callable, List, Tuple
from copy import deepcopy
from abc import ABC, abstractmethod

from src.additional.memory_pool import Pool
from src.model.operations import *
from enum import Enum


class Function(ABC):

  @abstractmethod
  def eval_func(self):
    pass
    

class Setup(Function):
  """
  Class Setup initializes weights. 
  """

  def __init__(
    self, 
    nscalars, 
    nvectors, 
    nfeatures,
    # nmatrices, # TODO: add matrices
  ):
    self._nscalars = nscalars
    self._nvectors = nvectors
    self._nfeatures = nfeatures

    self._scalars = np.zeros(self._nscalars)
    self._vectors = np.zeros(shape=(self._nvectors, self._nfeatures))

  def _change_weights(self):
    # print(self._scalars)
    for i in range(len(self._scalars)):
      self._scalars = np.random.uniform(low=-10, high=10, size=self._nscalars) # maybe should add low and high as parameters

    for i in range(len(self._vectors)):
      self._vectors[i] = np.random.uniform(low=-10, high=10, size=self._nfeatures) # maybe should add low and high as parameters

  def mutate(self, pool: Pool):
    """Change all weights with the probability of 0.3"""
    if np.random.binomial(1, p=0.3):
      self._change_weights()

  def set_weights(self, pool: Pool):
    self._scalars = deepcopy(pool.get_scalars())
    self._vectors = deepcopy(pool.get_vectors())

  def __repr__(self) -> str:
    return f"""
    \tscalars={self._scalars}, 
    \tvectors={self._vectors.tolist()}
    """

  def eval_func(self, pool: Pool):
    """Sets up the weigths"""
    pool.set_scalars(self._scalars)
    pool.set_vectors(self._vectors)
    # TOOD add matrices


class Learn(Function):
  """
  Class Learn learns weights from data.
  """

  def __init__(self, max_length: int):
    self._functions = []
    self._args = []
    self._max_len = max_length
    if self._max_len is None:
      self._max_len = np.inf


  def _add_or_remove_func(self, pool: Pool):
    nscalars = pool.nscalars
    nvectors = pool.nvectors
    type = np.random.choice(["add", "remove"], p=[0.7, 0.3])

    if type == "add" and len(self._functions) < self._max_len or len(self._functions) == 0:
      func_idx = np.random.choice(np.arange(N_OPS))
      self._functions.append(func_idx)
      
      arg_type = np.random.choice([SCALAR, VECTOR])
      if OPS_ARGS[func_idx]["arg_types"] == SCALAR:
        arg_type = SCALAR
      elif OPS_ARGS[func_idx]["arg_types"] == VECTOR:
        arg_type = VECTOR
    
      arg_indices = np.random.choice(
        np.arange(nscalars if arg_type == SCALAR else nvectors), 
        size=OPS_ARGS[func_idx]["n_args"]
      )
      
      return_type = OPS_ARGS[func_idx]["return_type"]
      if return_type == ANY:
        return_type = arg_type

      return_idx = np.random.choice(
        np.arange(nscalars if return_type == SCALAR else nvectors), 
        size=1 # for convenience 
      )

      self._args.append(dict(
        arg_indices=arg_indices,
        arg_type=arg_type,
        return_idx=return_idx,
        return_type=return_type
      ))

    elif type == "remove":
      func_idx = np.random.choice(np.arange(len(self._functions)))
      del self._functions[func_idx]
      del self._args[func_idx]

  def _randomize_component_function(self, pool: Pool):
    if len(self._functions) == 0:
      return
    
    func_idx = np.random.choice(np.arange(N_OPS))
    change_idx = np.random.choice(np.arange(len(self._functions)))

    arg_type = np.random.choice([SCALAR, VECTOR])
    if OPS_ARGS[func_idx]["arg_types"] == SCALAR:
      arg_type = SCALAR
    elif OPS_ARGS[func_idx]["arg_types"] == VECTOR:
      arg_type = VECTOR

    return_type = OPS_ARGS[func_idx]["return_type"]
    if return_type == ANY:
      return_type = arg_type 

    self._functions[change_idx] = func_idx 
    self._args[change_idx] = dict(
      arg_indices=np.random.choice(
        np.arange(pool.nscalars if arg_type == SCALAR else pool.nvectors), 
        size=OPS_ARGS[func_idx]["n_args"]
      ),
      arg_type=arg_type,
      return_idx=np.random.choice(
        np.arange(pool.nscalars if return_type == SCALAR else pool.nvectors), 
        size=1 
      ),
      return_type=return_type
    )

  def _change_arguments(self, pool: Pool):
    # randomly select a function and randomly pick different argument 
    if len(self._functions) == 0:
      return 
    change_idx = np.random.choice(np.arange(len(self._functions)))
    self._args[change_idx]["arg_indices"] = np.random.choice(
      np.arange(pool.nscalars if self._args[change_idx]["arg_type"] == SCALAR else pool.nvectors), 
      size=OPS_ARGS[self._functions[change_idx]]["n_args"]
    )
    self._args[change_idx]["return_idx"] = np.random.choice(
      np.arange(pool.nscalars if self._args[change_idx]["return_type"] == SCALAR else pool.nvectors), 
      size=1 
    )

  def mutate(self, pool: Pool):
    func = np.random.choice(
      [
        self._add_or_remove_func,
        self._randomize_component_function,
        self._change_arguments
      ]
    )
    func(pool)
  
  def eval_func(self, pool: Pool):
    """Learns weights"""

    for func_index, args in zip(self._functions, self._args):
      arg_type = args["arg_type"]
      return_type = args["return_type"]
      arg_indices = args["arg_indices"]
      return_idx = args["return_idx"]
      
      # compute function
      result = OPS[func_index](
        *(pool.get_scalars(arg_indices) if arg_type == SCALAR else pool.get_vectors(arg_indices))  
        )
          
      # saving result to memory
      if return_type == SCALAR:
          pool.set_scalar(index=return_idx, value=result)
      elif return_type == VECTOR:
          pool.set_vector(index=return_idx, value=result)

  def __repr__(self) -> str:
    output = ""
    for func_idx, func_arg in zip(self._functions, self._args):
      output += OPS_NAMES[func_idx]

      # function arguments
      if OPS_ARGS[func_idx]["n_args"] == 1:
        if func_arg["arg_type"] == SCALAR:
          output += f"(s{func_arg['arg_indices'].item()})"
        elif func_arg["arg_type"] == VECTOR:
          output += f"(v{func_arg['arg_indices'].item()})"
      elif OPS_ARGS[func_idx]["n_args"] == 2:
        if func_arg["arg_type"] == SCALAR:
          output += f"(s{func_arg['arg_indices'][0]}, s{func_arg['arg_indices'][1]})"
        elif func_arg["arg_type"] == VECTOR:
          output += f"(v{func_arg['arg_indices'][0]}, v{func_arg['arg_indices'][1]})"

      # result address
      if func_arg["return_type"] == SCALAR:
        output += f"=s{func_arg['return_idx'].item()}"
      elif func_arg["return_type"] == VECTOR:
        output += f"=v{func_arg['return_idx'].item()}"

      output += "\n\t\t\t\t"
    return output
        


class Predict(Learn):
  """
  Class predicts from input which is stored in address v0
  """

  def __init_(self):
    super().__init__(self)


class Model:
  """
  Class "Model" which trains and validates triple (setup, learn, predict) on
  train and val data. Its only public method is "evaluate".
  """

  def __init__(
    self, 
    nscalars: int, 
    nvectors: int, 
    nfeatures: int,
    max_learn_len: int,
    max_predict_len: int
  ):
    """
    Initialize the triple. All functions initially are empty.
    """

    self._setup = Setup(nscalars, nvectors, nfeatures)
    self._learn = Learn(max_learn_len)
    self._predict = Predict(max_predict_len)

  def setup(self, pool: Pool):
    self._setup.eval_func(pool)

  def learn(self, pool: Pool):
    self._learn.eval_func(pool)

  def predict(self, pool: Pool):
    self._predict.eval_func(pool)

  def _validate(
    self,
    pool: Pool,
    val_data: np.ndarray,
    normalize: Callable,
    loss: Callable
  ):
    sum_loss = 0.0
    for row in val_data:
      x = row[:-1]
      y = row[-1]
      
      pool.set_vector(index=0, value=x) # object is always put in address v0
      self.predict(pool)
      s1 = pool.get_scalar(index=1) # predict always lies in address s1
      s1 = normalize(s1)
    
      sum_loss += loss(y, s1)
      
    mean_loss = sum_loss / len(val_data)
    return mean_loss

  def mutate(self, pool: Pool):
    self._setup.mutate(pool)
    self._learn.mutate(pool)
    self._predict.mutate(pool)

  def _train(
    self, 
    pool: Pool,
    tr_data: np.ndarray,
    normalize: Callable
  ):
    pool.set_zeros()
    self.setup(pool)
    for row in tr_data:
      x = row[:-1]
      y = row[-1]
        
      pool.set_vector(index=0, value=x) # object is always put in address v0
      self.predict(pool)
        
      s1 = pool.get_scalar(index=1) # predict always lies in address s1
      s1 = normalize(s1)
        
      pool.set_scalar(index=1, value=s1)
      pool.set_scalar(index=0, value=y) # y is save in address s0
      self.learn(pool)
      self._setup.set_weights(pool)

  def evaluate(
    self,
    pool: Pool,
    # tr_data: np.ndarray,
    # val_data: np.ndarray,
    train_datasets: List[np.ndarray],
    valid_datasets: List[np.ndarray],
    normalize: Callable,
    loss: Callable
  ):
    """
    Evaluate model on one task.

    Args:
      pool: Pool of available variables to use
      tr_data: Training data
      val_data: Validation data
      normalize: Normalization function
      loss: Loss function
    """
    for train in train_datasets:
      self._train(pool, train, normalize)

    val_accuracy = 0
    for valid in valid_datasets:
      val_accuracy += self._validate(pool, valid, normalize, loss)

      val_accuracy  = val_accuracy / len(valid_datasets)

    return val_accuracy
  
  def __repr__(self) -> str:
    return f"""
    Model structure:

      Setup: {self._setup}
      Learn: 
      \t{self._learn}
      Predict: 
      \t{self._predict}
    """
