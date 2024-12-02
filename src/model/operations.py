import numpy as np 

SCALAR = 0
VECTOR = 1
ANY = 2

OPS_NAMES = [
  "addition",
  "matmul",
  "subtraction",
  "exp",
  "norm",
  "multiply"
]

OPS = [
  np.add,
  np.matmul,
  np.subtract,
  np.exp,
  np.linalg.norm,
  np.multiply
]

OPS_ARGS = {
  0: {
    "n_args": 2,
    "arg_types": ANY,
    "return_type": ANY # meaning corresponding to the arguments 
   },
  1: {
    "n_args": 2,
    "arg_types": VECTOR,
    "return_type": SCALAR
  },
  2: {
    "n_args": 2,
    "arg_types": ANY,
    "return_type": ANY
  },
  3: {
    "n_args": 1,
    "arg_types": ANY,
    "return_type": ANY
  },
  4: {
    "n_args": 1,
    "arg_types": ANY,
    "return_type": SCALAR
  },
  5: {
    "n_args": 2,
    "arg_types": ANY,
    "return_type": ANY
  }
}

N_OPS = len(OPS)