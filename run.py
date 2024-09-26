import hydra 
import logging 
import os
import numpy as np
import pickle 
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.config import EVOLUTION_TYPES
from src.additional.memory_pool import Pool
from src.additional.utils import sigmoid, abs_loss, get_best_model, get_prediction

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
  logger.info(OmegaConf.to_yaml(cfg))

  output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

  ### Create binary classification dataset
  logger.info("Creating dataset")
  n_samples = cfg.dataset.nsamples
  mean1 = np.random.randn(2)
  mean2 = np.random.randn(2)
  cov = np.random.randn(2, 2)
  X1 = np.random.multivariate_normal(mean1, cov.T @ cov, n_samples)
  X2 = np.random.multivariate_normal(mean2, cov.T @ cov, n_samples)
  X = np.vstack((X1, X2))
  y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
  dataset = np.hstack((X, y.reshape(len(y), 1)))
  np.random.shuffle(dataset)
  dtrain = dataset[: int(0.8 * n_samples)]
  dvalid = dataset[: int(0.2 * n_samples)]
  logger.info("Done creating datset")

  logger.info("Running evolution")
  # Setting up the experiment
  pool = Pool(cfg.pool.nscalars, cfg.pool.nvectors, cfg.dataset.nfeatures)
  evolution = EVOLUTION_TYPES[cfg.evolution.type](
    pool=pool,
    **cfg.evolution.params
  )

  # Run evolution
  evolution.initialize_population(tr_data=dtrain, val_data=dvalid, normalize=sigmoid, loss_func=abs_loss)
  evolution.run_evolution(dtrain, dvalid, sigmoid, abs_loss)
  history = evolution.get_history()

  # Compute accuracy for the best model
  best_model = get_best_model(history)
  logger.info(f"Best model accuracy on validation data: {accuracy_score(dvalid[:, -1], get_prediction(best_model, pool, dvalid, sigmoid))}")
  logger.info(best_model)

  # Save history 
  logger.info("Saving results to pickle file")
  with open(output_dir / "history", "wb") as fp:
    pickle.dump(history, fp)

  logger.info("Plotting results to result.png")
  losses = []
  for exp in history:
      losses.append(exp["loss"])
  plt.figure(figsize=(16, 7))
  plt.plot(np.arange(len(losses)),losses);
  plt.savefig(output_dir / "result.png")

  logger.info("Experiment completed successfully")

if __name__ == "__main__":
  main()
