import hydra 
import logging 
import os
import numpy as np
import pickle 
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, classification_report

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

  # mean1 = np.random.randn(2)
  # mean2 = np.random.randn(2)
  # cov = np.random.randn(2, 2)
  # X1 = np.random.multivariate_normal(mean1, cov.T @ cov, n_samples)
  # X2 = np.random.multivariate_normal(mean2, cov.T @ cov, n_samples)
  # X = np.vstack((X1, X2))
  # y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
  # dataset = np.hstack((X, y.reshape(len(y), 1)))
  # np.random.shuffle(dataset)

  # X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.4, random_state=0)
  X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
  dataset = np.hstack([X, y[:, np.newaxis]])

  dtrain = dataset[: int(0.8 * n_samples)]
  dvalid = dataset[: int(0.2 * n_samples)]

  # X, y = make_circles(n_samples=int(n_samples * 0.1), noise=0.1, factor=0.4, random_state=42)
  X, y = make_moons(n_samples=int(n_samples * 0.3), noise=0.2, random_state=0)
  test = np.hstack([X, y[:, np.newaxis]])

  logger.info("Done creating datset")

  logger.info("Running evolution")
  # Setting up the experiment
  pool = Pool(cfg.pool.nscalars, cfg.pool.nvectors, cfg.dataset.nfeatures)
  evolution = EVOLUTION_TYPES[cfg.evolution.type](
    pool=pool,
    max_learn_length=cfg.model.max_learn_length,
    max_predict_length=cfg.model.max_predict_length,
    **cfg.evolution.params,
  )

  # Run evolution
  evolution.initialize_population(tr_data=dtrain, val_data=dvalid, normalize=sigmoid, loss_func=abs_loss)
  evolution.run_evolution(dtrain, dvalid, sigmoid, abs_loss)
  history = evolution.get_history()

  # Compute accuracy for the best model
  best_model = get_best_model(history)
  preds = np.array(get_prediction(best_model, pool, test, sigmoid)).astype(float)
  logger.info(f"Predictions: {preds}")
  logger.info(f"True: {test[:, -1]}")
  logger.info(f"Best model report on validation data: {classification_report(test[:, -1], preds)}")
  logger.info(best_model)


  logger.info("Saving results to pickle file")
  # Save history 
  with open(output_dir / "history", "wb") as fp:
    pickle.dump(history, fp)
  # Save model
  with open(output_dir / "best_model", "wb") as fp:
    pickle.dump(best_model, fp)

  logger.info("Plotting results to result.png")
  losses = []
  for exp in history:
      losses.append(exp["loss"])
  plt.figure(figsize=(16, 7))
  plt.plot(np.arange(len(losses)),losses)
  plt.xlabel("Cycles")
  plt.ylabel(f"{abs_loss.__name__}")
  plt.savefig(output_dir / "result.png")

  logger.info("Experiment completed successfully")

if __name__ == "__main__":
  main()
