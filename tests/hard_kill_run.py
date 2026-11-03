"""Train the tiny test run and SIGKILL this process once ``checkpoint_last.pt`` of a step is saved.

Usage: ``python hard_kill_run.py OUT_DIR STEP``. Run by ``test_resume.py``; no cleanup runs.
"""

import os
import signal
import sys

from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mfm.training import TrackingConfig, TrainingObserver, train


class KillAfterLast(TrainingObserver):
    def __init__(self, step):
        self.step = step

    def on_checkpoint(self, record):
        if record["kind"] == "last" and record["step"] == self.step:
            os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    out_dir, step = sys.argv[1], int(sys.argv[2])
    train(
        [make_synthetic_corpus()],
        out_dir=out_dir,
        model_config=TINY,
        training=reduced(steps=9, eval_every=3, milestone_steps=()),
        device="cpu",
        tracking=TrackingConfig(enabled=False),
        observers=[KillAfterLast(step)],
    )
