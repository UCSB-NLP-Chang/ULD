import time
import json
from typing import Dict
from collections import defaultdict
from typing_extensions import override

import torch
import pandas as pd
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# This follows the SimpleProfiler from lightning, used to record training loop time


class SimpleProfileCallback(TrainerCallback):

    def __init__(self, dirpath, filename) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations: Dict = defaultdict(list)
        self.start_time = time.monotonic()

    def _make_report(self):
        report = []
        for action, d in self.recorded_durations.items():
            d_tensor = torch.tensor(d)
            sum_d = torch.sum(d_tensor).item()
            report.append({
                "action": action,
                "mean_duration": sum_d / len(d),
                "total_duration": sum_d
            })

        report = pd.DataFrame.from_records(
            report, columns=["action", "mean_duration", "total_duration"])
        return report

    @override
    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started.")
        self.current_actions[action_name] = time.monotonic()

    @override
    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started.")
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        #! initialize the duration times
        if state.is_world_process_zero:
            self.recorded_durations = defaultdict(list)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        report = self._make_report()
        report = report.sort_values(by="total_duration", ascending=False)
        report.to_csv(f"{self.dirpath}/{self.filename}.csv", index=False)
        json.dump(self.recorded_durations, open(
            f"{self.dirpath}/raw-{self.filename}.json", "w"), indent=4)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.start("train_epoch")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.stop("train_epoch")
