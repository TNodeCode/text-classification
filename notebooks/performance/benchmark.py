import evaluate
import numpy as np
import torch

from pathlib import Path
from time import perf_counter


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, class_labels, query: str, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.class_labels = class_labels
        self.query = query
        self.optim_type = optim_type
        self.accuracy_score = evaluate.load("accuracy")

    def compute_accuracy(self):
        """Compute the accuracy."""
        preds, labels = [], []
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(self.class_labels.str2int(pred))
            labels.append(label)
        accuracy = self.accuracy_score.compute(predictions=preds, references=labels)
        return accuracy

    def compute_size(self):
        """Compute size of the model."""
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # calculate size in MB
        size_mb = np.round(Path(tmp_path).stat().st_size / (1024 ** 2), 3)
        # delete temporary file
        tmp_path.unlink()
        return {"size_mb": size_mb}
    
    def time_pipeline(self):
        """Compute time that pipeline needs to process query."""
        latencies = []
        # warmup
        for _ in range(10):
            _ = self.pipeline(self.query)
        # measure time
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(self.query)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        """Run the benchmark."""
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics