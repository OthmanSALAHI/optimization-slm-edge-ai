import math
import threading
import time

import torch


def compute_loss(loss_values):
    """Compute average loss from a list of batch losses."""
    if not loss_values:
        return None
    return sum(loss_values) / len(loss_values)


def compute_perplexity(loss):
    """Compute perplexity from loss."""
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_training_time(start_time, end_time):
    """Compute elapsed training time in seconds."""
    return end_time - start_time


def compute_memory_usage(device):
    """Collect CPU and GPU memory usage when available."""
    memory = {
        "gpu_max_allocated_mb": None,
        "gpu_max_reserved_mb": None,
        "cpu_rss_mb": None,
    }

    if device.type == "cuda":
        memory["gpu_max_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            2,
        )
        memory["gpu_max_reserved_mb"] = round(
            torch.cuda.max_memory_reserved(device) / (1024 * 1024),
            2,
        )

    try:
        import psutil

        process = psutil.Process()
        memory["cpu_rss_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        pass

    return memory


def compute_inference_metrics(total_time_seconds, number_of_samples):
    """Compute inference time, latency, and throughput."""
    if number_of_samples == 0:
        return {
            "inference_time_seconds": None,
            "latency_ms_per_sample": None,
            "inference_samples_per_second": None,
        }

    return {
        "inference_time_seconds": total_time_seconds,
        "latency_ms_per_sample": (total_time_seconds / number_of_samples) * 1000,
        "inference_samples_per_second": number_of_samples / total_time_seconds
        if total_time_seconds > 0
        else None,
    }


def compute_compression_performance_tradeoff(
    final_model_size_mb,
    validation_loss,
    validation_perplexity,
):
    """Record the current no-compression baseline for later comparison."""
    return {
        "compression_applied": False,
        "compression_ratio": 1.0,
        "final_model_size_mb": final_model_size_mb,
        "quality_metric": "validation_loss and validation_perplexity",
        "validation_loss": validation_loss,
        "validation_perplexity": validation_perplexity,
        "note": "This is the fine-tuned baseline. Compare this with quantized or compressed models later.",
    }


class GpuEnergyTracker:
    """Estimate GPU energy usage with NVIDIA Management Library when available."""

    def __init__(self, device_index=0, sample_interval_seconds=0.5):
        self.device_index = device_index
        self.sample_interval_seconds = sample_interval_seconds
        self.samples_watts = []
        self.start_time = None
        self.end_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None
        self.available = False
        self.error = None

    def start(self):
        try:
            import pynvml

            self._nvml = pynvml
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False
            self.start_time = time.perf_counter()
            return

        self.start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_power, daemon=True)
        self._thread.start()

    def stop(self):
        self.end_time = time.perf_counter()

        if self.available:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()

            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _sample_power(self):
        while not self._stop_event.is_set():
            power_milliwatts = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            self.samples_watts.append(power_milliwatts / 1000)
            time.sleep(self.sample_interval_seconds)

    def summary(self):
        duration_seconds = None
        if self.start_time is not None and self.end_time is not None:
            duration_seconds = self.end_time - self.start_time

        if not self.available or not self.samples_watts or duration_seconds is None:
            return {
                "available": False,
                "energy_wh": None,
                "average_power_watts": None,
                "duration_seconds": duration_seconds,
                "note": "Install nvidia-ml-py and run on an NVIDIA GPU to estimate energy consumption.",
                "error": self.error,
            }

        average_power = sum(self.samples_watts) / len(self.samples_watts)
        energy_wh = average_power * duration_seconds / 3600

        return {
            "available": True,
            "energy_wh": energy_wh,
            "average_power_watts": average_power,
            "duration_seconds": duration_seconds,
            "samples": len(self.samples_watts),
        }
