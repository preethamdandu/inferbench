import threading
import time
from dataclasses import dataclass

import structlog

try:
    import pynvml  # type: ignore

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = structlog.get_logger()


@dataclass
class GPUStats:
    peak_memory_mb: float
    avg_utilization_pct: float


class GPUMonitor:
    def __init__(self, device_id: int = 0, interval_s: float = 0.1) -> None:
        self.device_id = device_id
        self.interval_s = interval_s
        self._running = False
        self._thread: threading.Thread | None = None

        self._peak_mem = 0.0
        self._utilizations: list[float] = []

    def _monitor_loop(self) -> None:
        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        except Exception as e:  # noqa: BLE001 — pynvml raises various undocumented subclasses
            logger.warning("Failed to init pynvml", error=str(e))
            return

        while self._running:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)

                mem_mb = float(mem_info.used) / (1024.0 * 1024.0)
                if mem_mb > self._peak_mem:
                    self._peak_mem = mem_mb

                self._utilizations.append(float(util_rates.gpu))

            except OSError:  # noqa: BLE001 — pynvml may raise on driver query failure
                pass

            time.sleep(self.interval_s)

        try:
            pynvml.nvmlShutdown()
        except OSError:
            pass

    def start(self) -> None:
        self._peak_mem = 0.0
        self._utilizations.clear()

        if PYNVML_AVAILABLE:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self) -> GPUStats:
        self._running = False
        if self._thread:
            self._thread.join()

        avg_util = sum(self._utilizations) / len(self._utilizations) if self._utilizations else 0.0

        return GPUStats(peak_memory_mb=self._peak_mem, avg_utilization_pct=avg_util)
