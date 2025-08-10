from dataclasses import dataclass

@dataclass
class ProviderMetrics:
    """Simple container for provider performance statistics."""
    total_requests: int = 0
    total_latency: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0

    def record(self, latency: float, tokens: int = 0, cost_per_1k: float = 0.0) -> None:
        """Record a new request's metrics."""
        self.total_requests += 1
        self.total_latency += latency
        self.total_tokens += tokens
        self.total_cost += (tokens / 1000.0) * cost_per_1k
