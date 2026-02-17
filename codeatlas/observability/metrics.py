from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "codeatlas_request_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "codeatlas_request_latency_seconds",
    "Request latency in seconds",
    ["path"],
)
