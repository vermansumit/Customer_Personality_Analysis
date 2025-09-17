import time
start = time.time()
pred = models.predict(df)
latency_ms = (time.time() - start) * 1000
print(f"Latency: {latency_ms:.2f} ms per request")