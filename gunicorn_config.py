import os
import multiprocessing

# Bind to the PORT environment variable
bind = "0.0.0.0:" + str(os.environ.get("PORT", 5000))

# Worker configuration - CRITICAL for memory optimization
workers = 1  # Single worker to minimize memory usage
worker_class = "sync"
threads = 1  # Reduce threads to save memory
timeout = 600  # Increased timeout for model loading (10 minutes)
keepalive = 2
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
max_requests = 50  # Restart worker after 50 requests to prevent memory leaks
max_requests_jitter = 10
worker_tmp_dir = "/dev/shm"  # Use shared memory for worker files (faster, less I/O)

# Startup
preload_app = True  # Load models before forking workers - saves memory

# Limit worker memory
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190
