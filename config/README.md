# Configuration Files

This directory contains configuration files for the application.

## Files

### `gunicorn_config.py`
Production server configuration for Gunicorn WSGI server.

**Key Settings:**
- Worker processes and threads
- Timeout configurations
- Logging settings
- Memory optimizations

### `.env.example`
Template for environment variables. Copy to `.env` and customize:

```bash
cp config/.env.example .env
```

**Environment Variables:**
- `FLASK_ENV`: Application environment (development/production)
- `SECRET_KEY`: Flask secret key for session management
- `MAX_UPLOAD_SIZE`: Maximum file upload size in MB
- `MODEL_CACHE_SIZE`: Model cache size for memory optimization
- `DEBUG`: Enable/disable debug mode

## Usage

### Development
```bash
export FLASK_ENV=development
python run.py
```

### Production
```bash
gunicorn --config config/gunicorn_config.py src.app:app
```

## Security Notes

- **Never commit `.env` files** with actual secrets
- Use strong random values for `SECRET_KEY`
- Review security settings before production deployment
- See `docs/DEPLOYMENT.md` for production best practices
