"""
Web server entry point.
Usage:
  python run_server.py
  python run_server.py --host 0.0.0.0 --port 8000
"""
import argparse
import uvicorn
from config import cfg


def main():
    p = argparse.ArgumentParser(description="Start Parking Detection web server")
    p.add_argument("--host",   default=cfg.server.host)
    p.add_argument("--port",   type=int, default=cfg.server.port)
    p.add_argument("--reload", action="store_true", default=cfg.server.reload)
    args = p.parse_args()

    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
