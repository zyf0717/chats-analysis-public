"""
This script runs the Shiny application defined in the `app` module.

When executed as the main module, it starts the Shiny development server
on all available IP addresses (0.0.0.0) and listens on port 8000.

Modules:
    app: The Shiny application instance.

Usage:
    Run this script directly to start the Shiny development server:
    $ python main.py
"""

from app import app

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
