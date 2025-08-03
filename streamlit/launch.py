#!/usr/bin/env python3
"""
Startup script for the PyPSA HRES Model Dashboard
This script checks dependencies and launches the Streamlit app

Use this to run the dashboard:
streamlit run streamlit/main.py
"""

import sys
import os
import subprocess
import importlib.util
import logging
from datetime import datetime

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'pypsa'
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print("pip install -r ../requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Starting PyPSA HRES Model Dashboard...")
    
    # Setup basic logging for launcher
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - LAUNCHER - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger = logging.getLogger('Dashboard_Launcher')
    logger.info("Launching PyPSA HRES Model Dashboard")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, "main.py")
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Main script path: {main_py}")
    
    try:
        # Launch streamlit
        logger.info("Starting Streamlit server on localhost:8501")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", main_py,
            "--server.port=8501",
            "--server.address=localhost"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user (Ctrl+C)")
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main function."""
    print("PyPSA HRES Model Dashboard Launcher")
    print("=" * 40)
    
    if check_dependencies():
        launch_dashboard()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
