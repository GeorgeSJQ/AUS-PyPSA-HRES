#!/usr/bin/env python3
"""
Main entry point for the PyPSA HRES Model Dashboard
Run this file to launch the Streamlit application

streamlit run streamlit/main.py
"""

import sys
import os
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    """Main function to set up and run the Streamlit app."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="PyPSA HRES Model Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Define pages
    pages = {
        "Model": [
            st.Page("model.py", title="HRES Model", icon="📊", default=True),
        ]
    }
    
    # Create navigation
    pg = st.navigation(pages)
    
    # Run the selected page
    pg.run()

if __name__ == "__main__":
    main()
