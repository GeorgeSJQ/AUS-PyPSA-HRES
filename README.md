# PyPSA HRES Model Project

This project implements a multi-period energy system model using PyPSA, with a focus on renewable integration, storage, and cost analysis. The workspace includes data processing, scenario analysis, and a Streamlit dashboard for interactive exploration.

## Project Structure

- `src/` - Core Python modules for modeling, configuration, and utilities.
- `data/` - Input data files (CSV, Excel) for costs, traces, and profiles.
- `save files/` - Output files and model results (NetCDF).
- `streamlit/` - Streamlit dashboard code and utilities.
- `Model_v9_multi.ipynb` - Main Jupyter notebook for running and analyzing the model.
- `requirements.txt` - Python dependencies for the project.

## Getting Started

1. **Set up a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the Streamlit dashboard:**
   ```powershell
   cd streamlit
   python launch.py
   ```
   Or directly:
   ```powershell
   streamlit run streamlit/main.py
   ```
4. **Work with the Jupyter notebook:**
   Open `Model_v9_multi.ipynb` in VS Code or JupyterLab.

## Notes
- Input data is located in the `data/` folder. Update or add new data as needed.
- Model outputs are saved in the `save files/` directory.
- The Streamlit dashboard provides interactive visualization and scenario analysis.

## License
This project is for academic and research purposes.
