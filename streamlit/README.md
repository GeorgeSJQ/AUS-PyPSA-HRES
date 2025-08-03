# PyPSA HRES Model Dashboard

A Streamlit dashboard for running and analyzing PyPSA hybrid renewable energy system (HRES) models.

## Features

- Interactive parameter selection for model configuration
- Real-time model execution with progress tracking
- Comprehensive results visualization and analysis
- Export/import functionality for model results
- Multi-period optimization support

## Getting Started

### Prerequisites

Make sure you have all required dependencies installed:

```bash
pip install -r ../requirements.txt
```

### Running the Dashboard

There are several ways to launch the dashboard:

#### Option 1: Using the launch script (Recommended)
```bash
python launch.py
```
This script checks dependencies and launches the dashboard with proper configuration.

#### Option 2: Using the batch file (Windows)
```bash
run_dashboard.bat
```
Double-click the batch file or run it from command prompt.

#### Option 3: Direct Streamlit command
```bash
streamlit run main.py
```

#### Option 4: Using Python directly
```bash
python main.py
```

The dashboard will open in your web browser at `http://localhost:8501`.

## Dashboard Sections

### Model Configuration (Sidebar)

- **Timeline Parameters**: Set start year, end year, investment periods, and discount rate
- **Technology Parameters**: Select technologies to include and REZ location
- **VRE Parameters**: Configure solar and wind degradation rates
- **Load Parameters**: Set load profile (currently supports flat load)

### Main Dashboard

- **Model Execution**: Run the optimization model with current parameters
- **Results Analysis**: View results across multiple tabs:
  - Overview: Key metrics and capacity mix
  - Generation: Generator details and capacities
  - Storage: Storage system analysis
  - Economics: Cost breakdown and financial metrics
  - Time Series: Temporal analysis (planned)
  - Detailed Stats: Complete model statistics

### File Operations

- **Save Results**: Export model results to NetCDF format
- **Load Results**: Import previously saved model results

## File Structure

```
streamlit/
├── main.py                 # Application entry point
├── model.py               # Main dashboard implementation
├── streamlit_utils.py     # Utility functions for UI components
├── launch.py              # Dependency checker and launcher script
├── run_dashboard.bat      # Windows batch file launcher
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── README.md             # This file
```

## Key Functions

### `streamlit_utils.py`

- `parameter_input_section()`: Creates the parameter input sidebar
- `year_selector()`: Custom year selection widget
- `investment_years_selector()`: Multi-select for investment years
- `display_model_status()`: Shows model execution status
- `create_results_tabs()`: Creates results display tabs
- `plot_capacity_results()`: Generates capacity visualization

### `model.py`

- Main dashboard logic and model execution
- Network setup and optimization
- Results display and analysis
- File save/load operations

## Usage Tips

1. **Start with default parameters** to get familiar with the interface
2. **Monitor the model status** indicator during execution
3. **Save your results** after successful model runs for later analysis
4. **Use the expandable sections** to organize your view
5. **Check the logs** in the terminal for detailed error messages

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `INPUTS.xlsx`, `GENCOST_CAPEX.csv`, and `GENCOST_VARIABLE.csv` are in the `data/` directory
2. **Solver errors**: Make sure you have Gurobi properly installed and licensed
3. **Memory issues**: For large models, consider reducing the time horizon or number of investment periods
4. **Import errors**: Verify all dependencies are installed and the Python path is correctly set

### Error Messages

- **"INPUTS.xlsx not found"**: Check that the data file exists in the correct location
- **"Solver failed"**: Verify Gurobi installation or try with a different solver
- **"Error generating weather traces"**: Check REZ ID and reference year parameters

## Extending the Dashboard

To add new features:

1. **New input parameters**: Add widgets in `parameter_input_section()` in `streamlit_utils.py`
2. **Custom visualizations**: Create new plotting functions in `streamlit_utils.py`
3. **Additional analysis**: Add new tabs in `create_results_tabs()` and implement in `model.py`
4. **Data export options**: Extend the save/load functionality in `streamlit_utils.py`

## Support

For issues or questions, please refer to the main project documentation or create an issue in the project repository.
