import pandas as pd
import numpy as np
import plotly.graph_objects as go

from typing import Union, List, Optional, Dict


def fill_missing_and_extend_data(df: pd.DataFrame, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None, freq: str = '30min', target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Used for cleaning web scraped data by filling missing periods and extending the date range.
    Fill missing periods in a dataframe and optionally extend it to a specified date range.
    
    Parameters:
    df: DataFrame with DatetimeIndex
    start_date: Start date for extended range (if None, uses df start)
    end_date: End date for extended range (if None, uses df end)
    freq: Frequency for the date range (default '30min')
    target_column: Specific column to focus on for finding continuous data (if None, uses all columns)
    
    Returns:
    DataFrame with filled missing periods and/or extended range
    """
    
    # Step 1: Find the largest continuous block of data
    df_sorted = df.sort_index()
    
    # If target_column is specified, focus on that column for finding continuous segments
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataframe")
        
        # Use only the target column to identify continuous segments
        df_target = df_sorted[[target_column]].dropna()
        print(f"Focusing on column: {target_column}")
    else:
        # Remove rows that are completely NaN across all columns
        df_target = df_sorted.dropna(how='all')
        print("Using all columns to identify continuous segments")
    
    if len(df_target) == 0:
        raise ValueError("No valid data found in the dataframe")
    
    # Find gaps in the timestamps based on the target column(s)
    time_diff = df_target.index.to_series().diff()
    expected_diff = pd.Timedelta(freq)
    
    # Allow for some tolerance (1.5x the expected frequency)
    gap_mask = time_diff > (expected_diff * 1.5)
    
    if gap_mask.any():
        # Find all continuous segments
        gap_indices = gap_mask[gap_mask].index.tolist()
        
        # Split data into continuous segments
        segments = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            gap_loc = df_target.index.get_loc(gap_idx)
            if gap_loc > start_idx:
                # Get the full dataframe segment (all columns) for these indices
                segment_indices = df_target.iloc[start_idx:gap_loc].index
                segments.append(df_sorted.loc[segment_indices])
            start_idx = gap_loc
        
        # Add the last segment
        if start_idx < len(df_target):
            segment_indices = df_target.iloc[start_idx:].index
            segments.append(df_sorted.loc[segment_indices])
        
        # Find the longest continuous segment
        if segments:
            longest_segment = max(segments, key=len)
        else:
            longest_segment = df_sorted.loc[df_target.index]
    else:
        # No major gaps found, use all data
        longest_segment = df_sorted.loc[df_target.index]
    
    print(f"Using continuous segment from {longest_segment.index.min()} to {longest_segment.index.max()}")
    print(f"Segment length: {len(longest_segment)} records")
    
    # Step 2: Create the target date range
    if start_date is None:
        start_date = df_sorted.index.min()
    if end_date is None:
        end_date = df_sorted.index.max()
    
    target_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Step 3: Create result dataframe with target range
    result_df = pd.DataFrame(index=target_range, columns=df.columns)
    
    # Step 4: Fill with original data where available
    for idx in df_sorted.index:
        if idx in result_df.index:
            result_df.loc[idx] = df_sorted.loc[idx]
    
    # Step 5: Fill missing periods by cycling through the longest continuous segment
    # Handle each column separately
    for column in df.columns:
        column_missing_mask = result_df[column].isnull()
        missing_indices = result_df[column_missing_mask].index
        
        if len(missing_indices) > 0 and len(longest_segment) > 0:
            # Get non-NaN values from the longest segment for this column
            segment_column_data = longest_segment[column].dropna()
            
            if len(segment_column_data) > 0:
                cycle_length = len(segment_column_data)
                
                for i, missing_idx in enumerate(missing_indices):
                    # Use modulo to cycle through the available data for this column
                    source_idx = i % cycle_length
                    source_value = segment_column_data.iloc[source_idx]
                    result_df.loc[missing_idx, column] = source_value
                
                print(f"Filled {len(missing_indices)} missing values for column '{column}'")
            else:
                print(f"Warning: No valid data found for column '{column}' in the longest segment")
    
    return result_df

# Renewables Ninja Trace Cleaning Function
def extended_vre_trace(start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str, trace_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend VRE trace data by repeating the original pattern across a new date range.
    
    Parameters:
    start_date: dt.datetime - Start date for the extended range
    end_date: dt.datetime - End date for the extended range  
    freq: str - Frequency for pandas date_range (e.g., '30min', 'H')
    trace_df: DataFrame - Original VRE trace data with DatetimeIndex
    
    Returns:
    DataFrame with extended date range and repeated VRE patterns
    """
    
    # Create the new target date range
    target_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create result dataframe with new date range
    result_df = pd.DataFrame(index=target_range, columns=trace_df.columns)
    
    # Get the original data length for cycling
    original_length = len(trace_df)
    
    if original_length == 0:
        raise ValueError("Input trace_df is empty")
    
    # Fill the result by cycling through the original data
    for i, timestamp in enumerate(target_range):
        # Use modulo to cycle through the original data
        source_idx = i % original_length
        result_df.iloc[i] = trace_df.iloc[source_idx]
    
    print(f"Extended VRE trace from {start_date} to {end_date}")
    print(f"Original data length: {original_length} records")
    print(f"Extended data length: {len(result_df)} records")
    print(f"Data repeated {len(result_df) // original_length} times with {len(result_df) % original_length} partial cycles")
    
    return result_df


# Fuel Price Volatility Scenario Functions

def apply_fuel_price_volatility(
    fuel_cost_series: pd.Series,
    periods_to_modify: Optional[Union[List[int], str]] = "all",
    min_increase_factor: float = 1.0,
    max_increase_factor: float = 2.0,
    volatility_type: str = "uniform",
    resolution: str = "timestep",
    random_seed: Optional[int] = None
) -> pd.Series:
    """
    Apply price volatility to a pandas MultiIndex Series with fuel costs.
    
    Parameters:
    -----------
    fuel_cost_series : pd.Series
        Input series with MultiIndex (period, timestep) containing fuel costs
    periods_to_modify : list of int, str, or None
        Specific periods to modify. If "all", modifies all periods. 
        If None or empty list, returns original series.
    min_increase_factor : float
        Minimum multiplication factor (>=1.0 to ensure prices increase)
    max_increase_factor : float  
        Maximum multiplication factor (>min_increase_factor)
    volatility_type : str
        Type of random distribution ("uniform", "normal", "lognormal")
    resolution : str
        Resolution for applying volatility factors. Options:
        - "timestep": Apply different factors to each timestep (default)
        - "monthly": Apply same factor to all timesteps within each month
        - "quarterly": Apply same factor to all timesteps within each quarter
        - "annually": Apply same factor to all timesteps within each year
    random_seed : int, optional
        Seed for reproducible results
        
    Returns:
    --------
    pd.Series
        New series with same structure but modified fuel prices
    """
    
    # Validation
    if min_increase_factor < 1.0:
        raise ValueError("min_increase_factor must be >= 1.0 to ensure price increases")
    if max_increase_factor <= min_increase_factor:
        raise ValueError("max_increase_factor must be > min_increase_factor")
    
    valid_resolutions = ["timestep", "monthly", "quarterly", "annually"]
    if resolution not in valid_resolutions:
        raise ValueError(f"resolution must be one of {valid_resolutions}")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create a copy of the original series
    modified_series = fuel_cost_series.copy()
    
    # Handle periods_to_modify parameter
    if periods_to_modify is None or (isinstance(periods_to_modify, list) and len(periods_to_modify) == 0):
        return modified_series
    
    if periods_to_modify == "all":
        periods_to_modify = fuel_cost_series.index.get_level_values('period').unique().tolist()
    elif not isinstance(periods_to_modify, list):
        periods_to_modify = [periods_to_modify]
    
    def generate_random_factors(size, volatility_type, min_factor, max_factor):
        """Generate random factors based on volatility type"""
        if volatility_type == "uniform":
            return np.random.uniform(min_factor, max_factor, size=size)
        elif volatility_type == "normal":
            mean_factor = (min_factor + max_factor) / 2
            std_factor = (max_factor - min_factor) / 6
            factors = np.random.normal(mean_factor, std_factor, size=size)
            return np.clip(factors, min_factor, max_factor)
        elif volatility_type == "lognormal":
            mean_log = np.log((min_factor + max_factor) / 2)
            std_log = 0.2
            factors = np.random.lognormal(mean_log, std_log, size=size)
            return np.clip(factors, min_factor, max_factor)
        else:
            raise ValueError("volatility_type must be 'uniform', 'normal', or 'lognormal'")
    
    # Apply volatility to specified periods
    for period in periods_to_modify:
        if period not in fuel_cost_series.index.get_level_values('period'):
            print(f"Warning: Period {period} not found in series. Skipping.")
            continue
            
        # Get data for this period
        period_mask = fuel_cost_series.index.get_level_values('period') == period
        period_data = fuel_cost_series[period_mask]
        period_timestamps = period_data.index.get_level_values('timestep')
        
        if resolution == "timestep":
            # Apply different factors to each timestep (original behavior)
            n_timesteps = len(period_data)
            random_factors = generate_random_factors(
                n_timesteps, volatility_type, min_increase_factor, max_increase_factor
            )
            
        elif resolution == "monthly":
            # Group by month and apply same factor within each month
            monthly_groups = period_timestamps.to_series().dt.to_period('M').unique()
            random_factors = np.zeros(len(period_data))
            
            for month in monthly_groups:
                month_mask = period_timestamps.to_series().dt.to_period('M') == month
                month_factor = generate_random_factors(
                    1, volatility_type, min_increase_factor, max_increase_factor
                )[0]
                random_factors[month_mask] = month_factor
                
        elif resolution == "quarterly":
            # Group by quarter and apply same factor within each quarter
            quarterly_groups = period_timestamps.to_series().dt.to_period('Q').unique()
            random_factors = np.zeros(len(period_data))
            
            for quarter in quarterly_groups:
                quarter_mask = period_timestamps.to_series().dt.to_period('Q') == quarter
                quarter_factor = generate_random_factors(
                    1, volatility_type, min_increase_factor, max_increase_factor
                )[0]
                random_factors[quarter_mask] = quarter_factor
                
        elif resolution == "annually":
            # Apply same factor to entire year
            annual_factor = generate_random_factors(
                1, volatility_type, min_increase_factor, max_increase_factor
            )[0]
            random_factors = np.full(len(period_data), annual_factor)
        
        # Apply random factors to the period data
        modified_period_data = period_data * random_factors
        
        # Update the modified series
        modified_series[period_mask] = modified_period_data
        
        print(f"Applied {resolution} volatility to period {period}: "
              f"factor range [{random_factors.min():.3f}, {random_factors.max():.3f}]")
        
        if resolution != "timestep":
            n_unique_factors = len(np.unique(random_factors))
            print(f"  - Applied {n_unique_factors} unique factors across {len(random_factors)} timesteps")
    
    return modified_series

def plot_multiperiod_cost_series(
    cost_series: Union[pd.Series, Dict[str, pd.Series]],
    title: str = "Cost Series Over Time",
    y_label: str = "Cost ($/MWh)",
    colors: list = None
) -> go.Figure:
    """
    Plot a multiperiod cost series using Plotly with period-level x-axis labels
    while retaining full time resolution. Can handle single series or multiple series.
    
    Parameters:
    -----------
    cost_series : pd.Series or dict
        Single MultiIndex series with (period, timestep) index containing cost data,
        or dictionary with {series_name: series_data} for multiple series
    title : str
        Plot title
    y_label : str
        Y-axis label
    show_period_boundaries : bool
        Whether to show vertical lines at period boundaries
    period_boundary_color : str
        Color for period boundary lines
    period_boundary_width : float
        Width of period boundary lines
    height : int
        Plot height in pixels
    width : int
        Plot width in pixels
    colors : list, optional
        List of colors to use for multiple series. If None, uses default colors.
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    
    # Default colors if not provided
    if colors is None:
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    # Check if input is a single series or dictionary
    if isinstance(cost_series, pd.Series):
        # Convert single series to dictionary for uniform processing
        series_dict = {'Cost': cost_series}
        is_single_series = True
    elif isinstance(cost_series, dict):
        series_dict = cost_series
        is_single_series = False
    else:
        raise ValueError("cost_series must be either a pandas Series or a dictionary of Series")
    
    # Create the figure
    fig = go.Figure()
    
    # Plot each series
    for i, (series_name, series_data) in enumerate(series_dict.items()):
        # Create a continuous index for plotting
        plot_df = series_data.reset_index()
        plot_df['plot_index'] = range(len(plot_df))
        
        # Select color
        color = colors[i % len(colors)]
        
        # Add the cost series line
        fig.add_trace(
            go.Scatter(
                x=plot_df['plot_index'],
                y=plot_df[series_data.name] if series_data.name else plot_df.iloc[:, 2],
                mode='lines',
                name=series_name if not is_single_series else None,
                line=dict(width=1.5, color=color),
                hovertemplate=f'<b>{series_name}</b><br>' +
                             '<b>Period:</b> %{customdata[0]}<br>' +
                             '<b>Timestamp:</b> %{customdata[1]}<br>' +
                             '<b>Cost:</b> %{y:.2f}<br>' +
                             '<extra></extra>',
                customdata=plot_df[['period', 'timestep']].values,
                showlegend=not is_single_series
            )
        )
    
    # Get period information from the first series for x-axis labels
    first_series = list(series_dict.values())[0]
    plot_df = first_series.reset_index()
    plot_df['plot_index'] = range(len(plot_df))
    
    periods = plot_df['period'].unique()
    period_starts = []
    period_labels = []
    
    for period in periods:
        period_mask = plot_df['period'] == period
        period_start_idx = plot_df[period_mask]['plot_index'].iloc[0]
        period_starts.append(period_start_idx)
        period_labels.append(str(period))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Investment Periods",
        yaxis_title=y_label,
        showlegend=not is_single_series,
        hovermode='x unified'
    )
    
    # Set custom x-axis ticks at period boundaries
    fig.update_xaxes(
        tickmode='array',
        tickvals=period_starts,
        ticktext=period_labels,
        tickangle=0
    )
    
    return fig