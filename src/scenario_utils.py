import pandas as pd
import numpy as np

def fill_missing_and_extend_data(df, start_date=None, end_date=None, freq='30min', target_column=None):
    """
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