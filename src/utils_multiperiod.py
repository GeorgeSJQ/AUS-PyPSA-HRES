import pandas as pd
import os
import numpy as np
import pypsa
import matplotlib.colors as mcolors
import calendar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Optional, List, Iterable, Dict, Union
from src.config import *
from isp_trace_parser import get_data


n = pypsa.Network()
# Get the directory of this file, then go up one level to get the project root
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# PyPSA Network Functions
# =========================================================
def restart_network(n):
    # Remove all buses, which will automatically remove connected components like generators, lines, etc.
    n.remove("Bus", n.buses.index)
    
    # Remove all generators
    n.remove("Generator", n.generators.index)
    
    # Remove all loads
    n.remove("Load", n.loads.index)
    
    # Remove all lines
    n.remove("Line", n.lines.index)

    # Remove all links
    n.remove("Link", n.links.index)
    
    # Remove all storage units
    n.remove("StorageUnit", n.storage_units.index)

    # Remove all stores
    n.remove("Store", n.stores.index)
    
    # Remove all other components (optional)
    n.remove("GlobalConstraint", n.global_constraints.index)
    n.remove("Carrier", n.carriers.index)
    n.remove("SubNetwork", n.sub_networks.index)

def total_generation(n):
    gen_by_carrier = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum()
    gen_summary = gen_by_carrier.groupby(['period']).sum().mul(0.5).T.div(1e3)
    gen_annual_avg = gen_summary.mean(axis=1).to_frame(name='AVG GENERATION (GWh)')

    return gen_annual_avg

def total_emissions(n):
    # Calculate total CO2 emissions per carrier
    emissions = ((n.generators_t.p/n.generators.efficiency) * n.generators['carrier'].map(n.carriers.co2_emissions)).T.groupby(n.generators.carrier).sum().div(1e6).sum(axis=1)  # in Mt CO₂
    emissions_df = emissions.reset_index()
    emissions_df.columns = ['Carrier', 'Emission_tCO2/MWh_th']
    return emissions_df

def nominal_total_system_costs(n):
    stats = n.statistics().groupby(level=1).sum()   
    # Calculate total expenditure
    if "Capital Expenditure" in stats.columns and "Operational Expenditure" in stats.columns:
        total_system_cost = stats[["Capital Expenditure", "Operational Expenditure"]].sum(axis=1).sum()
    
        return total_system_cost
    
def average_lifetime_srmc(n):
    """
    Calculate the average lifetime Short Run Marginal Cost (SRMC) for all generators.
    """
    # Main electricity price for compatibility
    if "electricity" in n.buses_t.marginal_price.columns:
        avg_electricity_price = n.buses_t.marginal_price["electricity"].mean()
    
        return avg_electricity_price


# =========================================================
# Cost Functions
# =========================================================

variable_name = {
    "Economiclife": "LIFESPAN",
    "Constructiontime": "BUILDTIME",
    "Efficiency": "EFF",
}

def get_gencost_capex(capex_path: str) -> pd.DataFrame:
    """
    Reads a CAPEX CSV file and returns a cleaned DataFrame with:
    - Columns: 'year', unit, and technology-specific CAPEX columns
    - Units standardized to /MW and /MWh
    """
    capex_df = pd.read_csv(capex_path)

    # Convert units to /MW and /MWh
    capex_df.loc[capex_df.unit.str.contains("/kW"), "value"] *= 1e3
    capex_df["unit"] = capex_df["unit"].str.replace("/kW", "/MW", regex=False)

    capex_df.loc[capex_df.unit.str.contains("/kWh"), "value"] *= 1e3
    capex_df["unit"] = capex_df["unit"].str.replace("/kWh", "/MWh", regex=False)

    # Pivot to wide format: one column per technology
    capex_wide = capex_df.pivot_table(
        index="year", columns="technology", values="value"
    )
    capex_wide.columns.name = None  # Remove columns name to clean up output

    # Reset index and rename technology columns to include '_CAPEX'
    capex_wide = capex_wide.reset_index()
    capex_wide.rename(
        columns={col: f"{col}_CAPEX" for col in capex_wide.columns if col != "year"},
        inplace=True
    )
    
    # Convert all column names to uppercase
    capex_wide.columns = [col.upper() for col in capex_wide.columns]

    return capex_wide.sort_values("YEAR").reset_index(drop=True)

def get_gencost_variables(variable_path: str, end_year: int) -> pd.DataFrame:
    """
    Reads variable costs, reshapes them to wide format, applies unit conversion,
    replaces variable names, drops unnecessary columns, computes fuel averages,
    and expands the data annually out to `end_year` with forward-filling.
    """
    variables_df = pd.read_csv(variable_path)

    # Convert units to /MW and /MWh
    variables_df.loc[variables_df.unit.str.contains("/kW"), "value"] *= 1e3
    variables_df["unit"] = variables_df["unit"].str.replace("/kW", "/MW", regex=False)

    variables_df.loc[variables_df.unit.str.contains("/kWh"), "value"] *= 1e3
    variables_df["unit"] = variables_df["unit"].str.replace("/kWh", "/MWh", regex=False)

    # Replace variable names and create 'tech_variable'
    variables_df["variable"] = variables_df["variable"].replace(variable_name)
    variables_df["tech_variable"] = variables_df["technology"] + "_" + variables_df["variable"]

    # Pivot to wide format
    variables_df = variables_df.pivot_table(
        index="year", columns="tech_variable", values="value"
    )

    variables_df.columns.name = None
    variables_df = variables_df.reset_index()
    variables_df.columns = [col.upper() for col in variables_df.columns]

    # 1. Drop columns with 'CAPITAL' or 'CF'
    drop_keywords = ["CAPITAL", "CF"]
    cols_to_drop = [col for col in variables_df.columns if any(key in col for key in drop_keywords)]
    variables_df.drop(columns=cols_to_drop, inplace=True)

    # 2. Replace FUEL_HIGH and FUEL_LOW with averaged FUEL
    fuel_cols = [col for col in variables_df.columns if col.endswith("_FUEL_HIGH") or col.endswith("_FUEL_LOW")]
    techs = set(col.replace("_FUEL_HIGH", "").replace("_FUEL_LOW", "") for col in fuel_cols)

    for tech in techs:
        high_col = f"{tech}_FUEL_HIGH"
        low_col = f"{tech}_FUEL_LOW"
        if high_col in variables_df.columns and low_col in variables_df.columns:
            variables_df[f"{tech}_FUEL"] = (variables_df[high_col] + variables_df[low_col]) / 2 * 3.6
            variables_df.drop(columns=[high_col, low_col], inplace=True)

    # 3. Drop all-zero columns (except YEAR)
    cols_to_check = variables_df.columns.difference(["YEAR"])
    zero_cols = variables_df[cols_to_check].columns[(variables_df[cols_to_check] == 0).all()]
    variables_df.drop(columns=zero_cols, inplace=True)

    # 4. Expand to full year range and fill in-between values correctly
    variables_df["YEAR"] = variables_df["YEAR"].round().astype(int)
    variables_df = variables_df.set_index("YEAR")
    full_years = range(2025, end_year + 1)
    variables_df = variables_df.reindex(full_years)
    variables_df = variables_df.ffill()
    variables_df = variables_df.reset_index()

    # 5. Add marginal cost columns: TECH_MC = TECH_VOM + TECH_FUEL / TECH_EFF
    techs = set()

    # Extract base tech names from _FUEL columns
    for col in variables_df.columns:
        if col.endswith("_FUEL"):
            tech = col[:-5]  # strip "_FUEL"
            techs.add(tech)

    # For each tech, create _MC column if all dependencies exist
    for tech in techs:
        vom_col = f"{tech}_VOM"
        fuel_col = f"{tech}_FUEL"
        eff_col = f"{tech}_EFF"
        if all(col in variables_df.columns for col in [vom_col, fuel_col, eff_col]):
            variables_df[f"{tech}_MC"] = variables_df[vom_col] + (variables_df[fuel_col] / variables_df[eff_col])

    return variables_df

def get_isp_tech_params(
    excel_path, 
    sheet_name="TECH_PARAMS", 
    usecols="A:U", 
    ):
    """
    Reads technology parameters from an Excel file and returns a DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing technology parameters.
        SUMMER_PEAK: % of nameplate capacity available during summer peak
        SUMMER_TYP: % of nameplate capacity available during summer typical
        WINTER_TYP: % of nameplate capacity available during winter typical
        OUTAGE: % of time out	
        DAYS_OUT: equivalent avg days per year on planned outage
        CONNECTION_COST: $/MW
        LEAD_TIME: years
        ECONOMIC_LIFE: years
        TECHNICAL_LIFE: years
        FOM: $/MW/annum
        VOM: $/MWh
        HEAT_RATE: GJ/MWh
        FUEL_COST: $/GJ for 2025-26
        EMISSIONS: kg/MWh for 2025-26
        AUX_LOAD: % of nameplate capacity used for auxiliary load
        SRMC: $/MWh for 2025-26
    """
    
    generator_params = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        usecols=usecols,
    )
    
    return generator_params

def get_isp_fuel_costs(excel_path, sheet_name="FUEL_COSTS", usecols="A:AJ"):
    """
    Reads fuel costs from an Excel file and returns a DataFrame.
    
    Parameters:
    -----------
    excel_path : str
        Path to the Excel file containing fuel costs
    sheet_name : str, default "FUEL_COSTS"
        Name of the sheet in the Excel file to read from
        
    Returns:
    """
    fuel_costs = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        usecols=usecols,
    )
    
    return fuel_costs

def get_isp_tech_build_costs(excel_path, sheet_name="TECH_BUILD_COSTS", usecols="A:AK", convert_to_MW=True):
    """
    Reads technology build costs from an Excel file and returns a DataFrame.
    
    Parameters:
    -----------
    excel_path : str
        Path to the Excel file containing build costs
    sheet_name : str, default "BUILD_COSTS"
        Name of the sheet in the Excel file to read from
        
    Returns:
    LOCATION_COST_FACTOR: decimal representing the cost factor for location
    
    """
    build_costs = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        usecols=usecols,
    )
    # Convert all cost data from $/kW to $/MW if specified
    if convert_to_MW:
        cost_columns = build_costs.columns[6:]
        for col in cost_columns:
            build_costs[col] *= 1e3  # Convert from $/kW to $/MW
    
    return build_costs

def get_annualized_cost(cost, lifetime, discount_rate):
    """Returns annualised cost"""
    r = discount_rate
    annuity_factor = r / (1.0 - 1.0 / (1.0 + r) ** lifetime)
    return cost * annuity_factor

def get_generator_capex(
        build_costs: pd.DataFrame, 
        tech_params: pd.DataFrame, 
        discount_rate: float, 
        year: int, 
        technology: str, 
        rez: str, 
        annuitise: bool,
        lifetime = None,
        additional_costs = 0,
        ) -> float:
    """
    Calculate the capital expenditure (CAPEX) for generators based on build costs and technology parameters.
    """
    capex = build_costs[(build_costs['GENERATOR'] == technology) & (build_costs['REZ'] == rez)][year].iloc[0]
    fom_series = tech_params[(tech_params['GENERATOR'] == technology) & (tech_params['REZ'] == rez)]['FOM']
    if fom_series.empty:
        fom = 0
    else:
        fom = fom_series.iloc[0]

    if annuitise and lifetime is not None:
        annutised_capex = get_annualized_cost(
            cost=capex + additional_costs,
            lifetime=lifetime,
            discount_rate=discount_rate
        )
        final_capex = annutised_capex + fom 
    else:
        final_capex = capex + fom + additional_costs

    return float(final_capex)

def get_generator_marginal_cost_series(
        n,
        tech_params: pd.DataFrame, 
        fuel_costs: pd.DataFrame,
        generator: str,
        fuel_type: str,
        rez: str,
    ) -> pd.Series:
    """
    Calculate the Short Run Marginal Cost (SRMC) for generators based on technology parameters and fuel costs.
    
    Parameters:
    -----------
    tech_params : pd.DataFrame
        DataFrame containing technology parameters including VOM, HEAT_RATE, and fuel type information
    fuel_costs : pd.DataFrame
        DataFrame containing fuel costs for different technologies and REZs by year
    fuel_type : str
        The fuel type to filter generators (e.g., "GAS", "COAL", "DIESEL")
        
    Returns:
    --------
    pd.Series
        Series with SRMC values for each technology and REZ combination by year
    """
    VOM = tech_params.loc[(tech_params['GENERATOR'] == generator) & (tech_params['REZ'] == rez), 'VOM'].values[0]
    heat_rate = tech_params.loc[(tech_params['GENERATOR'] == generator) & (tech_params['REZ'] == rez), 'HEAT_RATE'].values[0]
    
    # Select the row matching the criteria and return all columns to the right of 'TECH'
    row = fuel_costs.loc[
        (fuel_costs['FUEL_TYPE'] == fuel_type) &
        (fuel_costs['REZ'] == rez) &
        (fuel_costs['TECH'] == generator)
    ]
    # Find the index of the 'TECH' column
    tech_col_idx = row.columns.get_loc('TECH')
    # Ensure tech_col_idx is an integer before adding 1
    if not isinstance(tech_col_idx, int):
        tech_col_idx = int(tech_col_idx)
    # Return all columns to the right of 'TECH'
    marginal_cost = row.iloc[:, tech_col_idx + 1:].melt(
        var_name='YEAR',
        value_name='MARGINAL_COST'
    )
    marginal_cost['MARGINAL_COST'] = marginal_cost['MARGINAL_COST'] * heat_rate + VOM
    
    year_to_fuel_cost = marginal_cost.set_index('YEAR')['MARGINAL_COST'].to_dict()

    # Handle MultiIndex snapshots
    if isinstance(n.snapshots, pd.MultiIndex):
        # Extract years from the 'timestep' level
        years = n.snapshots.get_level_values('timestep').year
    else:
        # Single index - extract years directly
        years = n.snapshots.year
    
    marginal_cost_values = [year_to_fuel_cost.get(year, 0) for year in years]

    # Step 3: Create the Series with the same index as n.snapshots
    mc_series = pd.Series(marginal_cost_values, index=n.snapshots, name='MC')
    
    return mc_series

def filter_and_process_input_costs(
    inputs_df, 
    techs_to_keep = ["SOLAR_PV", "WIND", "BESS_1HR", "BESS_2HR","BESS_4HR", "BESS_8HR", "BESS_12HR", "BESS_24HR", "GAS_RECIP", "OCGT"], 
    annuitise_capex=True, 
    discount_rate=0.08, 
    model_horizon=None
    ):
    """
    Filter inputs dataframe by technologies and process CAPEX values.
    
    Parameters:
    -----------
    inputs_df : pd.DataFrame
        Input dataframe containing technology costs
    techs_to_keep : list
        List of technology names to keep in the dataset
    annuitise_capex : bool, default True
        Whether to annuitize CAPEX using get_annualized_cost()
        If False, simply adds FOM to CAPEX
    discount_rate : float, default 0.08
        Discount rate for annuitization (only used if annuitise_capex=True)
    tech_life : dict, optional
        Dictionary mapping technology names to their economic life
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with YEAR as index
    """
    # Create a copy to avoid modifying the original
    processed_df = inputs_df.copy()
    
    # Filter columns based on technologies to keep
    filtered_columns = [
        col for col in processed_df.columns if any(tech in col for tech in techs_to_keep)
    ]
    if "YEAR" not in filtered_columns:
        filtered_columns.append("YEAR")
    
    processed_df = processed_df[filtered_columns]
    
    # Process CAPEX values
    capex_columns = [col for col in processed_df.columns if col.endswith("_CAPEX")]
    
    for capex_column in capex_columns:
        technology = capex_column.replace("_CAPEX", "")
        if "BESS" in technology:
            technology = "BESS"
        
        # Get FOM values
        fom_column = f"{technology}_FOM"
        if fom_column in processed_df.columns:
            fom_values = processed_df[fom_column]
        else:
            print(f"No FOM column for {technology}. Assuming FOM = 0.")
            fom_values = 0
        
        # Process CAPEX based on annuitization choice
        if annuitise_capex:
            annuitised_capex = get_annualized_cost(
                cost=processed_df[capex_column], 
                lifetime=model_horizon, 
                discount_rate=discount_rate
            )
            processed_df[capex_column] = annuitised_capex + fom_values
        else:
            processed_df[capex_column] += fom_values
    
    # Set YEAR as index
    processed_df.set_index("YEAR", inplace=True)
    
    return processed_df

def create_multiindex_snapshots(
    start_date: str, 
    end_date: str, 
    freq: str = "30min", 
    investment_periods: List[int] = None
) -> pd.MultiIndex:
    """
    Create a MultiIndex with 'period' and 'timestep' levels from date range parameters.
    
    Parameters:
    -----------
    start_date : str
        Start date string (e.g., "2025-01-01")
    end_date : str
        End date string (e.g., "2030-12-31")
    freq : str, default "30min"
        Frequency for date range (e.g., "h" for hourly, "30min" for 30-minute intervals)
    investment_periods : List[int], optional
        List of investment period years (e.g., [2025, 2030, 2035])
        If None, uses unique years from the date range
    
    Returns:
    --------
    pd.MultiIndex
        MultiIndex with levels ('period', 'timestep') excluding February 29th dates
    """
    # Create the date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Remove February 29th dates
    date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]
    
    # Extract years from date range
    years = date_range.year
    
    # If investment_periods not provided, use unique years from date range
    if investment_periods is None:
        investment_periods = sorted(years.unique())
    
    # Assign periods to each timestamp
    periods = []
    for year in years:
        # Find the appropriate investment period for this year
        period = None
        for i, inv_period in enumerate(investment_periods):
            # Check if this is the last period or if year is before next period
            if i == len(investment_periods) - 1 or year < investment_periods[i + 1]:
                if year >= inv_period:
                    period = inv_period
                    break
        
        # If no period found, assign to the first period (for years before first period)
        # or to the last period (for years after last period)
        if period is None:
            if year < investment_periods[0]:
                period = investment_periods[0]
            else:
                period = investment_periods[-1]
        
        periods.append(period)
    
    # Create MultiIndex
    multiindex = pd.MultiIndex.from_arrays(
        [periods, date_range], 
        names=['period', 'timestep']
    )
    
    return multiindex

# Solar and Wind Trace Functions ===========================================================
def _create_multiindex_dataframe(df: pd.DataFrame, investment_periods: List[int]) -> pd.DataFrame:
    """
    Convert a DataFrame with DATETIME column to a MultiIndex DataFrame with 'period' and 'timestep' levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'DATETIME' column
    investment_periods : List[int]
        List of investment period years (e.g., [2025, 2030, 2035])
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with MultiIndex ('period', 'timestep')
    """
    if 'DATETIME' not in df.columns:
        raise ValueError("DataFrame must have a 'DATETIME' column")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract years from DATETIME
    years = df_copy['DATETIME'].dt.year
    
    # Assign periods to each timestamp
    periods = []
    for year in years:
        # Find the appropriate investment period for this year
        period = None
        for i, inv_period in enumerate(investment_periods):
            # Check if this is the last period or if year is before next period
            if i == len(investment_periods) - 1 or year < investment_periods[i + 1]:
                if year >= inv_period:
                    period = inv_period
                    break
        
        # If no period found, assign to the first period (for years before first period)
        # or to the last period (for years after last period)
        if period is None:
            if year < investment_periods[0]:
                period = investment_periods[0]
            else:
                period = investment_periods[-1]
        
        periods.append(period)
    
    # Create MultiIndex
    df_copy['period'] = periods
    df_copy = df_copy.set_index(['period', 'DATETIME'])
    df_copy.index.names = ['period', 'timestep']
    
    return df_copy

def get_solar_trace(
        start_year: int,            # In calendar year
        end_year: int,              # In calendar year
        rez_id: REZIDS,   
        reference_year: Optional[int] = 2016,
        solar_type: Optional[str] = 'SAT',
        multi_reference_year: Optional[bool] = False,
        year_type: Optional[str] = 'calendar',  # 'calendar' or 'fy'
        investment_periods: Optional[List[int]] = None,
        freq: Optional[str] = 'h',  # Frequency for date range
) -> pd.DataFrame:
    
    trace_dir = os.path.join(base_path, "data", "parsed_traces", "solar")
    
    if not multi_reference_year:
        solar_trace = get_data.solar_area_single_reference_year(
            start_year=start_year,
            end_year=end_year,
            reference_year=reference_year,
            area=rez_id,
            technology=solar_type,
            directory=trace_dir,
            year_type=year_type,
        ).rename(columns={'Datetime':'DATETIME', 'Value': 'SOLAR_TRACE'})
    
    else:
        years = list(range(start_year, end_year + 1))
        reference_years = list(range(2011, 2024))  # 2011 to 2023 inclusive
        assigned_years = {year: reference_years[i % len(reference_years)] for i, year in enumerate(years)}
        solar_trace = get_data.solar_area_multiple_reference_years(
            reference_years=assigned_years,
            area=rez_id,
            technology=solar_type,
            directory=trace_dir,
            year_type=year_type,
        ).rename(columns={'Datetime':'DATETIME', 'Value': 'SOLAR_TRACE'})
    solar_trace['DATETIME'] = pd.to_datetime(solar_trace['DATETIME'])
    if freq is not None:
        solar_trace = solar_trace.groupby(pd.Grouper(key='DATETIME', freq=freq)).aggregate({'SOLAR_TRACE': 'mean'}).reset_index()
    # Convert to MultiIndex if investment periods are provided
    if investment_periods is not None:
        solar_trace = _create_multiindex_dataframe(solar_trace, investment_periods)

    return solar_trace

def get_wind_trace(
        start_year: int,            # In calendar year
        end_year: int,              # In calendar year
        rez_id: REZIDS,   
        reference_year: Optional[int] = 2016,
        wind_type: Optional[str] = 'WH',
        multi_reference_year: Optional[bool] = False,
        year_type: Optional[str] = 'calendar',  # 'calendar' or 'fy'
        investment_periods: Optional[List[int]] = None,
        freq: Optional[str] = 'h',  # Frequency for date range
) -> pd.DataFrame:
    
    trace_dir = os.path.join(base_path, "data", "parsed_traces", "wind")
    
    if not multi_reference_year:
        wind_trace = get_data.wind_area_single_reference_year(
            start_year=start_year,
            end_year=end_year,
            reference_year=reference_year,
            area=rez_id,
            resource_quality=wind_type,
            directory=trace_dir,
            year_type=year_type,
        ).rename(columns={'Datetime':'DATETIME', 'Value': 'WIND_TRACE'})
    else:
        years = list(range(start_year, end_year + 1))
        reference_years = list(range(2011, 2024))
        assigned_years = {year: reference_years[i % len(reference_years)] for i, year in enumerate(years)}
        wind_trace = get_data.wind_area_multiple_reference_years(
            reference_years=assigned_years,
            area=rez_id,
            resource_quality=wind_type,
            directory=trace_dir,
            year_type=year_type,
        ).rename(columns={'Datetime':'DATETIME', 'Value': 'WIND_TRACE'})

    wind_trace['DATETIME'] = pd.to_datetime(wind_trace['DATETIME'])
    if freq is not None:
        wind_trace = wind_trace.groupby(pd.Grouper(key='DATETIME', freq=freq)).aggregate({'WIND_TRACE': 'mean'}).reset_index()
    # Convert to MultiIndex if investment periods are provided
    if investment_periods is not None:
        wind_trace = _create_multiindex_dataframe(wind_trace, investment_periods)
    
    return wind_trace

def solar_trace_construction(
        start_year: int,
        end_year: int,
        rez_ids: REZIDS,
        reference_year: int = 2016,
        solar_type: str = 'SAT',
        annual_degradation: Optional[float] = None,
        lifetime: Optional[int] = None,
        build_year: Optional[int] = None,
        multi_reference_year: Optional[bool] = False,
        year_type: Optional[str] = 'calendar',  # 'calendar' or 'fy'
        investment_periods: Optional[List[int]] = None,
        freq: Optional[str] = 'h',  # Frequency for date range
) -> pd.DataFrame:
    """
    Create a DataFrame with timestamps for the entire period and fill with solar trace data.
    Set values to 0 before build_year and apply degradation after build_year.
    """
    if build_year is None:
        build_year = start_year

    solar_trace = get_solar_trace(
        start_year=start_year, 
        end_year=end_year, 
        rez_id=rez_ids, 
        reference_year=reference_year, 
        solar_type=solar_type,
        multi_reference_year=multi_reference_year,
        year_type=year_type,
        investment_periods=investment_periods,
        freq=freq
    ).rename(columns={'SOLAR_TRACE': f'SOLAR_{build_year}'})

    # If we have a MultiIndex DataFrame, we need to work with the timestep level
    if isinstance(solar_trace.index, pd.MultiIndex):
        solar_trace.loc[solar_trace.index.get_level_values('timestep').year < build_year, f'SOLAR_{build_year}'] = 0.0
        if lifetime is not None:
            solar_trace.loc[solar_trace.index.get_level_values('timestep').year >= build_year + lifetime, f'SOLAR_{build_year}'] = 0.0
        solar_trace = solar_trace[solar_trace.index.get_level_values('timestep').year < end_year]
        
        if annual_degradation is not None and lifetime is not None:
            timestep_years = solar_trace.index.get_level_values('timestep').year
            for year in range(build_year, build_year + lifetime):
                year_mask = timestep_years == year
                if year_mask.any():
                    years_since_build = year - build_year
                    degradation_factor = (1 - annual_degradation) ** years_since_build
                    solar_trace.loc[year_mask, f'SOLAR_{build_year}'] *= degradation_factor
    else:
        # Handle regular DataFrame (backward compatibility)
        solar_trace.loc[solar_trace['DATETIME'].dt.year < build_year, f'SOLAR_{build_year}'] = 0.0
        if lifetime is not None:
            solar_trace.loc[solar_trace['DATETIME'].dt.year >= build_year + lifetime, f'SOLAR_{build_year}'] = 0.0
        solar_trace = solar_trace[solar_trace['DATETIME'].dt.year < end_year]
        
        if annual_degradation is not None and lifetime is not None:
            solar_trace['YEAR'] = solar_trace['DATETIME'].dt.year
            for year in range(build_year, build_year + lifetime):
                year_mask = solar_trace['YEAR'] == year
                if year_mask.any():
                    years_since_build = year - build_year
                    degradation_factor = (1 - annual_degradation) ** years_since_build
                    solar_trace.loc[year_mask, f'SOLAR_{build_year}'] *= degradation_factor
            solar_trace = solar_trace.drop(columns='YEAR')
        
        solar_trace = solar_trace.set_index('DATETIME')
    
    # Remove Feb 29th dates and add backfilled row at the beginning
    if isinstance(solar_trace.index, pd.MultiIndex):
        # Remove Feb 29th dates from MultiIndex
        dt_index = solar_trace.index.get_level_values('timestep')
        solar_trace = solar_trace[~((dt_index.month == 2) & (dt_index.day == 29))]
        
        # Add backfilled row at the beginning of the first period
        if freq is None or freq == '30min':
            first_period = solar_trace.index.get_level_values('period')[0]
            new_index = pd.MultiIndex.from_tuples(
                [(first_period, pd.Timestamp(f"{start_year}-01-01 00:00:00"))],
                names=solar_trace.index.names
            )
            
            # Create a DataFrame with NaN values for the new row
            new_row = pd.DataFrame(
                np.nan,
                index=new_index,
                columns=solar_trace.columns
            )
            
            # Concatenate the new row and the existing DataFrame
            solar_trace = pd.concat([new_row, solar_trace])
            # Sort the index to maintain chronological order
            solar_trace = solar_trace.sort_index()
            # Backfill the NaN values
            solar_trace = solar_trace.bfill()
    else:
        # Remove Feb 29th dates from regular DatetimeIndex
        solar_trace = solar_trace[~((solar_trace.index.month == 2) & (solar_trace.index.day == 29))]
        
        # Add backfilled row at the beginning
        if freq is None or freq == '30min':
            new_index = pd.DatetimeIndex([pd.Timestamp(f"{start_year}-01-01 00:00:00")])
            new_row = pd.DataFrame(
                np.nan,
                index=new_index,
                columns=solar_trace.columns
            )
            
            # Concatenate and backfill
            solar_trace = pd.concat([new_row, solar_trace])
            solar_trace = solar_trace.sort_index()
            solar_trace = solar_trace.bfill()
        
    return solar_trace

def wind_trace_construction(
        start_year: int,
        end_year: int,
        rez_ids: REZIDS,
        reference_year: int = 2016,
        wind_type: str = 'WH',
        annual_degradation: Optional[float] = None,
        lifetime: Optional[int] = None,
        build_year: Optional[int] = None,
        multi_reference_year: Optional[bool] = False,
        year_type: Optional[str] = 'calendar',  # 'calendar' or 'fy'
        investment_periods: Optional[List[int]] = None,
        freq: Optional[str] = 'h',  # Frequency for date range
) -> pd.DataFrame:
    """
    Create a DataFrame with timestamps for the entire period and fill with wind trace data.
    Set values to 0 before build_year and apply degradation after build_year.
    """
    if build_year is None:
        build_year = start_year

    wind_trace = get_wind_trace(
        start_year=start_year, 
        end_year=end_year, 
        rez_id=rez_ids, 
        reference_year=reference_year, 
        wind_type=wind_type,
        multi_reference_year=multi_reference_year,
        year_type=year_type,
        investment_periods=investment_periods,
        freq=freq
    ).rename(columns={'WIND_TRACE': f'WIND_{build_year}'})

    # If we have a MultiIndex DataFrame, we need to work with the timestep level
    if isinstance(wind_trace.index, pd.MultiIndex):
        wind_trace.loc[wind_trace.index.get_level_values('timestep').year < build_year, f'WIND_{build_year}'] = 0.0
        if lifetime is not None:
            wind_trace.loc[wind_trace.index.get_level_values('timestep').year >= build_year + lifetime, f'WIND_{build_year}'] = 0.0
        wind_trace = wind_trace[wind_trace.index.get_level_values('timestep').year < end_year]
        
        if annual_degradation is not None and lifetime is not None:
            timestep_years = wind_trace.index.get_level_values('timestep').year
            for year in range(build_year, build_year + lifetime):
                year_mask = timestep_years == year
                if year_mask.any():
                    years_since_build = year - build_year
                    degradation_factor = (1 - annual_degradation) ** years_since_build
                    wind_trace.loc[year_mask, f'WIND_{build_year}'] *= degradation_factor
    else:
        # Handle regular DataFrame (backward compatibility)
        wind_trace.loc[wind_trace['DATETIME'].dt.year < build_year, f'WIND_{build_year}'] = 0.0
        if lifetime is not None:
            wind_trace.loc[wind_trace['DATETIME'].dt.year >= build_year + lifetime, f'WIND_{build_year}'] = 0.0
        wind_trace = wind_trace[wind_trace['DATETIME'].dt.year < end_year]

        if annual_degradation is not None and lifetime is not None:
            wind_trace['YEAR'] = wind_trace['DATETIME'].dt.year
            for year in range(build_year, build_year + lifetime):
                year_mask = wind_trace['YEAR'] == year
                if year_mask.any():
                    years_since_build = year - build_year
                    degradation_factor = (1 - annual_degradation) ** years_since_build
                    wind_trace.loc[year_mask, f'WIND_{build_year}'] *= degradation_factor
            wind_trace = wind_trace.drop(columns='YEAR')
        
        wind_trace = wind_trace.set_index('DATETIME')
    
    # Remove Feb 29th dates and add backfilled row at the beginning
    if isinstance(wind_trace.index, pd.MultiIndex):
        # Remove Feb 29th dates from MultiIndex
        dt_index = wind_trace.index.get_level_values('timestep')
        wind_trace = wind_trace[~((dt_index.month == 2) & (dt_index.day == 29))]
        
        # Add backfilled row at the beginning of the first period
        if freq is None or freq == '30min':
            first_period = wind_trace.index.get_level_values('period')[0]
            new_index = pd.MultiIndex.from_tuples(
                [(first_period, pd.Timestamp(f"{start_year}-01-01 00:00:00"))],
                names=wind_trace.index.names
            )
            
            # Create a DataFrame with NaN values for the new row
            new_row = pd.DataFrame(
                np.nan,
                index=new_index,
                columns=wind_trace.columns
            )
            
            # Concatenate the new row and the existing DataFrame
            wind_trace = pd.concat([new_row, wind_trace])
            # Sort the index to maintain chronological order
            wind_trace = wind_trace.sort_index()
            # Backfill the NaN values
            wind_trace = wind_trace.bfill()
    else:
        # Remove Feb 29th dates from regular DatetimeIndex
        wind_trace = wind_trace[~((wind_trace.index.month == 2) & (wind_trace.index.day == 29))]
        
        # Add backfilled row at the beginning
        if freq is None or freq == '30min':
            new_index = pd.DatetimeIndex([pd.Timestamp(f"{start_year}-01-01 00:00:00")])
            new_row = pd.DataFrame(
                np.nan,
                index=new_index,
                columns=wind_trace.columns
            )
            
            # Concatenate and backfill
            wind_trace = pd.concat([new_row, wind_trace])
            wind_trace = wind_trace.sort_index()
            wind_trace = wind_trace.bfill()
        
    return wind_trace

def calculate_investment_period_weightings(end_year, investment_period_years, discount_rate):
    """
    Calculates investment period weightings for PyPSA multi-investment-period modeling.

    Parameters:
        end_year (int): The final year of the model horizon
        investment_period_years (list): A list of investment period start years (e.g., [2025, 2030, 2045] or [2025])
        discount_rate (float): Discount rate (e.g., 0.01 for 1%)

    Returns:
        pd.DataFrame: DataFrame with 'objective' and 'years' columns indexed by period start year.
    """
    
    # Create DataFrame with investment periods as index
    df = pd.DataFrame(index=pd.Index(investment_period_years, name='period'))
    
    # Handle single investment period case
    if len(investment_period_years) == 1:
        # Single period: all years from start to end belong to this period
        years_in_periods = [end_year - investment_period_years[0]]
    else:
        # Multiple periods: calculate years in each period using np.diff for intermediate periods
        # and adding the final period length
        years_diff = list(np.diff(investment_period_years))
        final_period_length = end_year - investment_period_years[-1]
        years_in_periods = years_diff + [final_period_length]
    
    df["years"] = years_in_periods
    
    # Calculate objective weights using cumulative time indexing
    r = discount_rate
    T = 0
    for period, nyears in df.years.items():
        discounts = [(1 / (1 + r) ** t) for t in range(T, T + nyears)]
        df.at[period, "objective"] = sum(discounts)
        T += nyears
    df = df[["objective", "years"]]
    
    return df

def calc_snapshot_weightings(
    n: pypsa.Network,
    ) -> pd.DataFrame:
    snapshot_length = len(n.snapshots)
    weighting = 8760 / (snapshot_length) 
    n.snapshot_weightings.loc[:, :] = weighting

def calc_custom_degradation(
    network_snapshots,      
    technology: str,
    build_year: int,
    annual_degradation: Optional[pd.Series] = None,
    annual_degradation_rate: Optional[float] = None,
    lifetime: int = None,
    initial_max_capacity: Optional[float] = 1.0,  # Initial capacity of the dispatchable unit
    min_capacity: Optional[float] = None,  # Minimum capacity limit
) -> pd.Series:
    """
    Calculate custom degradation for a technology over its lifetime.
    
    Parameters:
    -----------
    network_snapshots : pd.Index
        The network snapshots (datetime index or MultiIndex)
    technology : str
        Technology name for the series naming
    build_year : int
        Year when the technology is built
    annual_degradation : pd.Series, optional
        Series with degradation factors as values and either:
        - Years as index (e.g., [2025, 2026, 2027, ...])
        - Sequential index starting from 0 (e.g., [0, 1, 2, ...] for years since build)
    annual_degradation_rate : float, optional
        Fixed annual degradation rate (alternative to annual_degradation series)
    lifetime : int
        Technology lifetime in years
    initial_max_capacity : float, default 1.0
        Initial capacity of the dispatchable unit
    min_capacity : float, optional
        Minimum capacity limit
        
    Returns:
    --------
    pd.Series
        Series with degradation factors applied over the network snapshots
    """
    if annual_degradation is None and annual_degradation_rate is None:
        raise ValueError("Either annual_degradation series or annual_degradation_rate must be provided")
    
    if annual_degradation is not None and annual_degradation_rate is not None:
        raise ValueError("Provide either annual_degradation series OR annual_degradation_rate, not both")
    
    # Create the Series with explicit float dtype
    custom_trace = pd.Series(0.0, index=network_snapshots, 
                             name=f"{technology}_{build_year}", dtype=float)
    
    # Handle MultiIndex snapshots
    if isinstance(network_snapshots, pd.MultiIndex):
        # Extract years from the 'timestep' level
        years = network_snapshots.get_level_values('timestep').year
    else:
        # Single index - extract years directly
        years = network_snapshots.year
    
    # Set values to 0 for years before build year
    before_build_year = years < build_year
    custom_trace[before_build_year] = 0.0
    
    # Set values to 0 for years after lifetime
    if lifetime is not None:
        post_lifetime = years >= build_year + lifetime
        custom_trace[post_lifetime] = 0.0
    
    # Apply degradation for active years
    end_year = build_year + lifetime if lifetime is not None else years.max()
    
    if annual_degradation is not None:
        # Check if index represents absolute years or years since build
        if annual_degradation.index.min() >= build_year:
            # Index represents absolute years
            year_to_degradation = annual_degradation.to_dict()
        else:
            # Index represents years since build (0, 1, 2, ...)
            year_to_degradation = {build_year + i: val for i, val in annual_degradation.items()}
        
        # Apply degradation values directly to each year
        for year in range(build_year, end_year + 1):
            year_mask = years == year
            if year_mask.any():
                if year in year_to_degradation:
                    degradation_factor = year_to_degradation[year]
                else:
                    # If year not in series, use last available value or 1.0
                    if len(annual_degradation) > 0:
                        degradation_factor = annual_degradation.iloc[-1]
                    else:
                        degradation_factor = 1.0
                
                # Apply minimum capacity constraint if specified
                if min_capacity is not None:
                    degradation_factor = max(degradation_factor, min_capacity)
                
                custom_trace[year_mask] = degradation_factor
    else:
        # Use fixed degradation rate
        for year in range(build_year, end_year + 1):
            year_mask = years == year
            if year_mask.any():
                years_since_build = year - build_year
                degradation_factor = (1 - annual_degradation_rate) ** years_since_build
                
                if min_capacity is not None:
                    degradation_factor = max(degradation_factor, min_capacity)
                
                custom_trace[year_mask] = degradation_factor
    
    return custom_trace

def plot_load_profile(n, title="Average Load per Investment Period", start_date: str = None, end_date: str = None, freq: str = None):
    """
    Plot the average load profile per investment period using Plotly Graph Objects
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network with load data
    title : str
        Title for the plot
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object that can be displayed
    """
    # Handle MultiIndex by keeping only the timestep level
    load_df = n.loads_t.p_set.copy()
    if isinstance(load_df.index, pd.MultiIndex):
        # Reset index to only keep timestep level
        load_df.index = load_df.index.get_level_values('timestep')
    
    if start_date is None:
        start_date = load_df.index.min()
    if end_date is None:
        end_date = load_df.index.max()
    load_df = load_df.loc[start_date:end_date]
    
    if freq is not None:
        load_df = load_df.resample(freq).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each load
    for column in load_df.columns:
        fig.add_trace(
            go.Scatter(
                x=load_df.index,
                y=load_df[column],
                mode='lines',
                name=column
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title="MW",
        #height=300,
        #margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig

def plot_vre_gen_profiles(trace_df, title="VRE Generation Profiles", start_date: str = None, end_date: str = None, freq=None):
    
    # Handle MultiIndex by keeping only the timestep level
    if isinstance(trace_df.index, pd.MultiIndex):
        # Reset index to only keep timestep level
        trace_df = trace_df.copy()
        trace_df.index = trace_df.index.get_level_values('timestep')
    
    if start_date is None:
        start_date = trace_df.index.min()
    if end_date is None:
        end_date = trace_df.index.max()
    traces = trace_df.loc[start_date:end_date]
    
    if freq is not None:
        traces = traces.resample(freq).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each generator
    for column in traces.columns:
        fig.add_trace(
            go.Scatter(
                x=traces.index,
                y=traces[column] * 100,
                mode='lines',
                name=column
            )
        )
    fig.update_yaxes(
        range = [0, 100]
    )
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Datetime",
        yaxis_title="Max Output / Capacity (%)",
        #height=300,
        #margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig

def add_dispatchable_generators(
    n, 
    name,
    carrier,
    bus="electricity",
    p_nom=None,
    p_nom_extendable=True,
    marginal_cost=None,
    capital_cost=None,
    build_years=None,
    lifetime=None,
    committable=False,
    start_up_cost=None,
    shut_down_cost=None,
    stand_by_cost=None,
    min_up_time=None,
    min_down_time=None,
    up_time_before=None,
    down_time_before=None,
    ramp_limit_up=None,
    ramp_limit_down=None,
    ramp_limit_start_up=None,
    ramp_limit_shut_down=None,
    p_min_pu=None,
    p_max_pu=None,
    efficiency=None,
    overwrite=True
):
    """
    Add dispatchable generators to a PyPSA network, handling both multi-investment period and 
    single period setups with optional committable unit parameters.
    
    Parameters:
    -----------
    n : pysa.Network
        The PyPSA network to add generators to
    name : str
        Base name for the generator(s)
    carrier : str
        The carrier type (e.g., "GAS_RECIP", "COAL", "CCGT")
    bus : str, default "electricity"
        The bus to connect the generator to
    p_nom : float or None, default None
        Fixed nominal capacity (MW). If None, capacity will be optimized.
    p_nom_extendable : bool, default True
        Whether the capacity should be optimized
    marginal_cost : float, pd.Series, or dict, default None
        Marginal cost of generation. Can be:
        - Single value for time-invariant cost
        - pd.Series with MultiIndex matching n.snapshots for time-varying costs
        - Dict mapping years to costs for multi-investment periods
    capital_cost : float, dict, or None, default None
        Capital cost in currency/MW. Can be a dict mapping years to costs.
    build_years : list, default None
        Years to add generators in multi-investment period setup.
        If None, uses n.investment_periods if available, otherwise adds a single generator.
    lifetime : int or None, default None
        Generator lifetime in years
    committable : bool, default False
        Whether to model the generator as a committable unit with unit commitment constraints
    min_up_time : float or None, default None
        Minimum time a unit must stay online once started (hours)
    min_down_time : float or None, default None
        Minimum time a unit must stay offline once shut down (hours)  
    ramp_limit_up : float or None, default None
        Maximum upward ramp as fraction of p_nom per hour
    ramp_limit_down : float or None, default None
        Maximum downward ramp as fraction of p_nom per hour
    p_min_pu : float or None, default None
        Minimum output as fraction of p_nom when online
    efficiency : float or None, default None
        Efficiency of the generator (between 0 and 1)
    overwrite : bool, default True
        Whether to overwrite existing generators with the same name
        
    Returns:
    --------
    list
        Names of the generators that were added
    """
    # Handle multi-period setup
    is_multi_period = hasattr(n, 'investment_periods') and len(n.investment_periods) > 1
    
    if build_years is None:
        if is_multi_period:
            build_years = n.investment_periods
        else:
            build_years = [None]  # Single generator
    
    # Filter build years based on lifetime constraint
    if lifetime is not None and len(build_years) >= 1 and build_years[0] is not None:
        filtered_build_years = [build_years[0]]  # Always include the first build year
        first_build_year = build_years[0]
        
        for build_year in build_years[1:]:
            # Only add if build_year > first_build_year + lifetime -1
            if build_year > first_build_year + lifetime -1:
                filtered_build_years.append(build_year)
                first_build_year = build_year  # Update reference for next iteration
        
        build_years = filtered_build_years
    
    for i, build_year in enumerate(build_years):
        # Create generator name
        gen_name = f"{name}_{build_year}" if build_year is not None else name
        
        # Handle costs based on build year
        gen_capital_cost = capital_cost
        if isinstance(capital_cost, dict) and build_year is not None:
            gen_capital_cost = capital_cost.get(build_year, list(capital_cost.values())[0])
            
        gen_marginal_cost = marginal_cost
        if isinstance(marginal_cost, dict) and build_year is not None:
            gen_marginal_cost = marginal_cost.get(build_year)
        
        # Base generator parameters
        gen_params = {
            "bus": bus,
            "carrier": carrier,
            "p_nom": 0 if p_nom_extendable else p_nom,
            "p_nom_extendable": p_nom_extendable,
            "capital_cost": gen_capital_cost,
            "marginal_cost": gen_marginal_cost,
            "overwrite": overwrite
        }
        
        # Add build year and lifetime if multi-period or specified
        if build_year is not None:
            gen_params["build_year"] = build_year
        if lifetime is not None:
            gen_params["lifetime"] = lifetime
        if efficiency is not None:
            gen_params["efficiency"] = efficiency
        
        # Add committable unit parameters if needed
        if committable:
            gen_params["committable"] = True
            
            # Add optional unit commitment parameters if provided
            if start_up_cost is not None:
                gen_params["start_up_cost"] = start_up_cost
            if shut_down_cost is not None:
                gen_params["shut_down_cost"] = shut_down_cost
            if stand_by_cost is not None:
                gen_params["stand_by_cost"] = stand_by_cost
            if up_time_before is not None:
                gen_params["up_time_before"] = up_time_before
            if down_time_before is not None:
                gen_params["down_time_before"] = down_time_before
            if p_min_pu is not None:
                gen_params["p_min_pu"] = p_min_pu
            if p_max_pu is not None:
                gen_params["p_max_pu"] = p_max_pu
            if min_up_time is not None:
                gen_params["min_up_time"] = min_up_time
            if min_down_time is not None:
                gen_params["min_down_time"] = min_down_time
            if ramp_limit_up is not None:
                gen_params["ramp_limit_up"] = ramp_limit_up
            if ramp_limit_down is not None:
                gen_params["ramp_limit_down"] = ramp_limit_down
            if ramp_limit_start_up is not None:
                gen_params["ramp_limit_start_up"] = ramp_limit_start_up
            if ramp_limit_shut_down is not None:
                gen_params["ramp_limit_shut_down"] = ramp_limit_shut_down
        
        # Add the generator to the network
        n.add("Generator", gen_name, **gen_params)
        
def add_vre_generators(
    n,
    name,
    carrier,
    p_min_pu=None,
    p_max_pu=None,
    bus="electricity",
    p_nom=None,
    p_nom_extendable=True,
    capital_cost=None,
    marginal_cost=0,
    build_years=None,
    lifetime=None,
    overwrite=True
):
    """
    Add variable renewable energy (VRE) generators to a PyPSA network.
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network to add generators to
    name : str
        Base name for the generator(s)
    carrier : str
        The carrier type (e.g., "WIND", "SOLAR", "CSP")
    p_max_pu : pd.Series or None, default None
        Time series of per-unit availability, assumed to be pre-processed with
        appropriate values for each period
    bus : str, default "electricity"
        The bus to connect the generator to
    p_nom : float or None, default None
        Fixed nominal capacity (MW). If None, capacity will be optimized.
    p_nom_extendable : bool, default True
        Whether the capacity should be optimized
    capital_cost : float, dict, or None, default None
        Capital cost in currency/MW. Can be a dict mapping years to costs.
    marginal_cost : float, default 0
        Marginal cost of generation (typically near-zero for renewables)
    build_years : list, default None
        Years to add generators in multi-investment period setup.
        If None, uses n.investment_periods if available, otherwise adds a single generator.
    lifetime : int or None, default None
        Generator lifetime in years
    p_max_pu_scaling : float, default 1.0
        Scaling factor to apply to the p_max_pu time series
    overwrite : bool, default True
        Whether to overwrite existing generators with the same name
        
    Returns:
    --------
    list
        Names of the generators that were added
    """
    # Handle multi-period setup
    is_multi_period = hasattr(n, 'investment_periods') and len(n.investment_periods) > 1
    
    if build_years is None:
        if is_multi_period:
            build_years = n.investment_periods
        else:
            build_years = [None]  # Single generator
    
    # Filter build years based on lifetime constraint
    if lifetime is not None and len(build_years) > 1 and build_years[0] is not None:
        filtered_build_years = [build_years[0]]  # Always include the first build year
        first_build_year = build_years[0]
        
        for build_year in build_years[1:]:
            # Only add if build_year > first_build_year + lifetime - 1
            if build_year > first_build_year + lifetime - 1:
                filtered_build_years.append(build_year)
                first_build_year = build_year  # Update reference for next iteration
        
        build_years = filtered_build_years
    
    # Create a list to track added generator names
    added_generators = []
    
    for i, build_year in enumerate(build_years):
        # Create generator name
        gen_name = f"{name}_{build_year}" if build_year is not None else name
        
        # Handle costs based on build year
        gen_capital_cost = capital_cost
        if isinstance(capital_cost, dict) and build_year is not None:
            gen_capital_cost = capital_cost.get(build_year, list(capital_cost.values())[0])
        
        # Base generator parameters
        gen_params = {
            "bus": bus,
            "carrier": carrier,
            "p_min_pu": p_min_pu if p_max_pu is not None else 0,
            "p_max_pu": p_max_pu if p_max_pu is not None else 1.0,  # Default to 100% availability
            "p_nom": 0 if p_nom_extendable else p_nom,
            "p_nom_extendable": p_nom_extendable,
            "capital_cost": gen_capital_cost,
            "marginal_cost": marginal_cost,
            "overwrite": overwrite
        }
        
        # Add build year and lifetime if multi-period or specified
        if build_year is not None:
            gen_params["build_year"] = build_year
        if lifetime is not None:
            gen_params["lifetime"] = lifetime
            
        # Add the generator to the network
        n.add("Generator", gen_name, **gen_params)
        added_generators.append(gen_name)            
    
    return added_generators

def add_multiple_storage_units(
    n: pypsa.Network,
    storage_configs: dict[str, dict],
    p_min_pu: pd.Series | None = None,
    p_max_pu: pd.Series | None = None,
    bus: str = "electricity",
    p_nom_extendable: bool = True,
    marginal_cost: float = 0,
    efficiency_store: float = 0.92,
    efficiency_dispatch: float = 0.92,
    build_years: list[int] | None = None,
    lifetime: int | None = None,
    overwrite: bool = True
) -> dict[str, list[str]]:
    """
    Add multiple storage units to a PyPSA network with different configurations.
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network to add storage units to
    storage_configs : dict[str, dict]
        Dictionary where keys are storage names and values are dicts containing:
        - 'max_hours': Required - storage duration in hours
        - 'capital_cost': Required - capital cost (can be float or dict for multi-period)
        - 'carrier': Required - carrier type for this storage unit
        - Any other storage-specific parameters to override defaults
    p_min_pu, p_max_pu : pd.Series or None
        Default time series for all storage units (can be overridden per unit)
    bus : str, default "electricity"
        Default bus for all storage units
    p_nom_extendable : bool, default True
        Default setting for capacity optimization
    marginal_cost : float, default 0
        Default marginal cost for all storage units
    efficiency_store : float, default 0.92
        Default charging efficiency
    efficiency_dispatch : float, default 0.92
        Default discharging efficiency
    build_years : list[int] or None, default None
        Default build years (can be overridden per storage unit)
    lifetime : int or None, default None
        Default lifetime (can be overridden per storage unit)
    overwrite : bool, default True
        Whether to overwrite existing storage units
        
    Returns:
    --------
    dict[str, list[str]]
        Dictionary mapping storage config names to lists of added storage unit names
        
    Example:
    --------
    storage_configs = {
        'BESS_1HR': {
            'max_hours': 1,
            'capital_cost': bess_1hr_capex,
            'carrier': 'BESS'
        },
        'BESS_4HR': {
            'max_hours': 4,
            'capital_cost': bess_4hr_capex,
            'carrier': 'BESS'
        },
        'BESS_8HR': {
            'max_hours': 8,
            'capital_cost': bess_8hr_capex,
            'carrier': 'BESS',
            'lifetime': 15  # Override default lifetime for this unit
        }
    }
    
    added_units = add_multiple_storage_units(
        n=n,
        storage_configs=storage_configs,
        p_max_pu=BESS_2025,
        build_years=[2025, 2030, 2035],
        lifetime=11
    )
    """
    # Handle multi-period setup
    is_multi_period = hasattr(n, 'investment_periods') and len(n.investment_periods) > 1
    
    if build_years is None:
        if is_multi_period:
            build_years = n.investment_periods
        else:
            build_years = [None]
    
    # Dictionary to track all added storage units
    all_added_units: dict[str, list[str]] = {}
    
    # Process each storage configuration
    for storage_name, config in storage_configs.items():
        # Extract required parameters
        required_params = ['max_hours', 'capital_cost', 'carrier']
        for param in required_params:
            if param not in config:
                raise ValueError(f"'{param}' is required for storage config '{storage_name}'")
        
        max_hours = config['max_hours']
        capital_cost = config['capital_cost']
        carrier = config['carrier']
        
        # Get storage-specific parameters or use defaults
        storage_build_years = config.get('build_years', build_years)
        storage_lifetime = config.get('lifetime', lifetime)
        storage_bus = config.get('bus', bus)
        storage_p_min_pu = config.get('p_min_pu', p_min_pu)
        storage_p_max_pu = config.get('p_max_pu', p_max_pu)
        storage_p_nom_extendable = config.get('p_nom_extendable', p_nom_extendable)
        storage_marginal_cost = config.get('marginal_cost', marginal_cost)
        storage_efficiency_store = config.get('efficiency_store', efficiency_store)
        storage_efficiency_dispatch = config.get('efficiency_dispatch', efficiency_dispatch)
        storage_overwrite = config.get('overwrite', overwrite)
        
        # Filter build years based on lifetime constraint
        if storage_lifetime is not None and len(storage_build_years) > 1 and storage_build_years[0] is not None:
            filtered_build_years = [storage_build_years[0]]  # Always include the first build year
            first_build_year = storage_build_years[0]
            
            for build_year in storage_build_years[1:]:
                # Only add if build_year >= first_build_year + lifetime - 1
                if build_year > first_build_year + storage_lifetime - 1:
                    filtered_build_years.append(build_year)
                    first_build_year = build_year  # Update reference for next iteration
            
            storage_build_years = filtered_build_years
        
        # Create list to track added units for this configuration
        config_added_units: list[str] = []
        
        # Add storage units for each build year
        for i, build_year in enumerate(storage_build_years):
            # Create storage unit name
            unit_name = f"{storage_name}_{build_year}" if build_year is not None else storage_name
            
            # Handle costs based on build year
            unit_capital_cost = capital_cost
            if isinstance(capital_cost, dict) and build_year is not None:
                unit_capital_cost = capital_cost.get(build_year, list(capital_cost.values())[0])
            
            # Base storage unit parameters
            storage_params = {
                "bus": storage_bus,
                "carrier": carrier,
                "max_hours": max_hours,
                "p_nom": 0 if storage_p_nom_extendable else config.get('p_nom'),
                "p_nom_extendable": storage_p_nom_extendable,
                "capital_cost": unit_capital_cost,
                "marginal_cost": storage_marginal_cost,
                "efficiency_store": storage_efficiency_store,
                "efficiency_dispatch": storage_efficiency_dispatch,
                "overwrite": storage_overwrite
            }
            
            # Add optional p_min_pu and p_max_pu if provided
            if storage_p_min_pu is not None:
                storage_params["p_min_pu"] = storage_p_min_pu
            if storage_p_max_pu is not None:
                storage_params["p_max_pu"] = storage_p_max_pu
            
            # Add build year and lifetime if multi-period or specified
            if build_year is not None:
                storage_params["build_year"] = build_year
            if storage_lifetime is not None:
                storage_params["lifetime"] = storage_lifetime
                
            # Add the storage unit to the network
            n.add("StorageUnit", unit_name, **storage_params)
            config_added_units.append(unit_name)
        
        # Store the added units for this configuration
        all_added_units[storage_name] = config_added_units
    
    return all_added_units

def add_link_and_store_bess(
    n, 
    name, 
    store_capex_per_mwh,
    #p_nom=0, 
    #e_nom=0,
    eta_charge=0.92, 
    eta_discharge=0.92,
    #marginal_cost=0.0, 
    min_dod=0.0, 
    max_dod=1.0, 
    #p_nom_extendable=True,
    #e_nom_extendable=True,
    bus="electricity", 
    carrier="BESS",
    build_year=2025,
    lifetime=25,
    ):
    """
    Adds a Store+Link BESS model where cost is applied to energy capacity (Store).
    Duration (max_hours) and power (p_nom) are optimised freely.
    """
    energy_bus = f"{name}_bus"

    # Add internal energy bus
    n.add("Bus", energy_bus, carrier=carrier)

    # Discharging link (BESS → grid)
    n.add("Link", "BESS_discharge",
        bus0=energy_bus,
        bus1=bus,
        carrier=f"{carrier}_discharge",
        #p_nom=p_nom,
        p_nom_extendable=True,
        efficiency=eta_discharge,
        build_year=build_year,
        #marginal_cost=marginal_cost,
        #capital_cost=0.0  # optional: add small power conversion cost here
    )

    # Charging link (grid → BESS)
    n.add("Link", f"BESS_charge",
        bus0=bus,
        bus1=energy_bus,
        carrier=f"{carrier}_charge",
        #p_nom=p_nom,
        p_nom_extendable=True,
        efficiency=eta_charge,
        build_year=build_year,
        #capital_cost=0.0
    )

    # Store = energy capacity only
    n.add("Store", f"{name}_store",
        bus=energy_bus,
        carrier=carrier,
        #e_nom=e_nom,
        e_nom_extendable=True,
        e_cyclic=True,
        #e_min_pu=min_dod,
        #e_max_pu=max_dod,
        capital_cost=store_capex_per_mwh,
        build_year=build_year,
        lifetime=lifetime,
    )

# Results Anaysis Functions ==================================================================

def create_dispatch_plot(n, start_date=None, end_date=None, stack=True, interactive=True, y_range=[-200, 200]):
    """
    Create a plot of the system dispatch for a specific period, including storage charging/discharging.
    Supports both StorageUnit and Store components with DatetimeIndex slicing.
    
    Parameters:
    -----------
    n : pypsa.Network
        The optimized PyPSA network
    start_date : str or pd.Timestamp, optional
        Start date for plotting (e.g., "2026-04-20" or "2026-04-20 12:00")
        If None, uses first snapshot
    end_date : str or pd.Timestamp, optional  
        End date for plotting (e.g., "2026-04-21" or "2026-04-20 23:30")
        If None, uses last snapshot
    stack : bool, default True
        Whether to create a stacked area plot (True) or line plot (False)
    interactive : bool, default True
        Whether to use plotly for interactive plotting (True) or matplotlib (False)
    y_range : list, default [-200, 200]
        Y-axis range in MW
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure object
    """
    # Handle MultiIndex slicing properly
    if isinstance(n.snapshots, pd.MultiIndex):
        # For MultiIndex, we need to filter by the timestep level
        if start_date is not None:
            start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
            start_mask = n.snapshots.get_level_values('timestep') >= start_date
        else:
            start_mask = pd.Series(True, index=n.snapshots)
            
        if end_date is not None:
            end_date = pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
            end_mask = n.snapshots.get_level_values('timestep') <= end_date
        else:
            end_mask = pd.Series(True, index=n.snapshots)
            
        # Combine masks and filter snapshots
        time_mask = start_mask & end_mask
        filtered_snapshots = n.snapshots[time_mask]
        
        # Get generation data for filtered snapshots
        generation_data = n.generators_t.p.loc[filtered_snapshots]
        
    else:
        # Single index - use normal slicing
        if start_date is None:
            start_date = n.snapshots[0]
        if end_date is None:
            end_date = n.snapshots[-1]
        
        # Convert to pandas Timestamp if string
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        generation_data = n.generators_t.p.loc[start_date:end_date]
        filtered_snapshots = generation_data.index
    
    # Handle MultiIndex by keeping only the timestep level for plot_data
    if isinstance(generation_data.index, pd.MultiIndex):
        plot_index = generation_data.index.get_level_values('timestep')
    else:
        plot_index = generation_data.index
    
    plot_data = pd.DataFrame(index=plot_index)
    
    # Group generation by carrier
    p_by_carrier = generation_data.T.groupby(n.generators.carrier).sum().T
    
    # Handle MultiIndex for p_by_carrier if needed
    if isinstance(p_by_carrier.index, pd.MultiIndex):
        p_by_carrier.index = p_by_carrier.index.get_level_values('timestep')
    
    # Add generation data by carrier
    for carrier in p_by_carrier.columns:
        plot_data[carrier] = p_by_carrier[carrier]
    
    # Add StorageUnit data if present - group by carrier (simplified)
    if not n.storage_units.empty:
        storage_data = n.storage_units_t.p.loc[filtered_snapshots].T.groupby(n.storage_units.carrier).sum().T
        
        # Handle MultiIndex for storage_data
        if isinstance(storage_data.index, pd.MultiIndex):
            storage_data.index = storage_data.index.get_level_values('timestep')
        
        for carrier in storage_data.columns:
            # Split into positive (discharge) and negative (charge)
            discharge = storage_data[carrier].clip(lower=0)
            charge = storage_data[carrier].clip(upper=0)
            
            if (discharge > 0).any():
                plot_data[carrier] = discharge
                
            if (charge < 0).any():
                plot_data[f"{carrier} (charging)"] = charge
    
    # Add Store data if present
    if not n.stores.empty and hasattr(n.stores_t, "p"):
        for store, carrier in n.stores.carrier.items():
            store_p = n.stores_t.p.loc[filtered_snapshots, store]
            
            # Handle MultiIndex for store_p
            if isinstance(store_p.index, pd.MultiIndex):
                store_p.index = store_p.index.get_level_values('timestep')
            
            discharge = store_p.clip(lower=0)  # output to grid
            charge = store_p.clip(upper=0)     # input from grid (neg)

            if (discharge > 0).any():
                if carrier in plot_data.columns:
                    plot_data[carrier] += discharge
                else:
                    plot_data[carrier] = discharge

            if (charge < 0).any():
                charging_name = f"{carrier} (charging)"
                if charging_name in plot_data.columns:
                    plot_data[charging_name] += charge
                else:
                    plot_data[charging_name] = charge
    
    # Add load data
    load_data = n.loads_t.p_set.loc[filtered_snapshots].sum(axis=1)
    # Handle MultiIndex for load_data
    if isinstance(load_data.index, pd.MultiIndex):
        load_data.index = load_data.index.get_level_values('timestep')
    plot_data['Load'] = load_data
    
    if interactive:
        # Separate charging and generation columns for proper stacking
        charge_columns = [col for col in plot_data.columns if 'charging' in col]
        generation_columns = [col for col in plot_data.columns 
                             if col != 'Load' and 'charging' not in col]
        
        fig = go.Figure()
        
        if stack:
            # Create a stacked area chart for generation
            for col in generation_columns:
                # Get color from carriers if available, otherwise use default
                color = n.carriers.color.get(col, "#CCCCCC") if hasattr(n.carriers, 'color') and col in n.carriers.index else None
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data[col],
                        mode='lines',
                        line=dict(width=0),
                        stackgroup='generation',
                        name=col,
                        fillcolor=color,
                        hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>',
                        hoveron='points+fills'
                    )
                )
            
            # Add charging as separate stack group with negative values
            for col in charge_columns:
                base_carrier = col.replace(' (charging)', '')
                color = n.carriers.color.get(base_carrier, "#CCCCCC") if hasattr(n.carriers, 'color') and base_carrier in n.carriers.index else None
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data[col],
                        mode='lines',
                        line=dict(width=0),
                        stackgroup='charging',
                        name=col,
                        fillcolor=color,
                        hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>',
                        hoveron='points+fills'
                    )
                )
            
            # Add load line
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['Load'],
                    name='Load',
                    line=dict(color='black', width=2),
                    hovertemplate='Load: %{y:.1f} MW<extra></extra>'
                )
            )
        else:
            # Create a line chart for all components
            for col in generation_columns + charge_columns + ['Load']:
                line_color = 'black' if col == 'Load' else None
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data[col],
                        mode='lines',
                        name=col,
                        line=dict(color=line_color, width=2 if col == 'Load' else 1),
                        hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=f"Generation and Storage Dispatch",
            xaxis_title="Date/Time",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            yaxis_range=y_range
        )
        
        return fig
    
def create_energy_balance_plot(n, start_date=None, end_date=None, stack=True, y_range=[-200, 200]):
    """
    Create a plot of the system energy balance using PyPSA's energy_balance statistics.
    
    Parameters:
    -----------
    n : pypsa.Network
        The optimized PyPSA network
    start_date : str or pd.Timestamp, optional
        Start date for plotting (e.g., "2026-04-20" or "2026-04-20 12:00")
        If None, uses first snapshot
    end_date : str or pd.Timestamp, optional  
        End date for plotting (e.g., "2026-04-21" or "2026-04-20 23:30")
        If None, uses last snapshot
    stack : bool, default True
        Whether to create a stacked area plot (True) or line plot (False)
    y_range : list, default [-200, 200]
        Y-axis range in MW
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure object
    """
    # Get energy balance data from PyPSA statistics
    energy_balance = n.statistics.energy_balance(aggregate_time=False).T
    
    # Set default date range if not provided
    if start_date is None:
        start_date = n.snapshots[0]
    if end_date is None:
        end_date = n.snapshots[-1]
    
    # Convert to pandas Timestamp if string
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
    
    # Slice the energy balance data for the specified time period
    plot_data = energy_balance.loc[start_date:end_date].copy()
    
    # Separate positive (generation/discharge), negative (charging), and load components
    positive_components = []
    negative_components = []
    load_data = None
    
    for component_tuple in plot_data.columns:
        component_data = plot_data[component_tuple]
        
        # Create a readable name from the tuple
        if isinstance(component_tuple, tuple):
            # Handle multi-index structure: (component_type, carrier, bus)
            component_type, carrier, bus = component_tuple
            if component_type == 'StorageUnit':
                display_name = f"{carrier} Storage"
            elif component_type == 'Generator':
                display_name = carrier
            elif component_type == 'Load':
                display_name = "Load"
                # Store load data separately for line plot
                load_data = abs(component_data)
                continue  # Skip adding to positive/negative components
            else:
                display_name = f"{component_type} {carrier}"
        else:
            display_name = str(component_tuple)
        
        # Check if component has both positive and negative values
        if component_data.min() < 0 and component_data.max() > 0:
            # Split into positive and negative parts
            positive_part = component_data.clip(lower=0)
            negative_part = component_data.clip(upper=0)
            
            if positive_part.sum() > 0:
                positive_components.append((f"{display_name} (discharge)", positive_part, carrier))
            if negative_part.sum() < 0:
                negative_components.append((f"{display_name} (charge)", negative_part, carrier))
        elif component_data.max() > 0:
            # Only positive values
            positive_components.append((display_name, component_data, carrier))
        elif component_data.min() < 0:
            # Only negative values
            negative_components.append((display_name, component_data, carrier))
    
    fig = go.Figure()
    
    if stack:
        # Add positive components (generation/discharge) as stacked area
        for name, data, carrier in positive_components:
            # Get color from carriers if available
            color = None
            if hasattr(n.carriers, 'color'):
                if carrier in n.carriers.color:
                    color = n.carriers.color[carrier]
                elif carrier.upper() in n.carriers.color:
                    color = n.carriers.color[carrier.upper()]
                elif any(carrier.upper() in c.upper() for c in n.carriers.color.index):
                    matching_carrier = next(c for c in n.carriers.color.index if carrier.upper() in c.upper())
                    color = n.carriers.color[matching_carrier]
            
            if color is None:
                color = "#CCCCCC"
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='generation',
                    name=name,
                    fillcolor=color,
                    hovertemplate=f'{name}: %{{y:.1f}} MWh<extra></extra>',
                    hoveron='points+fills'
                )
            )
        
        # Add negative components (load/charging) as separate stack group
        for name, data, carrier in negative_components:
            # Get color from carriers if available
            color = None
            if hasattr(n.carriers, 'color'):
                if carrier in n.carriers.color:
                    color = n.carriers.color[carrier]
                elif carrier.upper() in n.carriers.color:
                    color = n.carriers.color[carrier.upper()]
                elif any(carrier.upper() in c.upper() for c in n.carriers.color.index):
                    matching_carrier = next(c for c in n.carriers.color.index if carrier.upper() in c.upper())
                    color = n.carriers.color[matching_carrier]
            
            if color is None:
                color = "#CCCCCC"
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='consumption',
                    name=name,
                    fillcolor=color,
                    hovertemplate=f'{name}: %{{y:.1f}} MWh<extra></extra>',
                    hoveron='points+fills'
                )
            )
    else:
        # Line plot mode
        for name, data, carrier in positive_components + negative_components:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=name,
                    hovertemplate=f'{name}: %{{y:.1f}} MWh<extra></extra>'
                )
            )
    
    # Add load as a black line (always on top)
    if load_data is not None:
        fig.add_trace(
            go.Scatter(
                x=load_data.index,
                y=load_data.values,
                mode='lines',
                line=dict(color='black', width=2),
                name='Load',
                hovertemplate='Load: %{y:.1f} MW<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=f"System Energy Balance",
        xaxis_title="Date/Time",
        yaxis_title="Energy (MWh)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis_range=y_range
    )
    
    return fig

def create_dispatch_plot_with_curtailment(n, start_date=None, end_date=None, stack=True, y_range=[-200, 200], 
                                          vre_carriers=["SOLAR_PV", "WIND"]):
    """
    Create a plot of the system dispatch for a specific period, including storage charging/discharging
    and showing curtailed VRE energy with lighter shades.
    
    Parameters:
    -----------
    n : pypsa.Network
        The optimized PyPSA network
    start_date : str or pd.Timestamp, optional
        Start date for plotting (e.g., "2026-04-20" or "2026-04-20 12:00")
        If None, uses first snapshot
    end_date : str or pd.Timestamp, optional  
        End date for plotting (e.g., "2026-04-21" or "2026-04-20 23:30")
        If None, uses last snapshot
    stack : bool, default True
        Whether to create a stacked area plot (True) or line plot (False)
    y_range : list, default [-200, 200]
        Y-axis range in MW
    vre_carriers : list, default ["PV", "WIND"]
        List of carriers considered as VRE for curtailment calculation
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure object
    """
    # Handle MultiIndex slicing properly
    if isinstance(n.snapshots, pd.MultiIndex):
        # For MultiIndex, we need to filter by the timestep level
        if start_date is not None:
            start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
            start_mask = n.snapshots.get_level_values('timestep') >= start_date
        else:
            start_mask = pd.Series(True, index=n.snapshots)
            
        if end_date is not None:
            end_date = pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
            end_mask = n.snapshots.get_level_values('timestep') <= end_date
        else:
            end_mask = pd.Series(True, index=n.snapshots)
            
        # Combine masks and filter snapshots
        time_mask = start_mask & end_mask
        filtered_snapshots = n.snapshots[time_mask]
        
        # Get generation data for filtered snapshots
        generation_data = n.generators_t.p.loc[filtered_snapshots]
        
    else:
        # Single index - use normal slicing
        if start_date is None:
            start_date = n.snapshots[0]
        if end_date is None:
            end_date = n.snapshots[-1]
        
        # Convert to pandas Timestamp if string
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        generation_data = n.generators_t.p.loc[start_date:end_date]
        filtered_snapshots = generation_data.index
    
    # Handle MultiIndex by keeping only the timestep level for plot_data
    if isinstance(generation_data.index, pd.MultiIndex):
        plot_index = generation_data.index.get_level_values('timestep')
    else:
        plot_index = generation_data.index
    
    plot_data = pd.DataFrame(index=plot_index)
    
    # Group generation by carrier
    p_by_carrier = generation_data.T.groupby(n.generators.carrier).sum().T
    
    # Handle MultiIndex for p_by_carrier if needed
    if isinstance(p_by_carrier.index, pd.MultiIndex):
        p_by_carrier.index = p_by_carrier.index.get_level_values('timestep')
    
    # Add generation data by carrier
    for carrier in p_by_carrier.columns:
        plot_data[carrier] = p_by_carrier[carrier]
    
    # Calculate curtailment for VRE carriers
    curtailment_data = {}
    
    for carrier in vre_carriers:
        vre_gens = n.generators.index[n.generators.carrier == carrier]
        if len(vre_gens) == 0:
            continue
            
        # Calculate maximum possible generation based on p_max_pu and p_nom_opt
        max_possible = pd.Series(0.0, index=plot_index)
        
        for gen in vre_gens:
            if gen in n.generators_t.p_max_pu:
                # Use time-varying availability profile
                p_max_data = n.generators_t.p_max_pu.loc[filtered_snapshots, gen]
                # Handle MultiIndex for p_max_data
                if isinstance(p_max_data.index, pd.MultiIndex):
                    p_max_data.index = p_max_data.index.get_level_values('timestep')
                max_possible += p_max_data * n.generators.p_nom_opt[gen]
            else:
                # Use a static value (typically 1.0)
                max_possible += n.generators.p_nom_opt[gen]
                
        # Actual dispatched generation
        if carrier in p_by_carrier.columns:
            actual = p_by_carrier[carrier]
            if isinstance(actual.index, pd.MultiIndex):
                actual.index = actual.index.get_level_values('timestep')
        else:
            actual = pd.Series(0.0, index=plot_index)
            
        # Calculate curtailment (max_possible - actual)
        curtailment = (max_possible - actual).clip(lower=0)
        curtailment_data[f"{carrier} (curtailed)"] = curtailment
    
    # Add curtailment data to the plot DataFrame
    for carrier, values in curtailment_data.items():
        plot_data[carrier] = values
    
    # Add StorageUnit data if present - group by carrier (simplified)
    if not n.storage_units.empty:
        storage_data = n.storage_units_t.p.loc[filtered_snapshots].T.groupby(n.storage_units.carrier).sum().T
        
        # Handle MultiIndex for storage_data
        if isinstance(storage_data.index, pd.MultiIndex):
            storage_data.index = storage_data.index.get_level_values('timestep')
        
        for carrier in storage_data.columns:
            # Split into positive (discharge) and negative (charge)
            discharge = storage_data[carrier].clip(lower=0)
            charge = storage_data[carrier].clip(upper=0)
            
            if (discharge > 0).any():
                plot_data[carrier] = discharge
                
            if (charge < 0).any():
                plot_data[f"{carrier} (charging)"] = charge
    
    # Add Store data if present
    if not n.stores.empty and hasattr(n.stores_t, "p"):
        for store, carrier in n.stores.carrier.items():
            store_p = n.stores_t.p.loc[filtered_snapshots, store]
            
            # Handle MultiIndex for store_p
            if isinstance(store_p.index, pd.MultiIndex):
                store_p.index = store_p.index.get_level_values('timestep')
            
            discharge = store_p.clip(lower=0)  # output to grid
            charge = store_p.clip(upper=0)     # input from grid (neg)

            if (discharge > 0).any():
                if carrier in plot_data.columns:
                    plot_data[carrier] += discharge
                else:
                    plot_data[carrier] = discharge

            if (charge < 0).any():
                charging_name = f"{carrier} (charging)"
                if charging_name in plot_data.columns:
                    plot_data[charging_name] += charge
                else:
                    plot_data[charging_name] = charge

    
    # Add load data
    load_data = n.loads_t.p_set.loc[filtered_snapshots].sum(axis=1)
    # Handle MultiIndex for load_data
    if isinstance(load_data.index, pd.MultiIndex):
        load_data.index = load_data.index.get_level_values('timestep')
    plot_data['Load'] = load_data
    
    # Rest of the plotting code remains the same...
    # Separate columns for proper stacking
    curtailment_columns = [col for col in plot_data.columns if 'curtailed' in col]
    charge_columns = [col for col in plot_data.columns if 'charging' in col]
    generation_columns = [col for col in plot_data.columns 
                         if col != 'Load' and 'charging' not in col and 'curtailed' not in col]
    
    fig = go.Figure()
    
    if stack:
        # Create a stacked area chart for generation
        for col in generation_columns:
            # Get color from carriers if available, otherwise use default
            color = n.carriers.color.get(col, "#CCCCCC") if hasattr(n.carriers, 'color') and col in n.carriers.index else None
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='generation',
                    name=col,
                    fillcolor=color,
                    hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>',
                    hoveron='points+fills'
                )
            )
        
        # Add curtailment with lighter colors on top of generation
        for col in curtailment_columns:
            base_carrier = col.replace(' (curtailed)', '')
            # Get the base color and make it lighter
            base_color = n.carriers.color.get(base_carrier, "#CCCCCC") if hasattr(n.carriers, 'color') and base_carrier in n.carriers.index else "#CCCCCC"
            
            rgb = mcolors.hex2color(base_color)
            lighter_color = mcolors.rgb2hex([min(1.0, c * 1.5) for c in rgb])  # Make 50% lighter
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='curtailment',
                    name=col,
                    fillcolor=lighter_color,
                    hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>',
                    hoveron='points+fills'
                )
            )
        
        # Add charging as separate stack group with negative values
        for col in charge_columns:
            base_carrier = col.replace(' (charging)', '')
            color = n.carriers.color.get(base_carrier, "#CCCCCC") if hasattr(n.carriers, 'color') and base_carrier in n.carriers.index else None
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='charging',
                    name=col,
                    fillcolor=color,
                    hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>',
                    hoveron='points+fills'
                )
            )
        
        # Add load line
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['Load'],
                name='Load',
                line=dict(color='black', width=2),
                hovertemplate='Load: %{y:.1f} MW<extra></extra>'
            )
        )
    else:
        # Create a line chart for all components
        for col in generation_columns + curtailment_columns + charge_columns + ['Load']:
            line_color = 'black' if col == 'Load' else None
            line_width = 2 if col == 'Load' else 1
            
            if 'curtailed' in col:
                base_carrier = col.replace(' (curtailed)', '')
                base_color = n.carriers.color.get(base_carrier, "#CCCCCC") if hasattr(n.carriers, 'color') and base_carrier in n.carriers.index else "#CCCCCC"
                rgb = mcolors.hex2color(base_color)
                line_color = mcolors.rgb2hex([min(1.0, c * 1.5) for c in rgb])
                line_dash = 'dash'
            else:
                line_dash = None
                
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    hovertemplate=f'{col}: %{{y:.1f}} MW<extra></extra>'
                )
            )
    
    # Calculate curtailment totals for the title
    total_generation = sum(plot_data[col].sum() for col in generation_columns)
    total_curtailment = sum(plot_data[col].sum() for col in curtailment_columns)
    curtailment_percentage = (total_curtailment / (total_generation + total_curtailment)) * 100 if total_generation + total_curtailment > 0 else 0
    
    # Create title with date range
    start_display = plot_data.index[0] if len(plot_data) > 0 else "N/A"
    end_display = plot_data.index[-1] if len(plot_data) > 0 else "N/A"
    
    if hasattr(start_display, 'date') and hasattr(end_display, 'date'):
        if start_display.date() == end_display.date():
            title_date = start_display.strftime('%Y-%m-%d')
        else:
            title_date = f"{start_display.strftime('%Y-%m-%d')} to {end_display.strftime('%Y-%m-%d')}"
    else:
        title_date = f"{start_display} to {end_display}"
    
    fig.update_layout(
        title=f"Generation and Storage Dispatch ({title_date})<br>Curtailment: {total_curtailment:.1f} MWh ({curtailment_percentage:.1f}%)",
        xaxis_title="Date/Time",
        yaxis_title="Power (MW)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis_range=y_range
    )
    
    return fig

def plot_generator_output_heatmap(n, carrier: str, start_date: str = None, end_date: str = None):
    """
    Plot generator power output as a heatmap for a specific carrier with:
    - Y-axis: Hour of Day (0-23)
    - X-axis: Day of Year (1-365, excluding leap days)
    - Data averaged across multiple years if date range spans multiple years
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network with optimized generators
    carrier : str
        Name of the carrier to plot (e.g., 'GAS', 'SOLAR_PV', 'WIND')
    start_date : str, optional
        Start year or date string (e.g., '2025' or '2025-01-01')
        If None, uses first year in snapshots
    end_date : str, optional
        End year or date string (e.g., '2028' or '2028-12-31')
        If None, uses last year in snapshots
    """
    # Get generator power output data
    generation_data = n.generators_t.p.copy()
    
    if generation_data.empty:
        print("No generator power output data found.")
        return None

    # Handle MultiIndex by keeping only the timestep level
    if isinstance(generation_data.index, pd.MultiIndex):
        generation_data.index = generation_data.index.get_level_values('timestep')

    # Handle date range
    if start_date is None:
        start_year = generation_data.index[0].year
    else:
        start_year = int(start_date.split('-')[0])
    
    if end_date is None:
        end_year = generation_data.index[-1].year
    else:
        end_year = int(end_date.split('-')[0])
    
    # Filter data by year range
    year_mask = (generation_data.index.year >= start_year) & (generation_data.index.year < end_year)
    generation_data = generation_data[year_mask]

    # Group generators by carrier and sum their output
    carrier_generation = generation_data.T.groupby(n.generators.carrier).sum().T
    
    # Check if the specified carrier exists
    if carrier not in carrier_generation.columns:
        available_carriers = list(carrier_generation.columns)
        print(f"Carrier '{carrier}' not found. Available carriers: {available_carriers}")
        return None
    
    # Get data for the specific carrier
    carrier_data = carrier_generation[carrier]
    
    # Check if carrier has any generation
    if carrier_data.max() == 0:
        print(f"No generation found for carrier '{carrier}' in the specified period.")
        return None

    # Exclude leap days (February 29th)
    is_leap_day = (carrier_data.index.month == 2) & (carrier_data.index.day == 29)
    carrier_data = carrier_data[~is_leap_day]

    # Create a DataFrame with the carrier data and time information
    df = pd.DataFrame({
        'generation': carrier_data,
        'day_of_year': carrier_data.index.dayofyear,
        'hour': carrier_data.index.hour
    })
    
    # Adjust day of year for dates after Feb 28 in leap years to maintain 1-365 range
    leap_year_mask = df.index.to_series().dt.is_leap_year
    after_feb28_in_leap = leap_year_mask & (df.index.month > 2)
    df.loc[after_feb28_in_leap, 'day_of_year'] -= 1

    # Group by day_of_year and hour, take mean across years
    grouped = df.groupby(['day_of_year', 'hour'])['generation'].mean().unstack(level=0)
    
    # Ensure we have all hours (0-23) and all days (1-365)
    full_hours = pd.Index(range(24), name='hour')
    full_days = pd.Index(range(1, 366), name='day_of_year')
    
    heatmap_data = grouped.reindex(index=full_hours, columns=full_days, fill_value=0)

    # Use the same rainbow colorscale as BESS SOC function
    colorscale = [
        [0.0, '#000080'],    # Dark blue (0%)
        [0.2, '#0000FF'],    # Blue (20%)
        [0.4, '#00FFFF'],    # Cyan (40%)  
        [0.5, '#00FF00'],    # Green (50%)
        [0.6, '#FFFF00'],    # Yellow (60%)
        [0.8, '#FF8000'],    # Orange (80%)
        [1.0, '#FF0000']     # Red (100%)
    ]

    # Calculate max value for colorscale
    max_output = heatmap_data.values.max()
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,  # Day of year (1-365)
        y=heatmap_data.index,    # Hour of day (0-23)
        colorscale=colorscale,
        zmin=0,
        zmax=max_output,
        hoverongaps=False,
        hovertemplate='<b>' + carrier + '</b><br>' +
                      'Day: %{x}<br>' +
                      'Hour: %{y}<br>' +
                      'Output: %{z:.1f} MW<br>' +
                      '<extra></extra>',
        colorbar=dict(
            title="Output (MW)",
            tickmode="linear",
            tick0=0,
            dtick=max_output/5 if max_output > 0 else 1,
        )
    ))

    # Update layout
    year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
    title_text = f"{carrier} Generator Output Heatmap ({year_range})"
    if start_year != end_year:
        title_text += f"<br><sub>Data averaged across {end_year - start_year} years</sub>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Day of Year",
        yaxis_title="Hour of Day",
        height=500,
        margin=dict(l=80, r=120, t=100, b=80)
    )

    # Update x and y axes
    fig.update_xaxes(
        dtick=30,  # Show every 30 days
        range=[1, 365],
        showticklabels=True
    )
    
    fig.update_yaxes(
        dtick=4,  # Show every 4 hours
        range=[0, 23],
        showticklabels=True
    )

    return fig

def plot_storage_soc(n, start_date: str = None, end_date: str = None, resample_freq=None):
    """
    Plot state of charge for all StorageUnits and Stores in the PyPSA network.

    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network with optimized storage
    start_date : str, optional
        Start datetime string (e.g., '2026-01-01')
    end_date : str, optional
        End datetime string (e.g., '2026-12-31')
    resample_freq : str, optional
        Time frequency to resample the data. Options: 'D', 'W', 'M', 'Y'
    """
    # Combine StorageUnits and Stores
    soc_data_su = n.storage_units_t.state_of_charge.copy() if not n.storage_units.empty else pd.DataFrame()
    soc_data_store = n.stores_t.e.copy() if not n.stores.empty else pd.DataFrame()

    # Handle MultiIndex by keeping only the timestep level
    if not soc_data_su.empty and isinstance(soc_data_su.index, pd.MultiIndex):
        soc_data_su.index = soc_data_su.index.get_level_values('timestep')
    
    if not soc_data_store.empty and isinstance(soc_data_store.index, pd.MultiIndex):
        soc_data_store.index = soc_data_store.index.get_level_values('timestep')

    # Combine and label types
    su_labels = [f"{col} [SU]" for col in soc_data_su.columns]
    store_labels = [f"{col} [Store]" for col in soc_data_store.columns]
    soc_data_su.columns = su_labels
    soc_data_store.columns = store_labels
    soc_data = pd.concat([soc_data_su, soc_data_store], axis=1)

    if soc_data.empty:
        print("No storage units or stores with state of charge found.")
        return None

    # Clip to date range
    if start_date is None:
        start_date = soc_data.index[0]
    if end_date is None:
        end_date = soc_data.index[-1]
    soc_data = soc_data.loc[start_date:end_date]

    # Remove empty columns
    non_zero_columns = [col for col in soc_data.columns if soc_data[col].max() > 0]
    soc_data = soc_data[non_zero_columns]

    # Get max theoretical SoC
    max_theoretical_soc = {}
    for col in soc_data.columns:
        name, typ = col.split(' [')
        if 'SU' in typ:
            p_nom_opt = n.storage_units.at[name, "p_nom_opt"]
            max_hours = n.storage_units.at[name, "max_hours"]
            max_theoretical_soc[col] = p_nom_opt * max_hours
        elif 'Store' in typ:
            if n.stores.at[name, "e_nom_extendable"]:
                max_theoretical_soc[col] = n.stores.at[name, "e_nom_opt"]
            else:
                max_theoretical_soc[col] = n.stores.at[name, "e_nom"]

    # Resample
    if resample_freq:
        soc_data = soc_data.resample(resample_freq).mean()

    # Plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, col in enumerate(soc_data.columns):
        color = colors[i % len(colors)]
        soc_percentage = soc_data[col] / max_theoretical_soc[col] * 100

        fig.add_trace(go.Scatter(x=soc_data.index, y=soc_data[col],
                                 name=f"{col} (MWh)",
                                 line=dict(color=color)), secondary_y=False)
        fig.add_trace(go.Scatter(x=soc_data.index, y=soc_percentage,
                                 name=f"{col} (%)",
                                 line=dict(color=color, dash='dot'),
                                 visible='legendonly'), secondary_y=True)

    fig.update_layout(
        title="State of Charge for StorageUnits and Stores",
        yaxis_title="State of Charge (MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(b=150)
    )

    fig.update_yaxes(title_text="State of Charge (MWh)", secondary_y=False)
    fig.update_yaxes(title_text="State of Charge (%)", secondary_y=True, range=[0, 100])
    fig.update_xaxes(title_text="Time")

    return fig

def plot_storage_soc_heatmap(n, start_date: str = None, end_date: str = None):
    """
    Plot state of charge for all StorageUnits and Stores as a heatmap with:
    - Y-axis: Hour of Day (0-23)
    - X-axis: Day of Year (1-365, excluding leap days)
    - Data averaged across multiple years if date range spans multiple years
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network with optimized storage
    start_date : str, optional
        Start year or date string (e.g., '2025' or '2025-01-01')
        If None, uses first year in snapshots
    end_date : str, optional
        End year or date string (e.g., '2028' or '2028-12-31')
        If None, uses last year in snapshots
    """
    
    # Combine StorageUnits and Stores
    soc_data_su = n.storage_units_t.state_of_charge.copy() if not n.storage_units.empty else pd.DataFrame()
    soc_data_store = n.stores_t.e.copy() if not n.stores.empty else pd.DataFrame()

    # Handle MultiIndex by keeping only the timestep level
    if not soc_data_su.empty and isinstance(soc_data_su.index, pd.MultiIndex):
        soc_data_su.index = soc_data_su.index.get_level_values('timestep')
    
    if not soc_data_store.empty and isinstance(soc_data_store.index, pd.MultiIndex):
        soc_data_store.index = soc_data_store.index.get_level_values('timestep')

    # Combine and label types
    su_labels = [f"{col} [SU]" for col in soc_data_su.columns]
    store_labels = [f"{col} [Store]" for col in soc_data_store.columns]
    soc_data_su.columns = su_labels
    soc_data_store.columns = store_labels
    soc_data = pd.concat([soc_data_su, soc_data_store], axis=1)

    if soc_data.empty:
        print("No storage units or stores with state of charge found.")
        return None

    # Handle date range
    if start_date is None:
        start_year = soc_data.index[0].year
    else:
        start_year = int(start_date.split('-')[0])
    
    if end_date is None:
        end_year = soc_data.index[-1].year
    else:
        end_year = int(end_date.split('-')[0])
    
    # Filter data by year range
    year_mask = (soc_data.index.year >= start_year) & (soc_data.index.year < end_year)
    soc_data = soc_data[year_mask]

    # Remove empty columns
    non_zero_columns = [col for col in soc_data.columns if soc_data[col].max() > 0]
    soc_data = soc_data[non_zero_columns]

    if soc_data.empty:
        print("No storage units with non-zero state of charge found in the specified period.")
        return None

    # Get max theoretical SoC for percentage calculation
    max_theoretical_soc = {}
    for col in soc_data.columns:
        name, typ = col.split(' [')
        if 'SU' in typ:
            p_nom_opt = n.storage_units.at[name, "p_nom_opt"]
            max_hours = n.storage_units.at[name, "max_hours"]
            max_theoretical_soc[col] = p_nom_opt * max_hours
        elif 'Store' in typ:
            if n.stores.at[name, "e_nom_extendable"]:
                max_theoretical_soc[col] = n.stores.at[name, "e_nom_opt"]
            else:
                max_theoretical_soc[col] = n.stores.at[name, "e_nom"]

    # Convert to percentage of maximum capacity
    soc_percentage = pd.DataFrame(index=soc_data.index)
    for col in soc_data.columns:
        if max_theoretical_soc[col] > 0:
            soc_percentage[col] = (soc_data[col] / max_theoretical_soc[col]) * 100
        else:
            soc_percentage[col] = 0

    # Exclude leap days (February 29th)
    is_leap_day = (soc_percentage.index.month == 2) & (soc_percentage.index.day == 29)
    soc_percentage = soc_percentage[~is_leap_day]

    # Add day of year and hour columns
    soc_percentage['day_of_year'] = soc_percentage.index.dayofyear
    soc_percentage['hour'] = soc_percentage.index.hour
    
    # Adjust day of year for dates after Feb 28 in leap years to maintain 1-365 range
    leap_year_mask = soc_percentage.index.to_series().dt.is_leap_year
    after_feb28_in_leap = leap_year_mask & (soc_percentage.index.month > 2)
    soc_percentage.loc[after_feb28_in_leap, 'day_of_year'] -= 1

    # Group by day_of_year and hour, then take the mean across years
    storage_columns = [col for col in soc_percentage.columns if col not in ['day_of_year', 'hour']]
    
    # Create heatmap data for each storage unit
    heatmap_data = {}
    
    for storage_col in storage_columns:
        # Group by day_of_year and hour, take mean across years
        grouped = soc_percentage.groupby(['day_of_year', 'hour'])[storage_col].mean().unstack(level=0)
        
        # Ensure we have all hours (0-23) and all days (1-365)
        full_hours = pd.Index(range(24), name='hour')
        full_days = pd.Index(range(1, 366), name='day_of_year')
        
        grouped = grouped.reindex(index=full_hours, columns=full_days, fill_value=0)
        heatmap_data[storage_col] = grouped
    
    n_storage = len(storage_columns)
    if n_storage == 0:
        print("No storage units with data found.")
        return None
    
    # Calculate subplot layout
    cols = min(2, n_storage)  # Max 2 columns
    rows = (n_storage + cols - 1) // cols  # Ceiling division
    
    subplot_titles = [col.replace(' [SU]', '').replace(' [Store]', '') for col in storage_columns]
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )

    # Create custom colorscale (blue to red through green/yellow)
    colorscale = [
        [0.0, '#000080'],    # Dark blue (0%)
        [0.2, '#0000FF'],    # Blue (20%)
        [0.4, '#00FFFF'],    # Cyan (40%)  
        [0.5, '#00FF00'],    # Green (50%)
        [0.6, '#FFFF00'],    # Yellow (60%)
        [0.8, '#FF8000'],    # Orange (80%)
        [1.0, '#FF0000']     # Red (100%)
    ]

    # Add heatmap for each storage unit
    for i, storage_col in enumerate(storage_columns):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        data = heatmap_data[storage_col]
        
        # Show colorbar only for the last subplot
        show_colorbar = (i == len(storage_columns) - 1)
        
        fig.add_trace(
            go.Heatmap(
                z=data.values,
                x=data.columns,  # Day of year (1-365)
                y=data.index,    # Hour of day (0-23)
                colorscale=colorscale,
                zmin=0,
                zmax=100,
                hoverongaps=False,
                hovertemplate='<b>' + storage_col.replace(' [SU]', '').replace(' [Store]', '') + '</b><br>' +
                              'Day: %{x}<br>' +
                              'Hour: %{y}<br>' +
                              'SOC: %{z:.1f}%<br>' +
                              '<extra></extra>',
                colorbar=dict(
                    title="SOC (%)",
                    tickmode="linear",
                    tick0=0,
                    dtick=20,
                    ticksuffix="%",
                    len=0.8,
                    x=1.02
                ) if show_colorbar else None,
                showscale=show_colorbar
            ),
            row=row, col=col
        )

    # Update layout
    year_range = f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
    title_text = f"BESS State of Charge Heatmap ({year_range})"
    if start_year != end_year:
        title_text += f"<br><sub>Data averaged across {end_year - start_year} years</sub>"
    
    fig.update_layout(
        title=title_text,
        height=max(400, 300 * rows),
        margin=dict(l=80, r=120, t=100, b=80)
    )

    # Update x and y axes for all subplots
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # X-axis (Day of Year) - show labels on all subplots
            fig.update_xaxes(
                title_text="Day of Year",  # Show title on all subplots
                dtick=90,  # Show every 30 days
                range=[1, 365],
                showticklabels=True,
                row=i, col=j
            )
            
            # Y-axis (Hour of Day) - show labels on all subplots
            fig.update_yaxes(
                title_text="Hour of Day",
                dtick=4,
                range=[0, 23],
                showticklabels=True,
                row=i, col=j
            )

    return fig

def plot_monthly_electric_production(
    n: pypsa.Network,
    start_year: int = None,
    end_year: int = None,
):
    if start_year is None:
        start_year = n.snapshots.get_level_values(0)[0]
    if end_year is None:
        end_year = n.snapshots.get_level_values(0)[-1]
    
    energy_balance = n.statistics.energy_balance(aggregate_time=False).groupby(level=1).sum()
    
    generator_carriers = set(n.generators.carrier.unique())
    generator_production = energy_balance[energy_balance.index.isin(generator_carriers)].T
    generator_production_filtered = generator_production.loc[start_year:end_year]
    
    storage_carriers = set(n.storage_units.carrier.unique())
    storage_net_charge = energy_balance[energy_balance.index.isin(storage_carriers)].T
    storage_net_charge_filtered = storage_net_charge.loc[start_year:end_year]

    # Get snapshot weightings and align indices
    weightings = n.snapshot_weightings['generators']
    weightings_filtered = weightings.loc[start_year:end_year]
    
    # Drop 'period' level from weightings to match generator_production_filtered structure
    weightings_aligned = weightings_filtered.droplevel('period')
    
    # Drop 'period' level from generator_production_filtered if it exists
    generator_production_filtered = generator_production_filtered.droplevel('period')
    storage_net_charge_filtered = storage_net_charge_filtered.droplevel('period')
    
    # Multiply by snapshot weightings
    generator_production_weighted = generator_production_filtered.mul(weightings_aligned, axis=0)
    storage_net_charge_weighted = storage_net_charge_filtered.mul(weightings_aligned, axis=0)
    total_production_weighted = pd.concat(
        [generator_production_weighted, storage_net_charge_weighted],
        axis=1,
        join='outer'
    )
    # Resample to monthly sums
    monthly_sum = total_production_weighted.resample('ME').sum()
    monthly_sum['month'] = monthly_sum.index.month
    monthly_sum['month'] = monthly_sum['month']

    # Average over years for each month and carrier
    monthly_avg = monthly_sum.groupby('month').mean()
    monthly_avg.index = monthly_avg.index.map(lambda x: calendar.month_abbr[x])
    
    # Create Plotly bar chart
    fig = go.Figure()
    
    # Add a bar for each carrier
    for carrier in monthly_avg.columns:
        fig.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg[carrier],
            name=carrier,
            marker_color=n.carriers.loc[carrier, 'color'] if carrier in n.carriers.index else None
        ))
    
    fig.update_layout(
        title='Monthly Average Generation by Carrier (Weighted)',
        xaxis_title='Month',
        yaxis_title='Generation (MWh)',
        barmode='relative',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
    )
    
    return fig

def generate_multiperiod_overview(
    network: pypsa.Network, 
    renewable_carriers: list[str] = ["SOLAR_PV", "WIND"],
    thermal_carriers: list[str] | None = None
    ) -> pd.Series:
    """
    Generate comprehensive overview statistics for a multiperiod PyPSA network.
    
    This function provides similar functionality to the single-period generate_overview
    but adapted for multiperiod investment planning models with multiple investment periods.
    
    Parameters:
    -----------
    network : pypsa.Network
        The optimized multiperiod PyPSA network
    threshold : float, default 0.1
        Threshold for excluding components contributing less than threshold EUR/MWh  
    include_lcoe : bool, default True
        Whether to calculate LCOE metrics for storage technologies
    include_storage_metrics : bool, default True
        Whether to include detailed storage utilization metrics
    exclude_carriers_from_curtailment : list[str] | None, default None
        List of carriers to exclude from curtailment calculation. If None, all carriers are included.
        
    Returns:
    --------
    pd.Series
        Comprehensive overview with the following metrics:
        - Average electricity and other bus prices
        - Total system load and cost metrics
        - Component-wise CAPEX, OPEX, TOTEX, and capacities
        - Curtailment rates by technology
        - LCOE by technology
        - Capacity factors (used and available)
        - Storage-specific LCOE metrics (if applicable)
    """
    
    results_overview = pd.Series(dtype=float)

    # === COMPONENT STATISTICS ===
    # Use PyPSA's built-in statistics method for comprehensive analysis
    try:
        stats = network.statistics().groupby(level=1).sum()
        weighting = network.snapshot_weightings['objective'].mean()
        
        # Add component-wise metrics to overview
        metric_mappings = [
            ("CAPEX", "Capital Expenditure"),
            ("OPEX", "Operational Expenditure"), 
            ("CAPACITY", "Optimal Capacity")
        ]
        
        for short_name, full_name in metric_mappings:
            if full_name in stats.columns:
                component_stats = stats[full_name].copy()
                # Apply weighting to OPEX values
                # if short_name == "OPEX":
                #     component_stats = component_stats * weighting
                component_stats = component_stats.rename(lambda x: f"{x} {short_name}")
                results_overview = pd.concat([results_overview, component_stats])
        
        # Calculate totex (capex + weighted opex) for each carrier
        if all(col in stats.columns for col in ["Capital Expenditure", "Operational Expenditure"]):
            totex_stats = (stats["Capital Expenditure"] + stats["Operational Expenditure"])
            totex_series = totex_stats.rename(lambda x: f"{x} TOTEX")
            results_overview = pd.concat([results_overview, totex_series])
                
        # === CURTAILMENT METRICS ===
        if "Curtailment" in stats.columns and "Supply" in stats.columns:
            # Get only carriers that are present in generators
            generator_carriers = set(network.generators.carrier.unique())
            if 'LoadShed' in generator_carriers:
                generator_carriers.discard('LoadShed')
            
            # Apply exclusion filter if provided
            if thermal_carriers:
                generator_carriers = generator_carriers - set(thermal_carriers)
            
            curtailment_rates = (stats["Curtailment"] / (stats["Supply"] + stats["Curtailment"]))
            # Filter to only generator carriers (and apply exclusions)
            curtailment_rates_filtered = curtailment_rates[curtailment_rates.index.isin(generator_carriers)]
            curtailment_series = curtailment_rates_filtered.rename(lambda x: f"{x} CURTAILMENT").dropna()
            results_overview = pd.concat([results_overview, curtailment_series])
        
        # Calculate available capacity factors
        if all(col in stats.columns for col in ["Supply", "Curtailment"]):
            # Get only carriers that are present in generators
            generator_carriers = set(network.generators.carrier.unique())
            if 'LoadShed' in generator_carriers:
                generator_carriers.discard('LoadShed')
            if thermal_carriers:
                thermal_gen = generator_carriers - set(thermal_carriers)
            
            cf_available = ((stats["Supply"] + stats["Curtailment"]) / 
                        (stats["Optimal Capacity"] * 8760)).dropna()
            # Filter to only generator carriers
            cf_available_filtered = cf_available[cf_available.index.isin(thermal_gen)]
            cf_available_renamed = cf_available_filtered.rename(lambda x: f"{x} CF AVAILABLE")

            # Add actual capacity factor
            cf_actual = ((stats["Supply"]) / 
                        (stats["Optimal Capacity"] * 8760)).dropna()
            # Filter to only generator carriers
            cf_actual_filtered = cf_actual[cf_actual.index.isin(generator_carriers)]
            cf_actual_renamed = cf_actual_filtered.rename(lambda x: f"{x} CF ACTUAL")

            # Add all capacity factors as a single concatenation to results_overview
            results_overview = pd.concat([results_overview, cf_available_renamed, cf_actual_renamed])
            
        stats_by_component = network.statistics().groupby(level=0).sum()
            
        # Add rows for annual load from statistics
        annual_load = abs(stats_by_component.loc["Load", "Energy Balance"])
        if "LoadShed" in stats.index:
            annual_load_shed = abs(stats.loc["LoadShed", "Energy Balance"])
            perct_load_shed = (annual_load_shed / annual_load)
            annual_load_shed_row = pd.DataFrame([annual_load_shed.values], columns=annual_load_shed.index, index=['Load Shed'])
            results_overview = pd.concat([results_overview, annual_load_shed_row])
            perct_load_shed_row = pd.DataFrame([perct_load_shed.values], columns=perct_load_shed.index, index=['Load Shed Percentage'])
            results_overview = pd.concat([results_overview, perct_load_shed_row])
        if "Load" in stats_by_component.index and "LoadShed" in stats.index:
            annual_load = annual_load - annual_load_shed
            
        annual_load_row = pd.DataFrame([annual_load.values], columns=annual_load.index, index=['Total Load'])
        results_overview = pd.concat([results_overview, annual_load_row])

        # Calculate % of load supplied by renewables
        if all(x in stats.index for x in renewable_carriers):
            renewable_supply = stats.loc[renewable_carriers, "Energy Balance"].sum()
            perct_supplied_by_renewables = (renewable_supply / annual_load)
            perct_supplied_by_renewables_row = pd.DataFrame([perct_supplied_by_renewables.values], columns=perct_supplied_by_renewables.index, index=['Load Supplied by Renewables'])
            results_overview = pd.concat([results_overview, perct_supplied_by_renewables_row])
        # Calculate % of load supplied by non-renewables
        if all(x in stats.index for x in thermal_carriers):
            non_renewable_supply = stats.loc[thermal_carriers, "Energy Balance"].sum()
            perct_supplied_by_non_renewables = (non_renewable_supply / annual_load)
            perct_supplied_by_non_renewables_row = pd.DataFrame([perct_supplied_by_non_renewables.values], columns=perct_supplied_by_non_renewables.index, index=['Load Supplied by Non-Renewables'])
            results_overview = pd.concat([results_overview, perct_supplied_by_non_renewables_row])

        results_overview_final = results_overview.dropna(axis=0, how='all').dropna(axis=1, how='all')
        results_overview_final = results_overview_final.loc[~(results_overview_final == 0).all(axis=1)].round(2)

    except Exception as e:
        print(f"Warning: Could not calculate statistics: {e}")
    
    # === SYSTEM LCOE ===
    try:
        system_lcoe = calculate_system_lcoe(network)
        print(f"Total system LCOE: ${system_lcoe:.2f}/MWh")
    except Exception as e:
        print(f"Warning: Could not calculate system LCOE: {e}")
    
    return results_overview_final


def calculate_system_lcoe(network: pypsa.Network) -> float:
    """Calculate system-wide LCOE from network statistics."""
    try:
        # Use PyPSA's statistics for accurate calculation
        stats = network.statistics()
        weighting = network.snapshot_weightings['objective'].mean()
        
        # Get total energy consumption (sum of all loads)
        energy_balance = stats.groupby(level=0).sum()
        
        if "Load" in energy_balance.index:
            total_load = abs(energy_balance.loc['Load','Energy Balance'])
        else:
            # Fallback calculation
            total_load = 0
            for load_name in network.loads.index:
                total_load += abs(network.loads_t.p_set[load_name].sum())
        
        # Get total system costs
        component_stats = stats.groupby(level=1).sum()
        
        if all(col in component_stats.columns for col in ["Capital Expenditure", "Operational Expenditure"]):
            total_costs = (component_stats["Capital Expenditure"] + 
                          component_stats["Operational Expenditure"]).sum()
        else:
            return np.nan
        
        if total_load.sum() > 0:
            return total_costs.sum() / total_load.sum()
        else:
            return np.nan
            
    except Exception as e:
        print(f"Warning: System LCOE calculation failed: {e}")
        return np.nan


def calculate_lifetime_emissions(network: pypsa.Network):
    emissions = (
        network.generators_t.p
        / network.generators.efficiency
        * network.generators.carrier.map(network.carriers.co2_emissions)
    )
    total_emissions = network.snapshot_weightings.generators @ emissions.sum(axis=1).div(1e6)  # Mt

    return total_emissions

