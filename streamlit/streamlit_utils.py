import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src.utils_multiperiod import *

def initialize_plot_session_state():
    """Initialize session state variables for storing plots."""
    plot_keys = [
        'dispatch_plot',
        'storage_soc_plot', 
        'generator_heatmap',
        'monthly_electrical_plot',
        'storage_soc_heatmap'
    ]
    
    for key in plot_keys:
        if key not in st.session_state:
            st.session_state[key] = None

def store_plot_in_session(plot_key: str, figure):
    """Store a plot figure in session state."""
    st.session_state[plot_key] = figure

def clear_all_plots():
    """Clear all stored plots from session state."""
    plot_keys = [
        'dispatch_plot',
        'storage_soc_plot', 
        'generator_heatmap',
        'monthly_electrical_plot',
        'storage_soc_heatmap'
    ]
    
    for key in plot_keys:
        if key in st.session_state:
            st.session_state[key] = None

def get_plot_count():
    """Get the number of stored plots."""
    plot_keys = [
        'dispatch_plot',
        'storage_soc_plot', 
        'generator_heatmap',
        'monthly_electrical_plot',
        'storage_soc_heatmap'
    ]
    
    return sum(1 for key in plot_keys if st.session_state.get(key) is not None)

def display_stored_plots():
    """Optional utility to show a summary of stored plots."""
    plot_configs = [
        ('dispatch_plot', 'Dispatch Plot'),
        ('storage_soc_plot', 'Storage State of Charge'),
        ('generator_heatmap', 'Generator Output Heatmap'),
        ('monthly_electrical_plot', 'Monthly Electrical Production'),
        ('storage_soc_heatmap', 'Storage SOC Heatmap')
    ]
    
    # Count how many plots are stored
    stored_plots = [(key, title) for key, title in plot_configs if st.session_state.get(key) is not None]
    
    if stored_plots:
        with st.expander(f"📊 Plot Summary ({len(stored_plots)} stored)", expanded=False):
            st.info("Generated plots are stored in session state and persist across form submissions.")
            
            # Show list of stored plots with clear options
            for plot_key, plot_title in stored_plots:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"✅ {plot_title}")
                with col2:
                    if st.button(f"Clear", key=f"summary_clear_{plot_key}"):
                        st.session_state[plot_key] = None
                        st.rerun()
            
            # Add button to clear all plots
            st.markdown("---")
            if st.button("Clear All Plots", key="summary_clear_all"):
                clear_all_plots()
                st.rerun()

def year_selector(
    label: str,
    min_year: int = 2020,
    max_year: int = 2055,
    default_year: int = 2025,
    key: Optional[str] = None
) -> int:
    """Custom year selector widget."""
    return st.selectbox(
        label,
        options=list(range(min_year, max_year + 1)),
        index=default_year - min_year,
        key=key
    )

def year_range_selector(
    start_label: str = "Start Year",
    end_label: str = "End Year",
    min_year: int = 2020,
    max_year: int = 2055,
    default_start: int = 2025,
    default_end: int = 2035,
    start_key: Optional[str] = None,
    end_key: Optional[str] = None
) -> Tuple[int, int]:
    """Year range selector with validation."""
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = year_selector(
            start_label,
            min_year=min_year,
            max_year=max_year,
            default_year=default_start,
            key=start_key
        )
    
    with col2:
        end_year = year_selector(
            end_label,
            min_year=start_year,
            max_year=max_year,
            default_year=max(default_end, start_year),
            key=end_key
        )
    
    return start_year, end_year

def investment_years_selector(
    start_year: int,
    end_year: int,
    default_years: Optional[List[int]] = None,
    key: Optional[str] = None
) -> List[int]:
    """Multi-select for investment years."""
    
    available_years = list(range(start_year, end_year + 1))
    if default_years is None:
        default_years = available_years
    
    return st.multiselect(
        "Investment Years",
        options=available_years,
        default=[y for y in default_years if y in available_years],
        key=key,
        help="Select years when new investments can be made"
    )

def tech_selector(
    available_techs: List[str],
    default_techs: Optional[List[str]] = None,
    key: Optional[str] = None
) -> List[str]:
    """Technology selector widget."""
    if default_techs is None:
        default_techs = available_techs
    
    return st.multiselect(
        "Technologies to Include",
        options=available_techs,
        default=default_techs,
        key=key
    )

def rez_selector(
    label: str = "REZ ID",
    default: str = "N3",
    key: Optional[str] = None
) -> str:
    """REZ ID selector."""
    from src.config import REZIDS
    
    return st.selectbox(
        label,
        options=[rez.value for rez in REZIDS],
        index=list(REZIDS).index(REZIDS.N3) if default == "N3" else 0,
        key=key
    )

def parameter_input_section():
    """Create parameter input section in sidebar."""
    
    # Timeline parameters
    with st.sidebar.expander("Timeline Parameters", expanded=True):
        start_year, end_year = year_range_selector(
            default_start=2025,
            default_end=2035,
            start_key="start_year",
            end_key="end_year"
        )
        
        investment_years = investment_years_selector(
            start_year, end_year, 
            default_years=[2025, 2028],
            key="investment_years"
        )
        
        discount_rate = st.number_input(
            "Discount Rate (WACC)",
            min_value=0.01,
            max_value=0.20,
            value=0.08,
            step=0.01,
            format="%.3f",
            key="discount_rate"
        )
    
    # Technology parameters
    with st.sidebar.expander("Technology Selection"):
        st.subheader("🔥 Dispatchable Generators")
        
        # Dispatchable generators section
        dispatchable_generators = []
        fuel_types = []
        
        # Number of dispatchable generators to add
        num_dispatchable = st.number_input(
            "Number of Dispatchable Generators",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            key="num_dispatchable"
        )
        
        dispatchable_options = ['GAS_RECIP', 'OCGT_SML', 'OCGT_LRG', 'CCGT', 'CCGT_CCS', 'BIOMASS']
        fuel_options = ['GAS', 'DIESEL', 'BIOMASS']
        
        for i in range(num_dispatchable):
            col1, col2 = st.columns(2)
            with col1:
                disp_gen = st.selectbox(
                    f"Dispatchable Gen {i+1}",
                    options=dispatchable_options,
                    index=0 if i == 0 else 1 if i < len(dispatchable_options) else 0,
                    key=f"dispatchable_gen_{i}"
                )
                if disp_gen:
                    dispatchable_generators.append(disp_gen)
            
            with col2:
                fuel_type = st.selectbox(
                    f"Fuel Type {i+1}",
                    options=fuel_options,
                    index=0,
                    key=f"fuel_type_{i}"
                )
                if fuel_type:
                    fuel_types.append(fuel_type)
        
        st.markdown("---")
        st.subheader("🌞 VRE Generators")
        
        # VRE generators section
        vre_options = ['SOLAR_PV', 'WIND']
        selected_vre = st.multiselect(
            "VRE Technologies",
            options=vre_options,
            default=['SOLAR_PV', 'WIND'],
            key="selected_vre"
        )
        
        st.markdown("---")
        st.subheader("🔋 Battery Energy Storage")
        
        # BESS selection
        bess_options = [f'BESS_{i}HR' for i in [1, 2, 4, 6, 8, 12, 16, 20, 24]]
        selected_bess = st.multiselect(
            "BESS Duration Options",
            options=bess_options,
            default=['BESS_4HR', 'BESS_8HR'],
            key="selected_bess"
        )
        
        # BESS degradation toggle
        bess_degradation_enabled = st.toggle(
            "Enable BESS Degradation",
            value=True,
            help="Include battery degradation over time in the model",
            key="bess_degradation_enabled"
        )
        
        # Combine all selected technologies
        selected_techs = dispatchable_generators + selected_vre + selected_bess
        
    # VRE parameters
    with st.sidebar.expander("VRE Parameters"):
        rez_id = rez_selector(key="rez_id")
        
        reference_year = year_selector(
            "Reference Weather Year",
            min_year=2011,
            max_year=2023,
            default_year=2016,
            key="reference_year"
        )
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Solar")
            solar_type = st.selectbox(
                "Solar Type",
                options=["SAT", "CST"],
                index=0,
                key="solar_type"
            )
            solar_degradation = st.number_input(
                "Solar Degradation (annual)",
                min_value=0.0,
                max_value=0.05,
                value=0.0035,
                step=0.0001,
                format="%.4f",
                key="solar_degradation"
            )
            solar_lifetime = st.number_input(
                "Solar Lifetime (years)",
                min_value=10,
                max_value=50,
                value=25,
                step=1,
                key="solar_lifetime"
            )
        
        with col2:
            st.subheader("Wind")
            wind_type = st.selectbox(
                "Wind Type",
                options=["WH"],
                index=0,
                key="wind_type"
            )
            wind_degradation = st.number_input(
                "Wind Degradation (annual)",
                min_value=0.0,
                max_value=0.05,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="wind_degradation"
            )
            wind_lifetime = st.number_input(
                "Wind Lifetime (years)",
                min_value=10,
                max_value=50,
                value=25,
                step=1,
                key="wind_lifetime"
            )
        
        # Multi-reference year toggle
        multi_reference_year = st.toggle(
            "Use Multi-Reference Year",
            value=True,
            help="Enable to use multiple reference years for weather data generation",
            key="multi_reference_year"
        )
    
    # Load parameters
    with st.sidebar.expander("Load Parameters"):
        load_profile = st.selectbox(
            "Load Profile Type",
            options=["Flat", "Custom"],
            index=0,
            key="load_profile_type"
        )
        
        if load_profile == "Flat":
            flat_load = st.number_input(
                "Flat Load (MW)",
                min_value=1.0,
                max_value=1000.0,
                value=40.0,
                step=1.0,
                key="flat_load"
            )
        else:
            flat_load = None
    
    return {
        'start_year': start_year,
        'end_year': end_year,
        'investment_years': investment_years,
        'discount_rate': discount_rate,
        'selected_techs': selected_techs,
        'dispatchable_generators': dispatchable_generators,
        'fuel_types': fuel_types,
        'selected_vre': selected_vre,
        'selected_bess': selected_bess,
        'bess_degradation_enabled': bess_degradation_enabled,
        'rez_id': rez_id,
        'reference_year': reference_year,
        'solar_type': solar_type,
        'solar_degradation': solar_degradation,
        'solar_lifetime': solar_lifetime,
        'wind_type': wind_type,
        'wind_degradation': wind_degradation,
        'wind_lifetime': wind_lifetime,
        'multi_reference_year': multi_reference_year,
        'load_profile_type': load_profile,
        'flat_load': flat_load if load_profile == "Flat" else None
    }

def display_model_status(status: str, message: str = ""):
    """Display model execution status."""
    if status == "running":
        st.info(f"🔄 Model is running... {message}")
    elif status == "success":
        st.success(f"✅ Model completed successfully! {message}")
    elif status == "error":
        st.error(f"❌ Model failed: {message}")
    elif status == "ready":
        st.info("Ready to run model")

def create_results_tabs():
    """Create tabs for results display."""
    return st.tabs([
        "📊 Overview", 
        "⚡ Generation", 
        "🔋 Storage", 
        "💰 Economics", 
        "📈 Time Series",
        "📋 Detailed Stats"
    ])

def display_overview_metrics(n, overview_df):
    """Display key metrics in the overview tab."""
    try:
        # Calculate key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            npv = n.objective / 1e6  # Million
            st.metric(
                "System NPV",
                f"${npv:.2f}M",
                help="NPV in millions"
            )

        with col2:
            # LCOE
            lcoe = calculate_system_lcoe(n)
            st.metric(
                "System LCOE",
                f"${lcoe:.2f}/MWh",
                help="Levelized Cost of Energy Over Model Horizon"
            )

        with col3:
            # Total system cost (Nominal)
            nominal_system_cost = nominal_total_system_costs(n) / 1e6  # Convert to millions
            st.metric(
                "Total System Cost (Nominal)",
                f"${nominal_system_cost:.2f}M",
                help="Total system cost in millions"
            )

        with col4:
            # Average lifetime SRMC (Nominal)
            avg_srmc = average_lifetime_srmc(n)
            st.metric(
                "Average Lifetime SRMC (Nominal)",
                f"${avg_srmc:.2f}/MWh",
                help="Average Short Run Marginal Cost over model lifetime"
            )

        with col5:
            # Renewable Share
            renewable_share = overview_df.loc['Load Supplied by Renewables'].mean() * 100
            st.metric(
                "Renewable Share",
                f"{renewable_share:.2f}%",
                help="Percentage of load supplied by renewable sources"
            )

        with col6:
            # CO2 Emissions
            total_emissions = calculate_lifetime_emissions(n)
            st.metric(
                "Total CO2 Emissions",
                f"{total_emissions:.2f} MtCO2",
                help="Total CO2 emissions in million tonnes"
            )

        
        return True
    except Exception as e:
        st.error(f"Error calculating overview metrics: {str(e)}")
        return False

def calculate_renewable_share(n):
    """Calculate renewable energy share."""
    try:
        stats = n.statistics()
        renewable_carriers = ['SOLAR_PV', 'WIND']
        thermal_carriers = ['GAS']
        
        renewable_gen = 0
        thermal_gen = 0
        
        for carrier in renewable_carriers:
            if ('Generator', carrier) in stats.columns:
                renewable_gen += stats[('Generator', carrier)].get('Supply', 0)
        
        for carrier in thermal_carriers:
            if ('Generator', carrier) in stats.columns:
                thermal_gen += stats[('Generator', carrier)].get('Supply', 0)
        
        total_gen = renewable_gen + thermal_gen
        return renewable_gen / total_gen if total_gen > 0 else 0
    except:
        return 0

def plot_capacity_results(n):
    """Plot capacity results."""
    try:
        # Get optimal capacities
        generator_capacities = n.generators.p_nom_opt
        storage_capacities = n.storage_units.p_nom_opt if hasattr(n, 'storage_units') and not n.storage_units.empty else pd.Series()
        
        # Create capacity plot
        fig = go.Figure()
        
        # Add generator capacities
        for gen_name, capacity in generator_capacities.items():
            if capacity > 0.01:  # Only show significant capacities
                fig.add_trace(go.Bar(
                    x=[gen_name],
                    y=[capacity],
                    name=gen_name,
                    text=f"{capacity:.1f} MW",
                    textposition="auto"
                ))
        
        # Add storage capacities
        for storage_name, capacity in storage_capacities.items():
            if capacity > 0.01:
                fig.add_trace(go.Bar(
                    x=[storage_name],
                    y=[capacity],
                    name=storage_name,
                    text=f"{capacity:.1f} MW",
                    textposition="auto"
                ))
        
        fig.update_layout(
            title="Optimal Capacity Mix",
            xaxis_title="Technology",
            yaxis_title="Capacity (MW)",
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating capacity plot: {str(e)}")
        return None

def save_results_to_file(n, filename: str):
    """Save model results to NetCDF file."""
    try:
        n.export_to_netcdf(filename)
        return True
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return False

def plot_totex_results(overview_df):
    """Plot TOTEX costs for each generator across the model horizon."""
    try:
        # Find all rows containing TOTEX
        totex_rows = overview_df.index[overview_df.index.str.contains('TOTEX', na=False)]
        
        if len(totex_rows) == 0:
            st.warning("No TOTEX data found in overview dataframe")
            return None
        
        # Extract generator names and sum TOTEX across all years
        generator_totex = {}
        
        for row_name in totex_rows:
            # Split on space and keep the left part as generator name
            generator_name = row_name.split(' ')[0]
            
            # Sum across all years (all columns) for this generator
            total_cost = overview_df.loc[row_name].sum()
            
            # Add to our dictionary (in case there are multiple TOTEX entries for same generator)
            if generator_name in generator_totex:
                generator_totex[generator_name] += total_cost
            else:
                generator_totex[generator_name] = total_cost
        
        # Create bar chart
        fig = go.Figure()
        
        # Sort by cost (descending) for better visualization
        sorted_generators = sorted(generator_totex.items(), key=lambda x: x[1], reverse=True)
        
        generator_names = [item[0] for item in sorted_generators]
        costs = [item[1] for item in sorted_generators]
        
        # Format costs for display (in millions)
        costs_millions = [cost / 1e6 for cost in costs]
        text_labels = [f"${cost/1e6:.1f}M" for cost in costs]
        
        fig.add_trace(go.Bar(
            x=generator_names,
            y=costs_millions,
            text=text_labels,
            textposition="auto",
            marker=dict(
                color=costs_millions,
                colorscale='Viridis',
                colorbar=dict(title="Cost ($M)")
            )
        ))
        
        fig.update_layout(
            title="Total Expenditure (TOTEX) by Generator Technology",
            xaxis_title="Generator Technology",
            yaxis_title="Total Cost (Million $)",
            showlegend=False,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating TOTEX plot: {str(e)}")
        return None

def load_results_from_file(n, filename: str):
    """Load model results from NetCDF file."""
    try:
        n.import_from_netcdf(filename)
        return True
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return False

def create_time_series_plots(n):
    """Create time series plots for the results."""
    try:
        # Initialize session state for plots
        initialize_plot_session_state()
        
        st.subheader("Dispatch Plots")
        
        # Dispatch Plot in expandable dropdown
        with st.expander("📊 Generate Dispatch Plot", expanded=False):
            # Create form for dispatch plot settings
            with st.form("dispatch_plot_form"):
                
                # Create list of months from 2025-01 to 2050-12
                months = []
                for year in range(2025, 2051):
                    for month in range(1, 13):
                        months.append(f"{year}-{month:02d}")
                
                # Date range slider
                date_range = st.select_slider(
                    "Select Date Range (Start - End)",
                    options=months,
                    value=(months[0], months[12]),  # Default: 2025-01 to 2026-01
                    help="Select start and end dates for the plot range"
                )
                
                start_date, end_date = date_range
                
                # Plot options - second row
                col1, col2, col3, col4, col5 = st.columns([0.15, 0.15, 0.4, 0.15, 0.15])
                with col1:
                    stack_plot = st.checkbox("Stack Plot", value=True)
                with col2:
                    plot_curtailment = st.checkbox("Plot Curtailment", value=False)
                with col4:
                    y_min = st.number_input("Y Min", value=-20)
                with col5:
                    y_max = st.number_input("Y Max", value=60)
                
                # Submit button
                generate_dispatch = st.form_submit_button("Generate Dispatch Plot", type="primary")
            
            # Generate and display plot within the same expander
            if generate_dispatch and not plot_curtailment:
                with st.spinner("Creating dispatch plot..."):
                    try:
                        fig = create_dispatch_plot(
                            n, 
                            start_date=start_date, 
                            end_date=end_date, 
                            stack=stack_plot, 
                            y_range=[y_min, y_max]
                        )
                        # Store in session state and display immediately
                        store_plot_in_session('dispatch_plot', fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("Dispatch plot generated!")
                    except Exception as e:
                        st.error(f"Error creating dispatch plot: {str(e)}")

            elif generate_dispatch and plot_curtailment:
                with st.spinner("Creating dispatch plot with curtailment..."):
                    try:
                        fig = create_dispatch_plot_with_curtailment(
                            n, 
                            start_date=start_date, 
                            end_date=end_date, 
                            stack=stack_plot, 
                            y_range=[y_min, y_max],
                            vre_carriers=['SOLAR_PV', 'WIND'],
                        )
                        # Store in session state and display immediately
                        store_plot_in_session('dispatch_plot', fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("Dispatch plot with curtailment generated!")
                    except Exception as e:
                        st.error(f"Error creating dispatch plot: {str(e)}")
            
            # Display existing plot if it exists and no new plot was generated
            elif st.session_state.get('dispatch_plot') is not None and not generate_dispatch:
                st.plotly_chart(st.session_state.get('dispatch_plot'), use_container_width=True)
                st.caption("Previously generated dispatch plot")
        
        # Storage SOC plot
        st.subheader("Storage State of Charge")
        
        # Storage SOC Plot in expandable dropdown
        with st.expander("🔋 Generate Storage SOC Plot", expanded=False):
            # Create form for storage SOC plot settings
            with st.form("storage_soc_form"):
                
                # Create list of months from 2025-01 to 2050-12
                months = []
                for year in range(2025, 2051):
                    for month in range(1, 13):
                        months.append(f"{year}-{month:02d}")
                
                # Date range slider for storage SOC
                soc_date_range = st.select_slider(
                    "Select Date Range for Storage SOC (Start - End)",
                    options=months,
                    value=(months[0], months[12]),  # Default: 2025-01 to 2026-01
                    help="Select start and end dates for the storage SOC plot range",
                    key="soc_date_range"
                )
                
                soc_start_date, soc_end_date = soc_date_range
                
                # Submit button
                generate_soc = st.form_submit_button("Generate Storage SOC Plot", type="primary")
            
            # Generate and display plot within the same expander
            if generate_soc:
                with st.spinner("Creating storage SOC plot..."):
                    try:
                        fig = plot_storage_soc(n, start_date=soc_start_date, end_date=soc_end_date)
                        # Store in session state and display immediately
                        store_plot_in_session('storage_soc_plot', fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("Storage SOC plot generated!")
                    except Exception as e:
                        st.error(f"Error creating storage SOC plot: {str(e)}")
            
            # Display existing plot if it exists and no new plot was generated
            elif st.session_state.get('storage_soc_plot') is not None and not generate_soc:
                st.plotly_chart(st.session_state.get('storage_soc_plot'), use_container_width=True)
                st.caption("Previously generated storage SOC plot")
                    
    except Exception as e:
        st.error(f"Error in time series plotting: {str(e)}")

def create_detailed_economics_view(n):
    """Create detailed economic analysis view."""
    try:
        # Total cost breakdown
        total_cost = n.objective / 1e6
        st.metric("Total System Cost", f"${total_cost:.2f}M")
        
        # OPEX vs CAPEX breakdown
        try:
            capex_stats = n.statistics.capex()
            opex_stats = n.statistics.opex()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CAPEX Breakdown")
                if not capex_stats.empty:
                    st.dataframe(capex_stats)
                else:
                    st.info("No CAPEX data available")
            
            with col2:
                st.subheader("OPEX Breakdown")
                if not opex_stats.empty:
                    st.dataframe(opex_stats)
                else:
                    st.info("No OPEX data available")
        
        except Exception as e:
            st.warning(f"Detailed cost breakdown not available: {str(e)}")
        
        # Energy balance
        try:
            energy_balance = n.statistics.energy_balance()
            st.subheader("Energy Balance")
            st.dataframe(energy_balance)
        except Exception as e:
            st.warning(f"Energy balance not available: {str(e)}")
        
    except Exception as e:
        st.error(f"Error in economic analysis: {str(e)}")

def create_generation_analysis(n):
    """Create detailed generation analysis."""
    slider_min = n.snapshots.get_level_values('timestep').min().year
    slider_max = n.snapshots.get_level_values('timestep').max().year
    
    # Initialize session state for plots
    initialize_plot_session_state()
    
    try:
        # Generator capacities and details
        if hasattr(n, 'generators') and not n.generators.empty:
            st.subheader("Generator Details")
            
            # Filter for meaningful capacities
            gen_data = n.generators[['carrier', 'p_nom_opt']]
            gen_data = gen_data[gen_data['p_nom_opt'] > 0.01]
            generation_by_technology = n.snapshot_weightings.generators @ n.generators_t.p.div(1e3)  # GWh

            # Ensure both have the same order and only plot common indexes
            common_indexes = gen_data.index.intersection(generation_by_technology.index)
            gen_data = gen_data.loc[common_indexes]
            generation_by_technology = generation_by_technology.loc[common_indexes]

            # Assign consistent colors
            color_palette = px.colors.qualitative.Plotly
            color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(common_indexes)}

            if not gen_data.empty:
                fig_pie_capacity_mix = go.Figure(data=[go.Pie(
                    labels=gen_data.index,
                    values=gen_data['p_nom_opt'],
                    textinfo='label+percent',
                    marker=dict(colors=[color_map[name] for name in gen_data.index]),
                    title="Capacity Mix"
                )])
            else:
                st.info("No generator capacities found")
            if not generation_by_technology.empty:
                fig_pie_generation_mix = go.Figure(data=[go.Pie(
                    labels=generation_by_technology.index,
                    values=generation_by_technology.values,
                    textinfo='label+percent',
                    marker=dict(colors=[color_map[name] for name in generation_by_technology.index]),
                    title="Generation by Technology"
                )])

            col1, col2 = st.columns(2)
            with col1:
                col1.plotly_chart(fig_pie_capacity_mix, use_container_width=True)
            with col2:
                col2.plotly_chart(fig_pie_generation_mix, use_container_width=True)
        
        # Generator heatmap section
        st.subheader("Generator Output Heatmap")
        
        with st.expander("🌡️ Generate Heatmap", expanded=False):
            # Create form for heatmap settings
            with st.form("heatmap_form"):
                # Carrier selection
                carrier_select = st.selectbox(
                    "Select Carrier for Heatmap",
                    options=["WIND", "SOLAR_PV", "GAS"],
                    help="Choose the technology carrier to visualize in the heatmap"
                )
                
                # Create list of months from n.snapshots.index.min() to n.snapshots.index.max()
                months = []
                for year in range(slider_min, slider_max + 1):
                    for month in range(1, 13):
                        months.append(f"{year}-{month:02d}")
                
                # Date range slider for heatmap
                heatmap_date_range = st.select_slider(
                    "Select Date Range for Heatmap (Start - End)",
                    options=months,
                    value=(months[0], months[12]),  # Default: 2025-01 to 2026-01
                    help="Select start and end dates for the heatmap range",
                    key="heatmap_date_range"
                )
                
                heatmap_start_date, heatmap_end_date = heatmap_date_range
                
                # Submit button
                generate_heatmap = st.form_submit_button("Generate Heatmap", type="primary")
            
            # Generate and display heatmap within the same expander
            if generate_heatmap:
                with st.spinner("Creating heatmap..."):
                    try:
                        fig = plot_generator_output_heatmap(
                            n, 
                            carrier=carrier_select, 
                            start_date=heatmap_start_date, 
                            end_date=heatmap_end_date
                        )
                        # Store in session state and display immediately
                        store_plot_in_session('generator_heatmap', fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("Generator heatmap generated!")
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
            
            # Display existing plot if it exists and no new plot was generated
            elif st.session_state.get('generator_heatmap') is not None and not generate_heatmap:
                st.plotly_chart(st.session_state.get('generator_heatmap'), use_container_width=True)
                st.caption("Previously generated generator heatmap")

        # Monthly electrical production
        st.subheader("Monthly Electrical Production")

        with st.expander("📅 Generate Monthly Production Plot", expanded=False):
            # Create form for monthly electrical production settings
            with st.form("monthly_electrical_production_form"):
                
                # Create list of years from n.snapshots.index.min() to n.snapshots.index.max()
                years = [str(year) for year in range(slider_min, slider_max + 1)]
                
                # Date range slider for production
                date_range = st.select_slider(
                    "Select Date Range for Monthly Production (Start - End)",
                    options=years,
                    value=(years[0], years[1]),  # Default: first two years
                    help="Select start and end dates for the monthly production range",
                    key="electrical_date_range"
                )

                electrical_start_date, electrical_end_date = date_range

                # Submit button
                generate_monthly_electrical = st.form_submit_button("Generate Monthly Electrical Production", type="primary")

            # Generate and display plot within the same expander
            if generate_monthly_electrical:
                with st.spinner("Creating monthly electrical production plot..."):
                    try:
                        fig = plot_monthly_electric_production(
                            n,
                            start_year=electrical_start_date,
                            end_year=electrical_end_date
                        )
                        # Store in session state and display immediately
                        store_plot_in_session('monthly_electrical_plot', fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("Monthly electrical production plot generated!")
                    except Exception as e:
                        st.error(f"Error creating plot: {str(e)}")
            
            # Display existing plot if it exists and no new plot was generated
            elif st.session_state.get('monthly_electrical_plot') is not None and not generate_monthly_electrical:
                st.plotly_chart(st.session_state.get('monthly_electrical_plot'), use_container_width=True)
                st.caption("Previously generated monthly electrical production plot")

    except Exception as e:
        st.error(f"Error in generation analysis: {str(e)}")

def create_storage_analysis(n):
    """Create detailed storage analysis."""
    # Initialize session state for plots
    initialize_plot_session_state()
    
    try:
        # Storage unit details
        if hasattr(n, 'storage_units') and not n.storage_units.empty:
            st.subheader("Storage Unit Details")
            
            storage_data = n.storage_units[['carrier', 'p_nom_opt', 'capital_cost', 'max_hours']]
            storage_data = storage_data[storage_data['p_nom_opt'] > 0.01]
            
            if not storage_data.empty:
                st.dataframe(storage_data.round(3))
                
                # Create two separate plots in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Power capacity plot
                    fig_power = go.Figure()
                    fig_power.add_trace(go.Bar(
                        x=storage_data.index,
                        y=storage_data['p_nom_opt'],
                        text=[f"{cap:.1f} MW" for cap in storage_data['p_nom_opt']],
                        textposition="auto",
                        marker=dict(color='lightblue'),
                        showlegend=False
                    ))
                    
                    fig_power.update_layout(
                        title="Power Capacity",
                        xaxis_title="Storage Technology",
                        yaxis_title="Power (MW)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_power, use_container_width=True)
                
                with col2:
                    # Energy capacity plot
                    energy_cap = storage_data['p_nom_opt'] * storage_data['max_hours']
                    fig_energy = go.Figure()
                    fig_energy.add_trace(go.Bar(
                        x=storage_data.index,
                        y=energy_cap,
                        text=[f"{cap:.1f} MWh" for cap in energy_cap],
                        textposition="auto",
                        marker=dict(color='lightgreen'),
                        showlegend=False
                    ))
                    
                    fig_energy.update_layout(
                        title="Energy Capacity",
                        xaxis_title="Storage Technology",
                        yaxis_title="Energy (MWh)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_energy, use_container_width=True)

                
                # Storage SOC Heatmap in expandable dropdown
                st.subheader("Storage SOC Heatmap")
                
                with st.expander("🌡️ Generate Storage SOC Heatmap", expanded=False):
                    # Create form for storage SOC heatmap settings
                    with st.form("storage_soc_heatmap_form"):
                        
                        # Create list of months from 2025-01 to 2050-12
                        months = []
                        for year in range(2025, 2051):
                            for month in range(1, 13):
                                months.append(f"{year}-{month:02d}")
                        
                        # Date range slider for storage SOC heatmap
                        storage_heatmap_date_range = st.select_slider(
                            "Select Date Range for Storage SOC Heatmap (Start - End)",
                            options=months,
                            value=(months[0], months[12]),  # Default: 2025-01 to 2026-01
                            help="Select start and end dates for the storage SOC heatmap range",
                            key="storage_heatmap_date_range"
                        )
                        
                        storage_heatmap_start_date, storage_heatmap_end_date = storage_heatmap_date_range
                        
                        # Submit button
                        generate_storage_heatmap = st.form_submit_button("Generate Storage SOC Heatmap", type="primary")
                    
                    # Generate and display storage SOC heatmap within the same expander
                    if generate_storage_heatmap:
                        with st.spinner("Creating storage SOC heatmap..."):
                            try:
                                fig = plot_storage_soc_heatmap(
                                    n, 
                                    start_date=storage_heatmap_start_date, 
                                    end_date=storage_heatmap_end_date
                                )
                                # Store in session state and display immediately
                                store_plot_in_session('storage_soc_heatmap', fig)
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("Storage SOC heatmap generated!")
                            except Exception as e:
                                st.error(f"Error creating storage SOC heatmap: {str(e)}")
                    
                    # Display existing plot if it exists and no new plot was generated
                    elif st.session_state.get('storage_soc_heatmap') is not None and not generate_storage_heatmap:
                        st.plotly_chart(st.session_state.get('storage_soc_heatmap'), use_container_width=True)
                        st.caption("Previously generated storage SOC heatmap")
            else:
                st.info("No storage capacities found")
        
        # Store details (if using link+store approach)
        if hasattr(n, 'stores') and not n.stores.empty:
            st.subheader("Store Details")
            store_data = n.stores[['carrier', 'e_nom_opt', 'capital_cost']]
            store_data = store_data[store_data['e_nom_opt'] > 0.01]
            if not store_data.empty:
                st.dataframe(store_data.round(3))
            
    except Exception as e:
        st.error(f"Error in storage analysis: {str(e)}")

def create_sensitivity_analysis_section():
    """Create sensitivity analysis section."""
    st.subheader("Sensitivity Analysis")
    
    with st.expander("CO₂ Price Sensitivity"):
        st.info("CO₂ price sensitivity analysis will be implemented here")
        # This would implement the CO2 sensitivity from the notebook
        
        co2_prices = st.multiselect(
            "CO₂ Prices ($/tCO₂)",
            options=[0, 25, 50, 100, 150],
            default=[0, 50, 100],
            help="Select CO₂ prices for sensitivity analysis"
        )
        
        if st.button("Run CO₂ Sensitivity"):
            st.info("CO₂ sensitivity analysis would run here")
    
    with st.expander("Discount Rate Sensitivity"):
        discount_rates = st.multiselect(
            "Discount Rates",
            options=[0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
            default=[0.06, 0.08, 0.10],
            help="Select discount rates for sensitivity analysis"
        )
        
        if st.button("Run Discount Rate Sensitivity"):
            st.info("Discount rate sensitivity analysis would run here")

def create_export_section(n):
    """Create export and download section."""
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to CSV
        if st.button("Export Generator Data to CSV"):
            try:
                if hasattr(n, 'generators') and not n.generators.empty:
                    csv_data = n.generators.to_csv()
                    st.download_button(
                        label="Download Generator CSV",
                        data=csv_data,
                        file_name="generator_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No generator data to export")
            except Exception as e:
                st.error(f"Error exporting generator data: {str(e)}")
    
    with col2:
        # Export statistics
        if st.button("Export Statistics to CSV"):
            try:
                if hasattr(n, 'statistics'):
                    stats_data = n.statistics().to_csv()
                    st.download_button(
                        label="Download Statistics CSV",
                        data=stats_data,
                        file_name="model_statistics.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No statistics data to export")
            except Exception as e:
                st.error(f"Error exporting statistics: {str(e)}")

def create_model_comparison_section():
    """Create section for comparing multiple model runs."""
    st.subheader("Model Comparison")
    
    if "model_runs" not in st.session_state:
        st.session_state.model_runs = {}
    
    # Save current run
    if st.session_state.network is not None:
        run_name = st.text_input("Run Name", value=f"Run_{len(st.session_state.model_runs) + 1}")
        
        if st.button("Save Current Run for Comparison"):
            try:
                # Store key metrics
                total_cost = st.session_state.network.objective / 1e6
                renewable_share = calculate_renewable_share(st.session_state.network)
                
                st.session_state.model_runs[run_name] = {
                    "total_cost": total_cost,
                    "renewable_share": renewable_share,
                    "timestamp": pd.Timestamp.now()
                }
                st.success(f"Run '{run_name}' saved for comparison")
            except Exception as e:
                st.error(f"Error saving run: {str(e)}")
    
    # Display comparison
    if st.session_state.model_runs:
        comparison_df = pd.DataFrame.from_dict(st.session_state.model_runs, orient='index')
        st.dataframe(comparison_df)
        
        # Clear all runs
        if st.button("Clear All Saved Runs"):
            st.session_state.model_runs = {}
            st.rerun()

def validate_parameters(params):
    """Validate user input parameters."""
    errors = []
    warnings = []
    
    # Check timeline parameters
    if params['start_year'] >= params['end_year']:
        errors.append("Start year must be before end year")
    
    if not params['investment_years']:
        errors.append("At least one investment year must be selected")
    
    if min(params['investment_years']) < params['start_year']:
        errors.append("Investment years cannot be before start year")
    
    if max(params['investment_years']) > params['end_year']:
        errors.append("Investment years cannot be after end year")
    
    # Check technology parameters
    if not params['selected_techs']:
        errors.append("At least one technology must be selected")
    
    # Check dispatchable generators and fuel types match
    if len(params['dispatchable_generators']) != len(params['fuel_types']):
        errors.append("Number of dispatchable generators must match number of fuel types")
    
    # Check for storage without generation
    storage_techs = params['selected_bess']
    generation_techs = params['selected_vre'] + params['dispatchable_generators']
    
    if storage_techs and not generation_techs:
        warnings.append("Storage technologies selected without generation technologies")
    
    # Check VRE selection
    if params['selected_vre'] and not any([params['solar_lifetime'], params['wind_lifetime']]):
        warnings.append("VRE technologies selected but lifetimes not properly configured")
    
    # Check load parameters
    if params['load_profile_type'] == 'Flat' and (params['flat_load'] is None or params['flat_load'] <= 0):
        errors.append("Flat load must be greater than 0")
    
    return errors, warnings

def display_validation_results(errors, warnings):
    """Display validation results to user."""
    if errors:
        st.error("⚠️ Please fix the following errors before running the model:")
        for error in errors:
            st.error(f"• {error}")
        return False
    
    if warnings:
        st.warning("⚠️ Please review the following warnings:")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    return True

def estimate_model_runtime(params):
    """Estimate model runtime based on parameters."""
    try:
        # Simple heuristic for runtime estimation
        years = params['end_year'] - params['start_year']
        num_techs = len(params['selected_techs'])
        num_periods = len(params['investment_years'])
        
        # Base time (minutes)
        base_time = 2
        
        # Scale factors
        year_factor = years * 0.5
        tech_factor = num_techs * 0.2
        period_factor = num_periods * 1.5
        
        estimated_minutes = base_time + year_factor + tech_factor + period_factor
        
        if estimated_minutes < 1:
            return "< 1 minute"
        elif estimated_minutes < 60:
            return f"~{int(estimated_minutes)} minutes"
        else:
            hours = int(estimated_minutes // 60)
            minutes = int(estimated_minutes % 60)
            return f"~{hours}h {minutes}m"
    
    except:
        return "Unknown"

def create_parameter_summary(params):
    """Create a summary table of current parameters."""
    summary_data = {
        "Parameter": [
            "Timeline",
            "Investment Periods",
            "Discount Rate",
            "Technologies",
            "REZ Location", 
            "Reference Year",
            "Load Profile"
        ],
        "Value": [
            f"{params['start_year']} - {params['end_year']}",
            ", ".join(map(str, params['investment_years'])),
            f"{params['discount_rate']:.1%}",
            f"{len(params['selected_techs'])} selected",
            params['rez_id'],
            str(params['reference_year']),
            f"{params['load_profile_type']}" + (f" ({params['flat_load']} MW)" if params['flat_load'] else "")
        ]
    }
    
    return pd.DataFrame(summary_data)

def create_model_status_card(status, message="", runtime_estimate=""):
    """Create an enhanced status card."""
    if status == "ready":
        st.info("🔧 Model ready to run")
        if runtime_estimate:
            st.info(f"⏱️ Estimated runtime: {runtime_estimate}")
    elif status == "running":
        with st.container():
            st.info("🔄 Model is running...")
            if message:
                st.caption(message)
            # Add a progress bar placeholder
            progress_bar = st.progress(0)
            # In a real implementation, you'd update this progress
    elif status == "success":
        st.success("✅ Model completed successfully!")
        if message:
            st.success(message)
    elif status == "error":
        st.error("❌ Model failed")
        if message:
            st.error(f"Error: {message}")

def check_data_files_exist():
    """Check if required data files exist."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_files = [
        os.path.join(base_path, 'data', 'INPUTS.xlsx'),
        os.path.join(base_path, 'data', 'GENCOST_CAPEX.csv'),
        os.path.join(base_path, 'data', 'GENCOST_VARIABLE.csv')
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))
    
    return missing_files

def create_data_file_status():
    """Create a status indicator for data files."""
    missing_files = check_data_files_exist()
    
    if not missing_files:
        st.success("✅ All required data files found")
        return True
    else:
        st.error("❌ Missing required data files:")
        for file in missing_files:
            st.error(f"• {file}")
        st.info("Please ensure the following files are in the 'data/' directory:")
        st.info("• INPUTS.xlsx")
        st.info("• GENCOST_CAPEX.csv") 
        st.info("• GENCOST_VARIABLE.csv")
        return False