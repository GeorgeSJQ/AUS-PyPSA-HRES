import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import session_state as ss
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import traceback
import time
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_utils import *
import pypsa
from src.utils_multiperiod import *
from src.config import *

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================
def setup_logging():
    """Setup logging configuration for the dashboard"""
    # Configure logging to only output to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Create logger for this module
    logger = logging.getLogger('PyPSA_Dashboard')
    logger.info("="*60)
    logger.info("PyPSA HRES Model Dashboard Starting")
    logger.info("="*60)
    return logger

# Initialize logging
logger = setup_logging()

# ==========================================================
# DASHBOARD START
# ==========================================================
st.title("PyPSA Stand-alone HRES Model Dashboard")
st.markdown("---")

logger.info("Dashboard interface loaded")
logger.info(f"Session ID: {st.session_state}")

# Initialize session state
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'model_status' not in st.session_state:
    st.session_state.model_status = 'ready'
    logger.info("Session state initialized to 'ready'")
if 'network' not in st.session_state:
    st.session_state.network = None

logger.info(f"Current model status: {st.session_state.model_status}")

# Sidebar for parameters
st.sidebar.header("Model Configuration")

# Get parameters from sidebar
params = parameter_input_section()
logger.info(f"Parameters configured: {list(params.keys())}")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Model Execution")
    
    # Check data files first
    data_files_ok = create_data_file_status()
    
    # Display current parameters
    with st.expander("Current Parameters", expanded=False):
        # Parameter summary table
        summary_df = create_parameter_summary(params)
        st.table(summary_df)
    
    # Validate parameters
    errors, warnings = validate_parameters(params)
    can_run = display_validation_results(errors, warnings) and data_files_ok
    
    # Runtime estimate
    if can_run:
        runtime_estimate = estimate_model_runtime(params)
        st.info(f"⏱️ Estimated runtime: {runtime_estimate}")

with col2:
    st.header("Actions")
    
    # Run model button - only enabled if validation passes
    run_model = st.button(
        "🚀 Run Model",
        type="primary",
        use_container_width=True,
        help="Execute the PyPSA optimization model",
        disabled=not can_run
    )
    
    # Load results section
    st.subheader("📁 Load Results")
    uploaded_file = st.file_uploader(
        "Choose a NetCDF file",
        type=['nc'],
        key="file_uploader",
        help="Upload a previously saved model results file"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file (different from what's currently loaded)
        if 'loaded_file_name' not in st.session_state or st.session_state.loaded_file_name != uploaded_file.name:
            logger.info(f"Loading results from uploaded file: {uploaded_file.name}")
            with st.spinner("Loading results..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    logger.info(f"Saving to temporary path: {temp_path}")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load into network
                    n = pypsa.Network()
                    n.import_from_netcdf(temp_path)
                    st.session_state.network = n
                    st.session_state.model_status = 'success'
                    st.session_state.loaded_file_name = uploaded_file.name
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    logger.info("Results loaded successfully from uploaded file")
                    st.success("Results loaded successfully!")
                    
                except Exception as e:
                    error_msg = f"Error loading file: {str(e)}"
                    logger.error(error_msg)
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    st.error(error_msg)
        else:
            st.info(f"File '{uploaded_file.name}' is already loaded.")
    
    # Clear results button
    if st.session_state.model_status == 'success':
        if st.button("🗑️ Clear Results", use_container_width=True, help="Clear current results to load a new file"):
            restart_network(n)
            st.session_state.network = None
            st.session_state.model_status = 'ready'
            if 'loaded_file_name' in st.session_state:
                del st.session_state.loaded_file_name
            logger.info("Results cleared by user")
            st.rerun()

# Display model status
display_model_status(st.session_state.model_status)

# Model execution
if run_model:
    logger.info("🚀 Starting model execution")
    logger.info(f"Selected technologies: {params['selected_techs']}")
    logger.info(f"Timeline: {params['start_year']}-{params['end_year']}")
    logger.info(f"Investment years: {params['investment_years']}")
    
    st.session_state.model_status = 'running'
    
    with st.spinner("Setting up model..."):
        try:
            logger.info("Initializing PyPSA network...")
            # Initialize network
            n = pypsa.Network()
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Set timeline
            start_year = params['start_year']
            end_year = params['end_year']
            investment_years = params['investment_years']
            r = params['discount_rate']
            
            logger.info(f"Timeline configured: {start_year}-{end_year}, Investment years: {investment_years}")
            st.info(f"Timeline: {start_year}-{end_year}, Investment years: {investment_years}")
            
            # Load cost data
            inputs_path = os.path.join(base_path, 'data', 'INPUTS.xlsx')
            logger.info(f"Loading cost data from: {inputs_path}")
            
            # Check if file exists
            if not os.path.exists(inputs_path):
                error_msg = f"INPUTS.xlsx not found at {inputs_path}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.model_status = 'error'
                st.stop()
            
            tech_params = get_isp_tech_params(inputs_path)
            fuel_costs = get_isp_fuel_costs(inputs_path)
            build_costs = get_isp_tech_build_costs(inputs_path, sheet_name='TECH_BUILD_COSTS')
            logger.info("Technology parameters loaded successfully")
            
            # Load GENCOST data
            capex_path = os.path.join(base_path, "data", "GENCOST_CAPEX.csv")
            variable_cost_path = os.path.join(base_path, "data", "GENCOST_VARIABLE.csv")
            logger.info(f"Loading GENCOST data from: {capex_path} and {variable_cost_path}")
            
            if not os.path.exists(capex_path) or not os.path.exists(variable_cost_path):
                error_msg = "GENCOST cost files not found"
                logger.error(f"{error_msg}: CAPEX={os.path.exists(capex_path)}, Variable={os.path.exists(variable_cost_path)}")
                st.error(error_msg)
                st.session_state.model_status = 'error'
                st.stop()
            
            cost_capex = get_gencost_capex(capex_path)
            cost_variable = get_gencost_variables(variable_cost_path, end_year=2055)
            logger.info("GENCOST data loaded successfully")
            
            # Merge and process cost data
            inputs_df = pd.merge(cost_capex, cost_variable, on="YEAR", how="outer").bfill()
            
            inputs_df = filter_and_process_input_costs(
                inputs_df=inputs_df,
                techs_to_keep=params['selected_techs'],
                annuitise_capex=True,
                discount_rate=r,
                model_horizon=end_year-start_year + 1,
            )
            
            logger.info("Cost data processed and filtered successfully")
            st.success("Cost data loaded successfully")
            
        except Exception as e:
            error_msg = f"Error setting up model: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.model_status = 'error'
            st.stop()
    
    with st.spinner("Generating weather traces..."):
        try:
            logger.info("Starting weather trace generation...")
            # VRE parameters
            rez_id = getattr(REZIDS, params['rez_id'])
            reference_year = params['reference_year']
            logger.info(f"REZ ID: {rez_id}, Reference year: {reference_year}")
            
            # Solar trace
            if 'SOLAR_PV' in params['selected_vre']:
                logger.info("Generating solar trace...")
                solar_trace = solar_trace_construction(
                    start_year=start_year,
                    end_year=end_year,
                    rez_ids=rez_id,
                    reference_year=reference_year,
                    solar_type=params['solar_type'],
                    annual_degradation=params['solar_degradation'],
                    lifetime=params['solar_lifetime'],
                    build_year=investment_years[0],
                    multi_reference_year=params['multi_reference_year'],
                    year_type='calendar',
                    investment_periods=investment_years
                )
                logger.info(f"Solar trace generated with shape: {solar_trace.shape}")
            else:
                solar_trace = pd.DataFrame()
                logger.info("Solar PV not selected, skipping solar trace generation")
            
            # Wind trace
            if 'WIND' in params['selected_vre']:
                logger.info("Generating wind trace...")
                wind_trace = wind_trace_construction(
                    start_year=start_year,
                    end_year=end_year,
                    rez_ids=rez_id,
                    reference_year=reference_year,
                    wind_type=params['wind_type'],
                    annual_degradation=params['wind_degradation'],
                    lifetime=params['wind_lifetime'],
                    build_year=investment_years[0],
                    multi_reference_year=params['multi_reference_year'],
                    year_type='calendar',
                    investment_periods=investment_years
                )
                logger.info(f"Wind trace generated with shape: {wind_trace.shape}")
            else:
                wind_trace = pd.DataFrame()
                logger.info("Wind not selected, skipping wind trace generation")
            
            # Combine traces
            trace_dfs = [df for df in [solar_trace, wind_trace] if not df.empty]
            if trace_dfs:
                max_output_trace = pd.concat(trace_dfs, axis=1, join='outer')
                logger.info(f"Combined trace shape: {max_output_trace.shape}")
            else:
                # Create dummy trace if no VRE selected
                dummy_index = pd.date_range(
                    start=f'{start_year}-01-01',
                    end=f'{end_year}-12-31 23:00:00',
                    freq='H'
                )
                max_output_trace = pd.DataFrame(index=dummy_index)
                logger.warning("No VRE technologies selected, created dummy trace")
            
            logger.info("Weather traces generated successfully")
            st.success("Weather traces generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating weather traces: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.model_status = 'error'
            st.stop()
    
    with st.spinner("Building network..."):
        try:
            logger.info("Building network structure...")
            # Set network snapshots and investment periods
            n.snapshots = max_output_trace.index
            n.investment_periods = investment_years
            logger.info(f"Network snapshots: {len(n.snapshots)} timesteps")
            logger.info(f"Investment periods: {investment_years}")
            
            # Set snapshot weightings
            n.snapshot_weightings['objective'] = 0.5
            n.snapshot_weightings['stores'] = 0.5
            n.snapshot_weightings['generators'] = 0.5
            
            # Investment period weightings
            investment_weightings = calculate_investment_period_weightings(
                end_year=end_year,
                investment_period_years=investment_years,
                discount_rate=r
            )
            n.investment_period_weightings = investment_weightings
            logger.info(f"Investment period weightings: {investment_weightings}")
            
            # Add carriers
            carriers = ["GAS", "DIESEL", "BIOMASS", "WIND", "SOLAR_PV", "BESS", "AC"]
            colors = ["indianred", "orange", "green", "dodgerblue", "gold", "yellowgreen", "black"]
            co2_emissions = [0.6, 0.8, 0.0, 0, 0, 0, 0]
            
            n.add(
                "Carrier",
                carriers,
                color=colors,
                co2_emissions=co2_emissions,
            )
            logger.info(f"Added carriers: {carriers}")
            
            # Add bus
            n.add("Bus", "electricity", carrier="AC")
            logger.info("Added electricity bus")
            
            # Add load
            if params['load_profile_type'] == 'Flat':
                load_fix = pd.Series(params['flat_load'], index=n.snapshots, name="load")
                n.add("Load", "load_flat", bus="electricity", carrier="AC", p_set=load_fix, overwrite=True)
                logger.info(f"Added flat load profile: {params['flat_load']} MW")
            
            logger.info("Network structure created successfully")
            st.success("Network structure created successfully")
            
        except Exception as e:
            error_msg = f"Error building network: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.model_status = 'error'
            st.stop()
    
    with st.spinner("Adding generators and storage..."):
        try:
            logger.info("Adding generators and storage components...")
            life = end_year - start_year  # Default lifetime for model
            logger.info(f"Using default lifetime: {life} years")
            
            # Add gas generator if selected
            if params['dispatchable_generators']:
                for i, (disp_tech, fuel_type) in enumerate(zip(params['dispatchable_generators'], params['fuel_types'])):
                    logger.info(f"Adding dispatchable generator {i+1}: {disp_tech} with fuel {fuel_type}")
                    
                    # Get costs for the specific technology
                    gas_capex = get_generator_capex(build_costs, tech_params, r, start_year, disp_tech, rez_id.value, annuitise=True, lifetime=life)
                    gas_opex = get_generator_marginal_cost_series(n, tech_params, fuel_costs, disp_tech, fuel_type, rez_id.value)
                    logger.info(f"{disp_tech} CAPEX: {gas_capex}, Fuel: {fuel_type}")
                    
                    # Add the generator with unique name
                    gen_name = f"{disp_tech}_{i+1}" if len(params['dispatchable_generators']) > 1 else disp_tech
                    add_dispatchable_generators(
                        n,
                        name=gen_name,
                        carrier=fuel_type,
                        bus="electricity",
                        capital_cost=gas_capex,
                        marginal_cost=gas_opex,
                        lifetime=life,
                        p_nom_extendable=True,
                        build_years=investment_years,
                        overwrite=True,
                    )
                    logger.info(f"Dispatchable generator {gen_name} added successfully")
            else:
                logger.info("No dispatchable technologies selected, skipping dispatchable generators")
            
            # Add solar if selected
            if 'SOLAR_PV' in params['selected_vre'] and not solar_trace.empty:
                logger.info("Adding solar PV generator...")
                solar_capex = get_generator_capex(build_costs, tech_params, r, start_year, 'SOLAR_PV', rez_id.value, annuitise=True, lifetime=life)
                logger.info(f"Solar CAPEX: {solar_capex}")
                
                add_vre_generators(
                    n,
                    name="SOLAR_PV",
                    carrier="SOLAR_PV",
                    bus="electricity",
                    p_max_pu=solar_trace.iloc[:, 0],
                    capital_cost=solar_capex,
                    build_years=investment_years,
                    lifetime=life,
                    p_nom_extendable=True,
                )
                logger.info("Solar PV generator added successfully")
            else:
                logger.info("Solar PV not selected or no solar trace available, skipping solar generator")
            
            # Add wind if selected
            if 'WIND' in params['selected_vre'] and not wind_trace.empty:
                logger.info("Adding wind generator...")
                wind_capex = get_generator_capex(build_costs, tech_params, r, start_year, 'WIND', rez_id.value, annuitise=True, lifetime=life)
                logger.info(f"Wind CAPEX: {wind_capex}")
                
                add_vre_generators(
                    n,
                    name="WIND",
                    carrier="WIND",
                    bus="electricity",
                    p_max_pu=wind_trace.iloc[:, 0],
                    capital_cost=wind_capex,
                    build_years=investment_years,
                    lifetime=life,
                    p_nom_extendable=True,
                )
                logger.info("Wind generator added successfully")
            else:
                logger.info("Wind not selected or no wind trace available, skipping wind generator")
            
            # Add BESS if selected
            if params['selected_bess']:
                logger.info(f"Adding BESS technologies: {params['selected_bess']}")
                
                # BESS degradation (only if enabled)
                if params['bess_degradation_enabled']:
                    BESS_degradation_series = pd.Series(
                        [1.00, 0.95, 0.92, 0.90, 0.88, 0.86, 0.85, 0.83, 0.81, 0.79,
                         0.78, 0.76, 0.74, 0.73, 0.71, 0.69, 0.67, 0.66, 0.64, 0.62, 
                         0.61, 0.59, 0.58, 0.56, 0.55],
                        index=range(25)
                    )
                    
                    BESS_trace = calc_custom_degradation(
                        network_snapshots=n.snapshots,
                        technology='BESS',
                        build_year=start_year,
                        annual_degradation=BESS_degradation_series,
                        lifetime=life
                    )
                    logger.info("BESS degradation enabled and calculated")
                else:
                    # No degradation - all ones
                    BESS_trace = pd.Series(1.0, index=n.snapshots, name='BESS')
                    logger.info("BESS degradation disabled - using constant efficiency")
                
                # Storage configurations
                storage_configs = {}
                for tech in params['selected_bess']:
                    hours = int(tech.split('_')[1].replace('HR', ''))
                    try:
                        capex = get_generator_capex(build_costs, tech_params, r, start_year, tech, rez_id.value, annuitise=True, lifetime=life)
                    except:
                        capex = inputs_df.loc[start_year, f'{tech}_CAPEX'] if f'{tech}_CAPEX' in inputs_df.columns else 1000
                    
                    storage_configs[tech] = {
                        'max_hours': hours,
                        'capital_cost': capex,
                        'carrier': 'BESS'
                    }
                
                logger.info(f"Storage configurations: {storage_configs}")
                
                # Add storage units
                add_multiple_storage_units(
                    n=n,
                    storage_configs=storage_configs,
                    p_max_pu=BESS_trace,
                    efficiency_store=0.92,
                    efficiency_dispatch=0.92,
                    lifetime=life,
                    build_years=investment_years,
                    p_nom_extendable=True,
                    overwrite=True
                )
                logger.info("BESS storage units added successfully")
            else:
                logger.info("No BESS technologies selected, skipping storage")
            
            logger.info("All generators and storage components added successfully")
            st.success("Generators and storage added successfully")
            
        except Exception as e:
            error_msg = f"Error adding generators and storage: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.model_status = 'error'
            st.stop()
    
    with st.spinner("Optimizing model..."):
        try:
            logger.info("Starting model optimization...")
            logger.info("Solver: Gurobi, Multi-investment periods: True")
            
            # Run optimization
            optimization_start = time.time()
            n.optimize(solver_name='gurobi', multi_investment_periods=True)
            optimization_time = time.time() - optimization_start
            
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            
            # Check optimization status
            if hasattr(n, 'model') and hasattr(n.model, 'status'):
                opt_status = n.model.status
                opt_termination_condition = n.model.termination_condition
                logger.info(f"Optimization status: {opt_status}")
                logger.info(f"Optimization termination condition: {opt_termination_condition}")

                # Check if optimization was successful
                if opt_status != 'ok' or opt_termination_condition != 'optimal':
                    error_msg = f"Optimization failed with status: {opt_status}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.model_status = 'error'
                    st.stop()
            else:
                logger.warning("Could not determine optimization status")
            
            # Log objective value
            if hasattr(n, 'objective'):
                logger.info(f"Objective value: {n.objective}")
            elif hasattr(n, 'model') and hasattr(n.model, 'objective'):
                logger.info(f"Objective value: {n.model.objective}")
            else:
                logger.warning("Could not retrieve objective value")
            
            # Store results
            st.session_state.network = n
            st.session_state.model_status = 'success'
            
            logger.info("✅ Model execution completed successfully!")
            st.success("Model optimization completed successfully!")
            
        except Exception as e:
            error_msg = f"Error during optimization: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            st.session_state.model_status = 'error'
            st.stop()

# Results display
if st.session_state.model_status == 'success' and st.session_state.network is not None:
    logger.info("Displaying results...")
    st.markdown("---")
    st.header("Results")
    
    n = st.session_state.network
    
    # Create results tabs
    tabs = create_results_tabs()
    
    with tabs[0]:  # Overview
        st.subheader("System Overview")
        overview_df = generate_multiperiod_overview(
            n, 
            renewable_carriers=['SOLAR_PV', 'WIND'], 
            thermal_carriers=['GAS']
        )
        display_overview_metrics(n, overview_df)
        
        # Capacity plot
        capacity_fig = plot_capacity_results(n)
        if capacity_fig:
            st.plotly_chart(capacity_fig, use_container_width=True)

        
        st.subheader("Multi-period Overview")
        st.dataframe(overview_df)
        
        # TOTEX plot
        totex_fig = plot_totex_results(overview_df)
        if totex_fig:
            st.plotly_chart(totex_fig, use_container_width=True)
    
    with tabs[1]:  # Generation
        create_generation_analysis(n)
    
    with tabs[2]:  # Storage
        create_storage_analysis(n)
    
    with tabs[3]:  # Economics
        create_detailed_economics_view(n)
    
    with tabs[4]:  # Time Series
        create_time_series_plots(n)
    
    with tabs[5]:  # Detailed Stats
        st.subheader("Detailed Statistics")
        try:
            if hasattr(n, 'statistics'):
                detailed_stats = n.statistics()
                st.dataframe(detailed_stats)
            else:
                st.info("No detailed statistics available")
        except Exception as e:
            st.error(f"Error displaying detailed statistics: {str(e)}")
        
        # Add sensitivity analysis
        create_sensitivity_analysis_section()
        
        # Add model comparison
        create_model_comparison_section()
    
    # Enhanced save/export section
    st.markdown("---")
    create_export_section(n)
    
    # Save results section
    st.markdown("---")
    st.subheader("Save Results")
    
    col1, col2 = st.columns(2)
    with col1:
        filename = st.text_input(
            "Filename",
            value=f"model_results_{time.strftime('%Y%m%d_%H%M%S')}.nc",
            help="NetCDF filename to save results"
        )
    
    with col2:
        if st.button("💾 Save Results", type="secondary"):
            logger.info(f"Saving results to file: {filename}")
            if save_results_to_file(n, filename):
                logger.info(f"Results successfully saved to {filename}")
                st.success(f"Results saved to {filename}")
            else:
                logger.error(f"Failed to save results to {filename}")

st.markdown("---")
st.caption("PyPSA Multi-period HRES Model Dashboard")
logger.info("Dashboard page rendered successfully")
