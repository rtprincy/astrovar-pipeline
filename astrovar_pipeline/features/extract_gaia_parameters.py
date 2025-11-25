from astroquery.gaia import Gaia

def query_gaia_params(source_ids):
    """
    Queries the Gaia TAP service to retrieve parameters for a list of Source IDs.
    
    Parameters:
    -----------
    source_ids : list of int or str
        A list of Gaia DR3 source_ids.
        
    Returns:
    --------
    astropy.table.Table
        A table containing the requested parameters.
    """
    
    # Convert all IDs to strings for the query
    ids_str = ','.join([str(sid) for sid in source_ids])
    
    # Construct the ADQL query
    # We select specific columns from the Gaia DR3 source table
    query = f"""
    SELECT 
        source_id,
        parallax,
        parallax_error,
        bp_rp,
        phot_g_mean_mag,
        phot_g_mean_mag + 5 + 5*LOG10(parallax/1000) AS g_abs,
        parallax_over_error,
        RUWE
    FROM gaiadr3.gaia_source
    WHERE source_id IN ({ids_str})
    """
    
    print(f"Querying Gaia Archive for {len(source_ids)} sources...")
    
    try:
        # Launch the query asynchronously
        job = Gaia.launch_job_async(query)
        
        # Get the results
        results = job.get_results()
        
        return results
        
    except Exception as e:
        print(f"An error occurred during the TAP query: {e}")
        return None
def query_vari_summary(source_ids):
    """
    Queries the Gaia TAP service to retrieve the variability summary 
    (gaiadr3.vari_summary) for a list of Source IDs.
    
    Parameters:
    -----------
    source_ids : list of int or str
        A list of Gaia DR3 source_ids.
        
    Returns:
    --------
    astropy.table.Table
        A table containing the vari_summary parameters (statistics, classification flags, etc.).
    """
    
    # Convert all IDs to strings for the query
    ids_str = ','.join([str(sid) for sid in source_ids])
    
    # Construct the ADQL query
    # SELECT * retrieves all columns (statistical parameters + boolean flags)
    query = f"""
    SELECT *
    FROM gaiadr3.vari_summary
    WHERE source_id IN ({ids_str})
    """
    
    print(f"Querying Gaia Variability Summary table for {len(source_ids)} sources...")
    
    try:
        # Launch the query asynchronously
        job = Gaia.launch_job_async(query)
        
        # Get the results
        results = job.get_results()
        
        return results
        
    except Exception as e:
        print(f"An error occurred during the Vari Summary query: {e}")
        return None

if __name__ == "__main__":
    # Example list of Gaia DR3 Source IDs
    # (These are real IDs for some well-known stars)
    my_source_ids = [5937083204838582784,5912901443721234304,5912575537299009536]
    
    # 1. Query Gaia Source Parameters
    table_source = query_gaia_params(my_source_ids)
    
    if table_source is not None:
        print("\n--- Gaia Source Results ---")
        # Print specific columns to verify calculation
        if 'g_abs_mag' in table_source.columns:
            print(table_source['source_id', 'phot_g_mean_mag', 'parallax', 'g_abs_mag'])
        else:
            print(table_source)

    # 2. Query Gaia Variability Summary
    table_vari = query_vari_summary(my_source_ids)
    
    if table_vari is not None:
        print("\n--- Gaia Variability Summary Results ---")
        if len(table_vari) > 0:
            # Print a few key statistical columns if they exist
            cols_to_show = ['source_id', 'num_selected_g_fov', 'mean_mag_g_fov', 'std_dev_mag_g_fov', 'abbe_mag_g_fov']
            # Only show columns that are actually in the result (in case some IDs have no vari data)
            available_cols = [col for col in cols_to_show if col in table_vari.columns]
            print(table_vari[available_cols])
        else:
            print("No variability summary data found for these source IDs.")