import xskillscore as xs

output_dir = "/home/jupyter-xavi/results/CRPS"

# Load observations (assuming you have them)
obs = "./TrainValTestSplit/train_era5_files.pkl"

# Dictionary of ensemble forecasts
ensembles = {
    "Tformer_all": Tformer_all,
    #"Tformer_one": Tformer_one,
    #"MBM_all": MBM_all,
    #"MBM_one": MBM_one,
    #"Raw": raw_fcs
}

# Process each ensemble
for name, ensemble in ensembles.items():
    print(f"Processing {name}...")
    
    obs_broadcast = observations
    
    # Initialize arrays to store CRPS values
    # Assuming we compute CRPS for each leadtime
    n_leadtimes = ensemble.shape[2]
    gaussian_crps = np.zeros(n_leadtimes)
    fair_crps = np.zeros(n_leadtimes)
    
    # Compute CRPS for each leadtime
    for lt in range(n_leadtimes):
        
        # Extract data for this leadtime
        ensemble_lt = ensemble[:, :, lt, :, :]  # (member, time, lat, lon)
        obs_lt = obs_broadcast[:, lt, :, :]  # (time, lat, lon)
        
        # Convert to xarray for xs.crps_gaussian
        ensemble_xr = xr.DataArray(
            ensemble_lt, 
            dims=["member", "time", "latitude", "longitude"]
        )
        obs_xr = xr.DataArray(
            obs_lt,
            dims=["time", "latitude", "longitude"]
        )
        
        # Compute Gaussian CRPS
        # Mean and standard deviation along member dimension
        mean = ensemble_xr.mean(dim="member")
        std = ensemble_xr.std(dim="member")
        g_crps = xs.crps_gaussian(obs_xr, mean, std)
        gaussian_crps[lt] = g_crps.mean().values
        
        # Compute fair ensemble CRPS
        f_crps = fair_crps_ensemble(obs_lt, ensemble_lt, axis=0)
        fair_crps[lt] = np.mean(f_crps)
    
    # Save results
    np.save(f"{output_dir}/{name}_gaussian_crps.npy", gaussian_crps)
    np.save(f"{output_dir}/{name}_fair_crps.npy", fair_crps)
    

print("All CRPS computations complete!")