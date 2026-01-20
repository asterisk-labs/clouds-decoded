from refl2prop.inference import CloudPropertyInverter
import numpy as np
import os
import rasterio as rio
from skimage.transform import resize


cth_map = np.load('/home/ali/projects/cloud-height-prototype/S2A_MSIL1C_20180609T094031_N0500_R036_T32NLJ_20230715T110315_outputs.npz')['grid'] # 2D array 

scene_dir = '/data/s2get/Sentinel-2/MSI/L1C/2018/06/09/S2A_MSIL1C_20180609T094031_N0500_R036_T32NLJ_20230715T110315.SAFE'

band_dir = scene_dir + '/GRANULE/'
band_dir += [d for d in os.listdir(band_dir)][0] + '/IMG_DATA/'

bands = ['B01', 'B02', 'B04', 'B08', 'B11', 'B12']

for b in bands:
    band_path = band_dir + [f for f in os.listdir(band_dir) if f.endswith(f'_{b}.jp2')][0]
    with rio.open(band_path) as src:
        band_data = (src.read(1).astype(np.float32) - 1000.0) / 10000.0 # Reflectance scaling

    # resize
    band_data = resize(band_data, cth_map.shape, order=1, mode='reflect', anti_aliasing=True)

    if b == 'B01':
        bands_dict = {b: band_data}
    else:
        bands_dict[b] = band_data
    

# 2. Init Inverter
inverter = CloudPropertyInverter("model.pth")

# 3. Prepare Inputs
# For now let's assume 50% shading everywhere
shading = np.ones_like(cth_map) * 0.5 
geometry = {
    'incidence_angle': 34.0,  # Solar Zenith Angle
    'mu': np.cos(np.deg2rad(2.0)), # Viewing Geometry
    'phi': np.deg2rad(45.0)
}
albedos = {}
for b in bands:
    albedos[b] = np.percentile(bands_dict[b], 1) # Example: 5th percentile as albedo, could be improved with better estimates and an actual cloud mask

# 4. Run
results = inverter.predict_scene(
    bands=bands_dict, # {'B01': array, ...}
    surface_albedo=albedos,
    geometry=geometry,
    cloud_top_height=cth_map / 1000.0,  # Convert to km
    shading_ratio=shading
)

print("Tau Map shape:", results['tau'].shape)

# Print actual input ranges from tensor
for i, band in enumerate(['B01', 'B02', 'B04', 'B08', 'B11', 'B12']):
    print(f"Input Band {band}: min={bands_dict[band].min()}, max={bands_dict[band].max()}")
print(f"albedo min/max: {[albedos[b] for b in bands]}")
print(f"cloud_top_height min/max: {np.nanmin(cth_map)}/{np.nanmax(cth_map)}")
print(f"shading_ratio min/max: {shading.min()}/{shading.max()}")

# Print statistics of each output layer
for key in results:
    print(f"{key}: min={np.nanmin(results[key])}, max={np.nanmax(results[key])}, mean={np.nanmean(results[key])}")

import pprint
pprint.pprint(inverter.model.ranges)  # Print input/output ranges used in normalization


# Plotting example (requires matplotlib)
import matplotlib.pyplot as plt

os.makedirs('example_outputs', exist_ok=True)

# Tau map
plt.figure(figsize=(10,10))
plt.imshow(np.log10(results['tau']), cmap='viridis')
plt.colorbar(label='Optical Thickness (Tau)')
plt.title('Retrieved Cloud Optical Thickness')
plt.savefig('example_outputs/retrieved_tau.png')
plt.close()

# Reff Liquid map
plt.figure(figsize=(10,10))
plt.imshow(results['r_eff_liq'], cmap='viridis')
plt.colorbar(label='Effective Radius Liquid (microns)')
plt.title('Retrieved Cloud Effective Radius (Liquid)')
plt.savefig('example_outputs/retrieved_reff_liquid.png')
plt.close()

# Reff Ice map
plt.figure(figsize=(10,10))
plt.imshow(results['r_eff_ice'], cmap='viridis')
plt.colorbar(label='Effective Radius Ice (microns)')
plt.title('Retrieved Cloud Effective Radius (Ice)')
plt.savefig('example_outputs/retrieved_reff_ice.png')
plt.close()     

# Ice/Liquid Ratio map
plt.figure(figsize=(10,10))
plt.imshow(results['ice_liq_ratio'], cmap='RdBu', vmin=0, vmax=1)
plt.colorbar(label='Ice/Liquid Ratio')
plt.title('Retrieved Ice/Liquid Ratio')
plt.savefig('example_outputs/retrieved_ice_liq_ratio.png')
plt.close()


# Make ice composite (B12, B11, B04)
ice_image = np.stack([
    bands_dict['B12'],  # SWIR - Red
    bands_dict['B11'],  # SWIR - Green
    bands_dict['B04']   # Red - Blue
], axis=-1)

ice_image = (ice_image - ice_image.min(axis=(0,1))) / (ice_image.max(axis=(0,1)) - ice_image.min(axis=(0,1)))
ice_image = (ice_image ** 0.65)
plt.figure(figsize=(10,10))
plt.imshow(np.clip(ice_image, 0, 1))
plt.title('Input Ice Composite (B12, B11, B04)')
plt.savefig('example_outputs/input_ice_composite.png')
plt.close()