import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data import heightsToOffsets, Sentinel2Scene, ColumnExtractor, getBands
from config import CloudHeightConfig
from constants import BAND_RESOLUTIONS


def correlateAtHeight(footprint_bands,centre,along_track_size,height,direction,pixel_size,remove_mean=False,reference_band='B02'):
    # Find the correlation of all the data at a given height around a given centre point along a column
    # The centre is in the rotated footprint pixel coordinates, so that everything here is simple and efficient
    offsets = heightsToOffsets([height]*len(footprint_bands),footprint_bands.keys(),pixel_size=pixel_size)

    if direction == 'up':
        offsets = -offsets

    patches = []
    for i,band in enumerate(footprint_bands.items()):
        name,data = band
        if name == reference_band:
            ref_n = i
        offset = offsets[i]
        along_track_start = int(centre - along_track_size/2 - offset)
        along_track_end = along_track_start + along_track_size

        patch = data[along_track_start:along_track_end,:]
        patches.append(patch)

    # Now we have the patches, we can correlate them. We want all pairwise correlations at each pixel summed
    corr = 0
    patch = patches[ref_n]
    if remove_mean:
        patch = ( patch - np.mean(patch) )
    for i,other_patch in enumerate(patches):
        if i != ref_n:
            if remove_mean:
                other_patch = ( other_patch - np.mean(other_patch) )
            corr += np.sum(patch * other_patch)
    return corr

def correlateAtHeights(footprint_bands,centre,along_track_size,heights,direction,pixel_size,reference_band='B02'):
    # Find the correlation of all the data at a given height around a given centre along a column
    # The centre point is in the rotated footprint pixel coordinates, so that everything here is simple and efficient
    scores = []
    for height in heights:
        scores.append(
            correlateAtHeight(
                footprint_bands,
                centre,along_track_size,
                height,
                direction,
                pixel_size,
                reference_band=reference_band
            )
        )
    return np.array(scores)

def processColumn(
        column,
        config,
        brightness_mask=True,
    ):
    """
    Takes the column of pixels, assumed to be fairly thin across.

    Parameters:
        column: Column object
        config: CloudHeightConfig, Configuration object
        brightness_mask: bool, Whether to use a mask for brightness

    Returns:
        retrieved_heights: np.array, Array of retrieved heights
        retrieved_coords: np.array, Array of coordinates of the retrieved heights
    """

    # Unpack the configuration
    reference_band = config.reference_band
    convolved_size_along_track = config.convolved_size_along_track
    along_track_resolution = config.along_track_resolution
    across_track_resolution = config.across_track_resolution
    along_track_stride = config.stride
    heights = config.heights
    max_height = config.max_height
    cloudy_thresh = config.cloudy_thresh
    threshold_band = config.threshold_band
    
    along_track_size = convolved_size_along_track // along_track_resolution 
    along_track_stride = along_track_stride // along_track_resolution
    gradient_power_ratio = along_track_resolution / across_track_resolution

    # Create mask for brightness
    if brightness_mask:
        mask = column.getMask(threshold_band,cloudy_thresh)
        
    # Take gradients of the bands
    if config.target_features == "reflectance":
        target_features = column.bands
    elif config.target_features == "gradient":
        target_features = column.getGradients(gradient_power_ratio)
    # TODO: Check if block is actually finding correct range that heights should be calculated for...

    # Find the max offset that we need to consider
    max_offset = heightsToOffsets(
        [max_height]*len(target_features),
        target_features.keys(),
        along_track_resolution
    )
    max_offset = int(np.ceil(max_offset.max()))
    if column.direction == 'up':
        centres = np.arange(
            along_track_size//2,
            column.bands[reference_band].shape[0]-along_track_size//2-max_offset,
            along_track_stride
            )
    else:
        centres = np.arange(
            along_track_size//2+max_offset,
            column.bands[reference_band].shape[0]-along_track_size//2,
            along_track_stride
            )

    # Extract the coordinates of the centres
    width = target_features[reference_band].shape[1]
    centre_x = width//2
    extracted_coords = column.points[centres,centre_x,:]
    retrieved_heights = []
    retrieved_coords = []
    for centre,coord in zip(centres,extracted_coords):
        # Check if reflectance is > 0.2 in the reference band
        if brightness_mask:
            if mask[centre,centre_x] == 0:
                continue
            scores = correlateAtHeights(
                target_features,
                centre,
                along_track_size,
                heights,
                column.direction,
                along_track_resolution,
                reference_band=reference_band
            )
            height = heights[np.argmax(scores)]
            retrieved_heights.append(height)
            retrieved_coords.append(coord)
    return retrieved_heights,retrieved_coords

def processScene(config_file=None):
    """
    Process a scene, extracting the heights at each point

    Parameters:
        config_file: str or CloudHeightConfig, Path to the configuration file or configuration object

    Returns:
        final_heights: np.array, Array of retrieved heights
        final_coords: np.array, Array of coordinates of the retrieved heights
    """
    if isinstance(config_file,str) or config_file is None:
        conf = CloudHeightConfig(config_file=config_file)
    elif isinstance(config_file,CloudHeightConfig):
        conf = config_file
    else:
        raise ValueError("config_file must be a string or CloudHeightConfig object")

    scene = Sentinel2Scene(conf.scene_dir)
    column_extractor = ColumnExtractor(scene,conf,conf.hack_image_azimuth)

    times = {'extraction':0,'processing':0}
    
    final_heights, final_coords = [], []
    for idx in tqdm(range(len(column_extractor))):
        t0 = time.time()
        col = column_extractor[idx]
        times['extraction'] += time.time()-t0

        if col is None:
            continue
        t0 = time.time()
        retrieved_heights,retrieved_coords = processColumn(
            col,
            conf,
            brightness_mask=True
            )
        times['processing'] += time.time()-t0
        if retrieved_heights is not None:
            final_heights.extend(retrieved_heights)
            final_coords.extend(retrieved_coords)
    return np.array(final_heights),np.array(final_coords), times


def debug_plot(extracted,extracted_points,footprint_id):
    """
    Debug plot of the extracted data
    """
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    c = extracted['B02'].ravel()
    print(extracted_points.shape)
    print(extracted['B02'].shape)
    ax.scatter(extracted_points[...,0].ravel(),109800-extracted_points[...,1].ravel(),c=c,cmap='plasma',s=3,alpha=0.5)
    ax.set_xlim([0,109800])
    ax.set_ylim([0,109800])
    fig.savefig(f'debug_{footprint_id}.png')
    plt.close(fig)
    return None

def plot_height(config,final_heights,final_coords):
    """
    Simple display of the heights found in the scene

    Parameters:
        config: CloudHeightConfig, Configuration object
        final_heights: np.array, Array of retrieved heights
        final_coords: np.array, Array of coordinates of the retrieved heights

    Returns:
        None
    """
    rgb = getBands(config.scene_dir, ['B04','B03','B02'])
    rgb = (np.stack([rgb['B04'],rgb['B03'],rgb['B02']],axis=-1)-1000)/10_000
    rgb = np.clip((rgb**0.7),0,1) # Gamma correction
    resolution = BAND_RESOLUTIONS[config.reference_band]
    point_size = 6*(config.stride/500)
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    ax.imshow(rgb,extent=[0,rgb.shape[1]*resolution,0,rgb.shape[0]*resolution])

    mask = final_heights > 0
    final_heights = final_heights[mask]
    final_coords = final_coords[mask]

    final_heights_log = np.log10(final_heights)
    ax.scatter(final_coords[:,0],rgb.shape[1]*resolution-final_coords[:,1],c=final_heights_log,cmap='jet',s=point_size,alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    if config.plot_writeto is not None:
        fig.savefig(config.plot_writeto)
    plt.show()
    

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process a scene to retrieve cloud heights')
    parser.add_argument('--config','-c',type=str,help='Path to the configuration file')
    parser.add_argument('--output','-o',type=str,help='Path to the output file')
    parser.add_argument('--plot','-p',action='store_true',help='Plot the heights')
    args = parser.parse_args()

    config = CloudHeightConfig(args.config)

    final_heights,final_coords,times = processScene(config)
    print(times)
    # final_heights,final_coords = np.load('output.npz')['heights'],np.load('output.npz')['coords']
    if args.plot:
        plot_height(config,final_heights,final_coords)
    if args.output is not None:
        np.savez(args.output,heights=final_heights,coords=final_coords)
        
