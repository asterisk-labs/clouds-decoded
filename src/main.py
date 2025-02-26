import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from data import heightsToOffsets, getValidFootprintIDs, getValidFootprintShape, \
    extractAndRotateFootprint, getFootprints, getOrbitImageAngle, getBands
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

def processColumnStrip(
        column_bands,
        column_coords,
        config,
        direction,
        brightness_mask=True,
    ):
    """
    Takes the column of pixels, assumed to be fairly thin across. 
    We then upsample the data along the track direction, and correlate the data at each height for each patch down the column.

    Parameters:
        column_bands: dict, Dictionary of bands in the column
        column_coords: np.array, Array of coordinates of the pixels in the column
        config: CloudHeightConfig, Configuration object
        direction: str, Direction of the column
        brightness_mask: bool, Whether to use a mask for brightness

    Returns:
        retrieved_heights: np.array, Array of retrieved heights
        retrieved_coords: np.array, Array of coordinates of the retrieved heights
    """

    # Unpack the configuration
    reference_band = config.reference_band
    original_resolution = BAND_RESOLUTIONS[reference_band]
    convolved_size_along_track = config.convolved_size_along_track
    along_track_upsampled_resolution = config.convolution_upsampling_resolution
    along_track_stride = config.stride
    heights = config.heights
    max_height = config.max_height
    cloudy_thresh = config.cloudy_thresh
    threshold_band = config.threshold_band
    

    along_track_upsampling_rate = int(original_resolution / along_track_upsampled_resolution)
    along_track_size = convolved_size_along_track // along_track_upsampled_resolution 
    along_track_stride = along_track_stride // along_track_upsampled_resolution
    pixel_size = original_resolution / along_track_upsampling_rate

    # Upsample all the data along the track (vertical) direction
    if along_track_upsampling_rate != 1:
        column_bands = {
            name:skimage.transform.resize(
                band,
                (band.shape[0]*along_track_upsampling_rate,band.shape[1]),
                order=1,
                anti_aliasing=False
            ) for name,band in column_bands.items()
        }

    # Create mask for brightness
    if brightness_mask:
        mask = column_bands[threshold_band] > cloudy_thresh
        
    # Take gradients of the bands
    column_bands = {
        name: np.stack((
            np.gradient(band,axis=0), 
            np.gradient(band,axis=1)/along_track_upsampling_rate # TODO: Check if these gradients have same relative power along and across track
        ), axis=-1) for name,band in column_bands.items()
    }
    # TODO: Check if block is actually finding correct range that heights should be calculated for...

    # Find the max offset that we need to consider
    max_offset = heightsToOffsets(
        [max_height]*len(column_bands),
        column_bands.keys(),
        pixel_size)
    max_offset = int(np.ceil(max_offset.max()))

    # Using max offset, we determine the range of centres that we can use
    centres = np.arange(
        along_track_size+max_offset,
        column_bands[reference_band].shape[0]-along_track_size-max_offset,
        along_track_stride
    )

    ####################### 

    # Extract the coordinates of the centres
    width = column_bands[reference_band].shape[1]
    centre_x = width//2
    extracted_coords = column_coords[centres//along_track_upsampling_rate,centre_x]
    retrieved_heights = []
    retrieved_coords = []
    for centre,coord in zip(centres,extracted_coords):
        # Check if reflectance is > 0.2 in the reference band
        if brightness_mask:
            if mask[centre,centre_x] == 0:
                continue
            scores = correlateAtHeights(
                column_bands,
                centre,
                along_track_size,
                heights,
                direction,
                pixel_size,
                reference_band=reference_band
            )
            height = heights[np.argmax(scores)]
            retrieved_heights.append(height)
            retrieved_coords.append(coord)
    return retrieved_heights,retrieved_coords


def processFootprint(footprint_bands,footprint_coords,footprint_direction,conf):
    """
    Process a footprint, extracting the heights at each point

    Parameters:
        footprint_bands: dict, Dictionary of bands in the footprint
        footprint_coords: np.array, Array of coordinates of the pixels in the footprint
        footprint_direction: str, Direction of the footprint
        conf: CloudHeightConfig, Configuration object

    Returns:
        retrieved_heights: np.array, Array of retrieved heights
        retrieved_coords: np.array, Array of coordinates of the retrieved heights
    """

    col_width = int(conf.convolved_size_across_track // conf.footprint_resolution)
    col_step = int(conf.stride // conf.footprint_resolution)
    col_starts = np.arange(0,footprint_bands[conf.reference_band].shape[1]-col_width,col_step)

    retrieved_heights = []
    retrieved_coords = []
    for i,col_start in tqdm(enumerate(col_starts),total=len(col_starts)):
        col_extracted_pair = {}
        for name,band in footprint_bands.items():
            col_extracted_pair[name] = band[:,col_start:col_start+col_width]
        col_extracted_points = footprint_coords[:,col_start:col_start+col_width]
        
        col_heights,col_coords = processColumnStrip(
            col_extracted_pair,
            col_extracted_points,
            conf,
            footprint_direction,
            brightness_mask=True
        )
        retrieved_heights.extend(col_heights)
        retrieved_coords.extend(col_coords)
    
    return retrieved_heights,retrieved_coords

# retrieved_heights,retrieved_coords = processFootprint(extracted,extracted_points,params)

# # 3D plot of the retrieved heights
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(retrieved_coords[:,0],retrieved_coords[:,1],retrieved_heights,c=retrieved_heights)
# plt.show()


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

    fps = getFootprints(conf.scene_dir,bands=conf.bands)
    footprint_ids = getValidFootprintIDs(fps)
    image_angle = getOrbitImageAngle(conf.scene_dir)

    band_data = getBands(conf.scene_dir,conf.bands)

    final_heights = []
    final_coords = []
    for fp_id in footprint_ids[1:-1]:
        if fp_id % 2 == 0:
            direction = 'up'
        else:
            direction = 'down'
        valid_footprint = getValidFootprintShape(fps, fp_id)
        extracted,extracted_points = extractAndRotateFootprint(
            band_data,
            valid_footprint,
            image_angle,
            conf.footprint_resolution,
            reference_band=conf.reference_band
        )
        retrieved_heights,retrieved_coords = processFootprint(extracted,extracted_points,direction,conf)
        final_heights.extend(retrieved_heights)
        final_coords.extend(retrieved_coords)
    final_heights = np.array(final_heights)
    final_coords = np.array(final_coords)
    return final_heights,final_coords

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

    final_heights,final_coords = processScene(config)
    final_heights,final_coords = np.load('output.npz')['heights'],np.load('output.npz')['coords']
    if args.plot:
        plot_height(config,final_heights,final_coords)
    if args.output is not None:
        np.savez(args.output,heights=final_heights,coords=final_coords)