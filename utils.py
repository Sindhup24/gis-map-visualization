import os

import ntpath
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt

import datetime
from datetime import timedelta, timezone, datetime

import shutil
import rasterio

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import matplotlib.offsetbox as offsetbox

#from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from windmap_streamline import Streamlines
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from moviepy.editor import ImageSequenceClip
from shapely.geometry import box
from shapely.geometry import Point
from descartes import PolygonPatch


def shapefile_patches(shapefile):
    """Reads a shapefile and converts it to patches for matplotlib plotting."""
    gdf = gpd.read_file(shapefile)
    """patches = [PolygonPatch(feature, edgecolor="black", facecolor="none") for feature in gdf.geometry]"""
    # Here you might need to adjust gdf based on plot_extent, cen_forecast_extent, etc.
    # Placeholder logic:
    fig, ax = plt.subplots()
    gdf.plot(ax=ax)  # This is very generic; your actual code might need to be different.

    # Assuming you need to return the figure and the processed geodataframe for further use:
    return fig, gdf
    """return patches"""

def forecast_extent(data):
    """Calculates the geographic extent of the forecast area from data."""
    min_lat, max_lat = data['s'], data['n']
    min_lon, max_lon = data['w'], data['e']
    centroid_lon = (min_lon + max_lon) / 2
    centroid_lat = (min_lat + max_lat) / 2
    return (centroid_lon, centroid_lat)

def mapploter(data, extent):
    """Plots data on a map within the specified extent."""
    plt.figure(figsize=(10, 8))
    plt.imshow(data, extent=extent, interpolation='nearest')
    plt.title('Map Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Data values')
    plt.show()

def windmap_ploter(plot_extent, fig, cf_ncfile, timestep, plotsize, mainmappost, df_map, ws_dataclasslist, legendpost, ws_colorlist, ws_legendlabel, variable, ws_legendtitle):
    """
    A function to plot wind data maps. Adjust the contents to suit actual usage.
    
    Arguments:
    - plot_extent: The geographic extent for the plot.
    - fig: The figure object where the plot will be drawn.
    - cf_ncfile: The NetCDF file containing wind data.
    - timestep: The specific timestep to plot.
    - plotsize: Dimensions of the plot.
    - mainmappost: Main map positioning settings.
    - df_map: Data frame or similar object containing geographic data for the map.
    - ws_dataclasslist: Wind speed data classifications.
    - legendpost: Positioning for the legend.
    - ws_colorlist: List of colors for different wind speeds.
    - ws_legendlabel: Labels for the legend.
    - variable: The variable to be plotted.
    - ws_legendtitle: Title for the legend.
    
    Returns:
    - Tuple of the configured plot and time from the NetCDF file.
    """
    # Implementation depends on data handling and plotting requirements.
    # This is a placeholder for your actual plotting logic.
    return fig, "Time from NetCDF"

def make_ist_time(utc_time):
    """Converts UTC datetime or string to IST (Indian Standard Time), which is UTC +5:30."""
    if isinstance(utc_time, datetime):
        ist_time = utc_time + timedelta(hours=5, minutes=30)
    elif isinstance(utc_time, str):
        try:
            # Try parsing the string with a known format
            utc_time = datetime.strptime(utc_time, '%a %b %d %H:%M:%S %Y')
            ist_time = utc_time + timedelta(hours=5, minutes=30)
        except ValueError:
            # Handle unexpected formats or provide a default
            print(f"Warning: The time data {utc_time} does not match expected format.")
            return utc_time  # or return a default datetime if appropriate
    else:
        raise TypeError("Unsupported type for utc_time. Expected datetime.datetime or str.")

    return ist_time.strftime('%Y-%m-%d %H:%M:%S')  # Optionally convert back to string if needed

def make_ist_time_video(utc_time):
    """Formats UTC time to IST for video use, returns a string."""
    ist_time = make_ist_time(utc_time)
    return ist_time.strftime('%Y-%m-%d %H:%M:%S')

def foldercreator(folder_path):
    """Creates a folder if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return "Folder created: " + folder_path
    else:
        return "Folder already exists: " + folder_path

def videomaker(frames, fps=30, output='output.mp4'):
    """Creates a video from a sequence of image frames."""
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output, fps=fps)
    return "Video created successfully at: " + output

def make_ncfiles_fortest(data=None, output='test.nc'):
    """Generates a NetCDF file from given data for testing purposes."""
    # Assuming 'data' is a dictionary of arrays, typically used with NetCDF4
    if data is None:
        data = {
            'values': [10, 20, 30, 40, 50]  # Default values
        }
    from netCDF4 import Dataset
    root_grp = Dataset(output, 'w', format='NETCDF4')
    # Example of creating dimensions
    root_grp.createDimension('dim', None)
    # Example of creating a variable
    variable = root_grp.createVariable('var', 'f4', ('dim',))
    variable[:] = data['values']
    root_grp.close()
    return "NetCDF file created successfully at: " + output
    
def textplacer(fig, text, x, y, width, height, title, fontsize, ha):
    """Places text on a plot with specific formatting."""
    ax = fig.add_axes([x, y, width, height])  # This assumes that the position and size are meant to define an axes within the figure
    ax.text(0.5, 0.5, title, fontsize=fontsize, ha=ha, va='center')  # Text centered within the new axes
    ax.axis('off')  # Optionally turn off the axis if it's just for placing text


    
    
def colorcoder(colorlist):
    """
    Converts sequence or string of colour code to tupple

    Parameters
    ----------
    colorlist : list
        list of colourcode in form of string or sequence
    
    Returns
    
    RGB tuple of three floats from 0-1

    """
    clconv = matplotlib.colors.ColorConverter().to_rgb
    color_code=[clconv(color) for color in colorlist]
    return color_code


def colorize(array, cmap):
    """
    Helper function for normalizing colourmap 

    Parameters
    ----------
    array : numpy array
        
    cmap: maptlotlib colour map
    
    Returns
    -------
    Normalized color map 
    """
    #normed_data = (array - array.min()) / (array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(array)


def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    Colorbar helper functions, which create legends in the plot
    

    Parameters
    ----------
    ncolors: int
              number of colors in legend
    cmap: matplotlib color map obj
      
    
    Returns
    -------
    returns colorbar object to be used as legend in the plot
    """
    #cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable,orientation='horizontal',**kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar



def custom_legend_placer(cf_ncdata,da_plt,dataclasslist,dataclasslabel,legendpost,colorlist,legendtitle):
    """
    Helper function to place the legend and runs internal colour bar functions

    Parameters
    ----------
    cf_ncdata : int
        Description of arg1
    da_plt: matplotlib plt axes

    dataclasslist: legend cutoff value list

    dataclasslabel: legend label list

    legendpost: legend position descrption dict
    
    colorlist: list of colour list for legend and map ploting
    
    legendtitle: legend title

    Returns
    -------
    none

    """
    color_code=colorcoder(colorlist)
    legendpos=da_plt.axes([legendpost['x'],legendpost['y'],legendpost['width'],legendpost['height']],frame_on=False,zorder=10) 
    colormap = LinearSegmentedColormap.from_list("my_colormap",color_code, N=len(dataclasslist), gamma=1.0)
    class_pdata = np.digitize(cf_ncdata, dataclasslist)
    colored_data = colorize(class_pdata, colormap)
    cb = colorbar_index(ncolors=len(dataclasslabel), cmap=colormap,  labels=dataclasslabel,cax = legendpos);cb.ax.tick_params(labelsize=8)
    cb.ax.set_title(legendtitle,fontsize=8,fontweight='bold') 
    return colored_data, colormap


def fixed_locator_lon(x_min,x_max):
    no_of_grid=round(x_max-x_min)
    step=round(x_max-x_min)/no_of_grid
    locators=np.arange(x_min,x_max,step)
    return locators


def add_gridlines(mainmap,x_min,y_min,x_max,y_max):
    gl = mainmap.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_left = False
    #gl.xlabels_bottom = False
    gl.xlabels_top = False
    #locators=fixed_locator_lon(x_min,x_max)
    gl.xlocator = mticker.FixedLocator([x_min+1,x_max-1])
    #gl.ylocator = mticker.FixedLocator([y_min-1,y_max+1])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}

def logoplacer(fig, logofile, logopos):
    """
    Places logo over a plot.

    Parameters
    ----------
    logofile : file path
        path of the logo
    da_plt : variable
        matplotlib.pyplot as plt , da_plt = plt
    pos_var : dict
        containing x,y,width,height        

    
    Returns
    -------
    
        Description of return value

    """
    # #xpos,ypos,width,height=pos_var['x'],pos_var['y'],pos_var['width'],pos_var['height']
    # logo = inset_axes(mainmap,
    #                 width=1,                     # inch
    #                 height=1,                    # inch
    #                 bbox_transform=mainmap.transAxes, # relative axes coordinates
    #                 bbox_to_anchor=(0.5,0.5),    # relative axes coordinates
    #                 loc=3)    
    #logo= mainmap.inset_axes([0.84, -0.05, 0.15, 0.15],frame_on=False,fc='None',alpha=0)
    #logo = inset_axes(da_plt, width="30%",  height="30%")
    #axins.imshow(Z2, extent=extent, interpolation="nearest",
    #      origin="lower")
    logo = plt.imread(logofile)
    #logo=da_plt.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    #logo=fig.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    logo_axes = fig.add_axes([logopos['x'], logopos['y'], logopos['width'], logopos['height']], frame_on=False)
    logo_axes.imshow(logo)
    logo_axes.axis('off')  # Hide the axes


def get_windspeed(u_pdata1,v_pdata1):
    """
    from : https://github.com/blaylockbk/Ute_WRF/blob/master/functions/wind_calcs.py
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    U=u_pdata1
    V=v_pdata1
    WSPD = np.sqrt(np.square(U)+np.square(V))
    return WSPD



def wm_ws_map(params,windspeed,u_pdata1,v_pdata1,vds):
    """
    Use u and v vector to plot the wind stream line and 
    overlay with wind speed plot. There is a shape file
    is overlayed for to give int boundary 

    Parameters
    ----------
    params : parameter objects
        Object variable for folder name, file name etc
    windspeed : Numpy 2D array
        Wind speed generated from u and v compute, fliped to match the canvas
    u_pdata1 : Numpy 2D array
        Wind u vector, fliped to match the canvas
    v_pdata1 : TYPE
        Wind u vector, fliped to match the canvas
    vds : TYPE
        rasterio object to get extent and other maping descriptions

    Returns
    -------
    Generate map png file

    """
   
    x_min,x_max,y_min,y_max=vds.bounds[0],vds.bounds[2],vds.bounds[1],vds.bounds[3]
    mx = np.linspace(x_min, x_max, vds.width)
    my = np.linspace(y_min, y_max, vds.height)
    ncx, ncy = np.meshgrid(mx, my)
    print('do_windmap')
    #legendpost={'x':0.01,'y':0.03,'width':0.97,'height':0.03}
    dataclasslist=[4, 8, 14, 18, 22, 26]
    dataclasslabel=[4, 8, 14, 18, 22, 26]
    colorlist=['#20b2aa','#9acd32','#ffd700','#ff8c00','#ff0000','#800000','#330000']
    da_plt=plt
    da_plt.figure(figsize=(params.plotsize['width'],params.plotsize['height']))
    mainmap=da_plt.axes((params.mainmappost['x'],params.mainmappost['y'],params.mainmappost['width'],params.mainmappost['height']), projection=ccrs.PlateCarree(),zorder=10)
    #bg_shapes = list(shpreader.Reader(params.background_shpfile).geometries())
    json_file = "../static/ea_ghcf_simple.json"
    with open(json_file, "r") as f:
        geom = json.load(f)
    gdf = gp.GeoDataFrame.from_features(geom)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_geometries(
        gdf["geometry"], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="black"
    )
    add_gridlines(mainmap,x_min,y_min,x_max,y_max)
    mainmap.add_geometries(bg_shapes, ccrs.PlateCarree(),
    edgecolor='black', facecolor='none', alpha=1)
    formatted_dataclasslabel = [ '%.0f' % elem for elem in dataclasslabel ]
    colored_data, colormap=custom_legend_placer(windspeed,da_plt,dataclasslist,formatted_dataclasslabel,params.legendpost,colorlist,'Knot(nm/hr)')
    mainmap.imshow(colored_data, cmap=colormap,extent=(x_min,x_max,y_min,y_max))
    #mainmap.imshow(windspeed)
    s = Streamlines(ncx, ncy, u_pdata1, v_pdata1)
    for streamline in s.streamlines:
        x, y = streamline
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n = len(segments)
        D = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=-1))
        L = D.cumsum().reshape(n,1) + np.random.uniform(0,1)
        C = np.zeros((n,3))
        C[:] = (L*1.5) % 1
        #linewidths = np.zeros(n)
        #linewidths[:] = 1.5 - ((L.reshape(n)*1.5) % 1)
        # line = LineCollection(segments, color=colors, linewidth=linewidths)
        line = LineCollection(segments, color=C, linewidth=0.5)
        #lengths.append(L)
        #colors.append(C)
        #lines.append(line)
        mainmap.add_collection(line)
    textplacer('firstitle',da_plt,params.pos_firstitle,params.firstitle,10,'center')
    datefmt=params.startdate
    run=params.run
    textplacer('vartitle',da_plt,params.pos_vartitle,f'Wind speed in Knot(nm/hr), based on GFS {datefmt} run {run}',10,'center')
    textplacer('thirdtitle',da_plt,params.pos_thirdtitle,'On '+params.istdatefmt+ ' IST',10,'center')
    logoplacer(params.logofile,da_plt,params.logopos)
    output_png_file=params.localpath+'{}.png'.format(params.utcdatefmt)
    da_plt.savefig(output_png_file, transparent=False)
    da_plt.close()
    return output_png_file
