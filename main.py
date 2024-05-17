import numpy as np

from utils import shapefile_patches
from utils import forecast_extent
from utils import mapploter
from utils import windmap_ploter
from utils import textplacer
from utils import logoplacer

from utils import make_ist_time
from utils import make_ist_time_video
from utils import foldercreator
from utils import videomaker
from utils import make_ncfiles_fortest

## Process switch
do_windmapplot=1

#do_videocreate=1


## plot characters

plotsize={'width':8,'height':8}

mainmappost={'x':0.1,'y':0.16,'width':0.8,'height':0.65}

legendpost={'x':0.35,'y':0.03,'width':0.3,'height':0.03}

plot_extent={'s':7.125,'n':38.88,'w':67.12,'e':98.88}

geo_fe=forecast_extent(plot_extent)

cen_forecast_extent=geo_fe

colorlist=['#0a0f41','#0a0f87','#3737ff','#6e6eff','#a5a5ff','#ffc8c8','#ff6e6e','#ff3737','#ff0000','#a52a2a','#870404']

ws_colorlist=['#20b2aa','#9acd32','#ffd700','#ff8c00','#ff0000','#800000','#330000']

ws_legendtitle='(meters per sec)'

temp_legendtitle='(degrees celcius)'

# define data locations and different folder from each process
plotdirectory='.'

temp_path=plotdirectory+'tempplot/'
wind_path=plotdirectory+'speed/'
win_map=plotdirectory+'windmap/'
precept_path=plotdirectory+'preceptplot/'
inversion_path=plotdirectory+'inversionplot/'

shapefile='ea_ghcf_simple.json'

make_ncfiles_fortest()

cf_ncfile='Sample_Wind_Data.nc'

## text contents
pos_firstitle={'x':0,'y':0.86,'width':1,'height':0.2}
firstitle='First line title'

pos_vartitle={'x':0,'y':0.80,'width':1,'height':0.2}

pos_thirdtitle={'x':0,'y':0.75,'width':1,'height':0.2}
thirdtitle='Second line title'

pos_standardtext={'x':0.18,'y':-0.01,'width':0.3,'height':0.1}
standardtext='sample text'

pos_datasourcetext={'x':-0.14,'y':-0.01,'width':0.3,'height':0.1}
datasourcetext='Output of windspeed'


logofile='test.png'
logopos={'x':0.84, 'y':-0.05, 'width':0.15, 'height':0.2}


## Video configuration
#choose the fpz
#choose the output format. 'mp4' or 'gif'
# choose the file name

chosenfps=3

outputformat='mp4'

##############

fulltimestep=np.arange(0,24)


if do_windmapplot:
    ws_dataclasslist=[2, 4, 6, 8, 10, 12]
    ws_legendlabel=[2, 4, 6, 8, 10, 12]
    variable='wind speed'
    vartitle='wind direction'
    for timestep in fulltimestep:
        fig, df_map=shapefile_patches(shapefile)
        concmap, nctime=windmap_ploter(plot_extent,fig,cf_ncfile,timestep,plotsize,mainmappost,df_map,ws_dataclasslist,legendpost,ws_colorlist,ws_legendlabel,variable,ws_legendtitle)
        """textplacer(firstitle,fig,pos_firstitle['x'],pos_firstitle['y'],pos_firstitle['width'],pos_firstitle['height'],firstitle.title(),18,'center')"""
        print("fig type:", type(fig))
        print("Text:", firstitle.title())
        print("Position and size:", pos_firstitle['x'], pos_firstitle['y'], pos_firstitle['width'], pos_firstitle['height'])
        print("Font size:", 18)
        print("Horizontal alignment:", 'center')
        print("nctime before conversion:", nctime)
try:
        ist_time=make_ist_time(nctime)
        print("IST Time:", ist_time)
except Exception as e:
    print("Error converting time:", str(e))
    
textplacer(fig,firstitle, pos_firstitle['x'], pos_firstitle['y'], pos_firstitle['width'], pos_firstitle['height'], firstitle.title(),18, 'center')
textplacer(fig,vartitle,pos_vartitle['x'],pos_vartitle['y'],pos_vartitle['width'],pos_vartitle['height'],variable.title(),30,'center')
textplacer(fig,thirdtitle,pos_thirdtitle['x'],pos_thirdtitle['y'],pos_thirdtitle['width'],pos_thirdtitle['height'],'On '+ist_time,20,'center')
textplacer(fig,datasourcetext,pos_datasourcetext['x'],pos_datasourcetext['y'],pos_datasourcetext['width'],pos_datasourcetext['height'],datasourcetext,8,'left')
print("Type of fig:", type(fig))
print("Type of logofile:", type(logofile))
print("Content of logopos:", logopos)
logoplacer(fig, logofile, logopos)
logopos = {'x': 0.8, 'y': 0.1, 'width': 0.1, 'height': 0.1}  # Example positions
logofile = '/Users/sinugp/logo.png'
logoplacer(fig, logofile, logopos)
fig.savefig(win_map+'{0}.png'.format(str(timestep)), transparent=False)
fig.clear()
print(timestep)
