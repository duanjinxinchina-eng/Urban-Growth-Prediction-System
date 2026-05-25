import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from rasterio.plot import show
import cartopy.crs as ccrs
import contextily as ctx
from PIL import Image



photo_number = sys.argv[1]
file = sys.argv[2]
save_name = sys.argv[3]
workspace = sys.argv[4]

def show_study_area(file):
    gdf = gpd.read_file(file)

    if not gdf.crs == 4329:
        gdf_wgs84 = gdf.copy()
        gdf_wgs84 = gdf_wgs84.to_crs(epsg=4329)  # WGS84
        gdf = gdf_wgs84

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # projection 4329

    # study area
    gdf.plot(
            ax=ax,
            edgecolor='r',
            linewidth=1.5,
            facecolor='lightblue',
            alpha=0.4,
            transform=ccrs.PlateCarree()
    )

    ax.set_frame_on(False)

    # base map
    ctx.add_basemap(
            ax,
            alpha=0.9,
            crs=gdf.crs.to_string(),
            source='https://c.tile.thunderforest.com/cycle/{z}/{x}/{y}.png?apikey=41f4f936f1d148f69cbd100812875c88',
            zoom=10,
            transform=ccrs.PlateCarree()
    )


    # grid lines
    gl = ax.gridlines(
            linestyle='--',
            draw_labels=True,
            colors='black'
    )
    gl.top_labels = False
    gl.left_labels = False

    plt.savefig(
            './png/study_area.png',
            bbox_inches='tight',
            transparent=True
    )
    img = Image.open('./png/study_area.png')
    image_file = os.path.join(workspace, 'study_area.gif')
    img.save(image_file, 'GIF', save_all=True)

def show_landuse(file, title, save_name):
    colors = {
        1: '#f4f3c3',
        2: '#4c983a',
        5: '#a3d6f5',
        8: '#e56766'
    }

    labels = {
        1: 'Agricultural land',
        2: 'Vegetation',
        5: 'Water bodies',
        8: 'Built-up'
    }
    # Construct a color mapping
    sorted_keys = sorted(colors.keys())  # [1, 2, 5, 8]
    color_list = [colors[i] for i in sorted_keys]
    cmap = ListedColormap(color_list)
    cmap.set_bad(color='white') # new added

    # Construct the boundary of the actual classification values
    bounds = sorted_keys + [sorted_keys[-1] + 1]  # [1, 2, 5, 8, 9]
    norm = BoundaryNorm(bounds, len(color_list))

    # load data
    full_name = os.path.join(workspace, file)
    with rasterio.open(full_name) as src:
        data = src.read(1)
        # data = np.ma.masked_equal(data, 255)
        data = np.ma.masked_where(~np.isin(data, sorted_keys), data)


    # fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    show(data, ax=ax, cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in sorted_keys]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)

    # save figure
    plt.tight_layout()
    # plt.savefig('./png/'+save_name+".png", dpi=300, bbox_inches='tight')
    plt.savefig('./png/' + save_name + ".png", dpi=300, bbox_inches='tight', facecolor='white')
    im = Image.open('./png/'+save_name+'.png')
    image_file = os.path.join(workspace, save_name + '.gif')
    im.save(image_file, format='GIF')

def add_north(ax, labelsize=18, loc_x=0.88, loc_y=0.85, width=0.04, height=0.13, pad=0.14):
    """
    Draw a scale bar with an 'N' (north) annotation.

    Main parameters:
    :param ax: The target Axes object to draw on (e.g., obtained via plt.gca()).
    :param labelsize: Font size of the 'N' label.
    :param loc_x: Horizontal position of the annotation, defined as the proportion of the Axes width (centered at the bottom of the text).
    :param loc_y: Vertical position of the annotation, defined as the proportion of the Axes height (centered at the bottom of the text).
    :param width: Width of the north arrow as a proportion of the Axes.
    :param height: Height of the north arrow as a proportion of the Axes.
    :param pad: Spacing between the text and the symbol, as a proportion of the Axes.
    :return: None
    """

    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
    right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
    top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
    center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen * loc_x,
            y=miny + ylen * (loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)

colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']  # from low to high
cmap = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)
def show_dem_slope(save_name):

    colors = ['#f58c86', '#f6d586', '#f3ed99', '#669a38', '#6897cf']  # from low to high
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1])

    # load data
    full_dem_name = os.path.join(workspace, 'dem.tif')
    full_slope_name = os.path.join(workspace, 'slope.tif')
    dem = rasterio.open(full_dem_name)
    slope = rasterio.open(full_slope_name)

    # show figuure
    ax1 = fig.add_subplot(gs[0, 0])  # DEM
    ax2 = fig.add_subplot(gs[0, 1])  # Slope
    show(dem, ax=ax1, cmap=cmap, title='DEM')
    show(slope, ax=ax2, cmap=cmap, title='Slope')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    add_north(ax1)
    add_north(ax2)

    # color bar
    ax_colorbar = fig.add_subplot(gs[1, :])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_colorbar.imshow(gradient, aspect='auto', cmap=cmap)
    ax_colorbar.set_xticks([0, 128, 255])
    ax_colorbar.set_xticklabels(['Low', 'Mid', 'High'])
    ax_colorbar.set_yticks([])
    ax_colorbar.set_title("Color Scale", fontsize=10)

    # overal layout
    plt.tight_layout()
    fig.savefig('./png/'+save_name+'.png', dpi=300, bbox_inches='tight')

    im = Image.open('./png/'+save_name+'.png')
    image_file = os.path.join(workspace, save_name + '.gif')
    im.save(image_file, format='GIF')

    plt.close(fig)
def show_feature(file, save_name):
    full_name = os.path.join(workspace, file)
    eu = rasterio.open(full_name)

    fig, (ax1, ax2) = plt.subplots(
        2,1,
        figsize=(10, 8),
        gridspec_kw={'height_ratios': [20, 1]},)
    show(eu, ax=ax1, cmap=cmap)
    title = save_name.replace("_", " ")
    ax1.set_title(title)
    ax1.set_xticks([])
    ax1.set_yticks([])
    add_north(ax1)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax2.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticks([0, 128, 255], labels=['Low', 'Mid', 'High'])

    fig.tight_layout()
    fig.savefig('./png/'+save_name+'.png', dpi=150, bbox_inches='tight')

    im = Image.open('./png/'+save_name+'.png')
    image_file = os.path.join(workspace, save_name + '.gif')
    im.save(image_file, format='GIF')

    plt.close(fig)


if __name__ == "__main__":
    if photo_number == '1': # 绘制研究区位
        show_study_area(file)
    elif photo_number == '2': # 绘制原始土地利用
        show_landuse(file, 'Former land use', save_name)
    elif photo_number == '3':
        show_landuse(file, 'Latter land use', save_name)
    elif photo_number == '4':
        show_landuse(file, 'pred', 'Predicted land use', save_name)
    elif photo_number == '5': # 绘制dem和slope
        show_dem_slope(save_name)
    elif photo_number == '6': # 绘制eu
        show_feature(file, save_name)