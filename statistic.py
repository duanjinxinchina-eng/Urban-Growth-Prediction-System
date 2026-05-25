from PIL import Image
import os
import sys
import glob
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import imageio.v2 as imageio
from matplotlib.colors import ListedColormap, BoundaryNorm
from bokeh.io.export import export_png
import holoviews as hv
from holoviews import opts

workspace = sys.argv[1]
begin_year = sys.argv[2]
begin_year = int(begin_year)
end_year = sys.argv[3]
end_year = int(end_year)
years = sys.argv[4]
years = int(years)

def save_raster(data_shp, pred_csv):
        shp_file = os.path.join(workspace, data_shp)
        csv_file = os.path.join(workspace, pred_csv)
        output_dir = os.path.join(workspace, 'raster_output')

        gdf = gpd.read_file(shp_file)
        data = pd.read_csv(csv_file)
        columns = data.columns[1:]
        # print(columns)

        for col in columns:
            gdf[col] = data[col]

        # WGS 84 / UTM 49N（EPSG:32649）
        gdf = gdf.set_crs(epsg=32649, allow_override=True)

        # =================point → raster==============================
        # raster parameter
        pixel_size = 300  # 300m
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width = int((xmax - xmin) / pixel_size)
        height = int((ymax - ymin) / pixel_size)
        transform = from_origin(xmin, ymax, pixel_size, pixel_size)

        os.makedirs(output_dir, exist_ok=True)

        # create raster for every land use map

        for year in columns[0:years + 1]:
            out_array = np.full((height, width), np.nan, dtype=np.float32)

            for idx, row in gdf.iterrows():
                col = int((row.geometry.x - xmin) / pixel_size)
                row_ = int((ymax - row.geometry.y) / pixel_size)
                if 0 <= row_ < height and 0 <= col < width:
                    out_array[row_, col] = row[year]

            out_path = os.path.join(output_dir, f"predicted_{year}.tif")

            with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype='float32',
                    crs=gdf.crs,
                    transform=transform,
                    nodata=np.nan,
            ) as dst:
                dst.write(out_array, 1)


        out_array = np.full((height, width), np.nan, dtype=np.float32)

        for idx, row in gdf.iterrows():
            col = int((row.geometry.x - xmin) / pixel_size)
            row_ = int((ymax - row.geometry.y) / pixel_size)
            if 0 <= row_ < height and 0 <= col < width:
                out_array[row_, col] = row[str(end_year + 1)]

        out_path = os.path.join(output_dir, f"actual_{end_year}.tif")

        with rasterio.open(
                out_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs=gdf.crs,
                transform=transform,
                nodata=np.nan,
        ) as dst:
            dst.write(out_array, 1)

def show_pred():
    colors = {
        0: '#f4f3c3',
        1: '#4c983a',
        2: '#a3d6f5',
        3: '#e56766',
    }
    labels = {
        0: 'Agricultural land',
        1: 'Vegetation',
        2: 'Water bodies',
        3: 'Built-up'
    }


    color_list = [colors[i] for i in sorted(colors.keys())]
    cmap = ListedColormap(color_list)
    bounds = list(range(len(colors) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    raster_dir = os.path.join(workspace, "raster_output")
    output_dir = os.path.join(raster_dir, 'output_images')
    output_gif = os.path.join(output_dir, "predicted_animation.gif")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_files = sorted(glob.glob(os.path.join(raster_dir, "predicted_*.tif")))
    frames = []

    figsize = (5, 4)
    dpi = 300

    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            year = os.path.basename(tif_file).split('_')[-1].split('.')[0]

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(data, cmap=cmap, norm=norm)
            ax.set_title(f"Former and Predicted Land Use - {year}", fontsize=14)
            ax.axis('off')

            output_png_path = os.path.join(output_dir, f"Former and predicted_{year}.png")
            plt.savefig(output_png_path, dpi=dpi, facecolor='white', bbox_inches='tight')
            plt.close()


            frames.append(imageio.imread(output_png_path))

    imageio.mimsave(output_gif, frames)

    actual_path = os.path.join(raster_dir, f"actual_{end_year}.tif")
    with rasterio.open(actual_path) as src:
        data = src.read(1)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(f"Actual Land Use - {end_year}", fontsize=14)
        ax.axis('off')

        output_png_path = os.path.join(output_dir, "Actual_end_year.png")
        plt.savefig(output_png_path, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close()
        im = Image.open(output_png_path)
        image_file = os.path.join(output_dir, "Actual_end_year.gif")
        im.save(image_file, format='GIF')

def show_chord():
    hv.extension('bokeh')


    node_colors = {
        0: '#f4f3c3',
        1: '#4c983a',
        2: '#a3d6f5',
        3: '#e56766'
    }

    node_labels = {
        0: 'Agricultural land',
        1: 'Vegetation',
        2: 'Water bodies',
        3: 'Built-up'
    }


    file = os.path.join(workspace, 'predicted.csv')
    df = pd.read_csv(file)
    beginyear = str(begin_year)
    endyear = str(end_year + 1)
    relation_matrix = df.pivot_table(index=beginyear, columns=endyear, aggfunc='size', fill_value=0)

    relation_matrix = relation_matrix.reset_index()
    data = relation_matrix.melt(id_vars=beginyear)
    data = data[data['value'] > 0].copy()
    data['value'] = np.log1p(data['value'])

    data['edge_color'] = data[beginyear].map(node_colors)



    node = pd.DataFrame(data[beginyear].unique(),
                        columns=['node'])
    node['label'] = node['node'].map(node_labels)
    node['node_color'] = node['node'].map(node_colors)


    nodes = hv.Dataset(node, 'node', )
    chord = hv.Chord((data, nodes),
                     [beginyear, endyear],
                     ['value', 'edge_color'])

    chord.opts(
        opts.Chord(
            node_color='node_color',
            edge_color='edge_color',
            height=500,
            width=500,
            labels=None,
            edge_alpha=0.8,
            node_size=15,
            edge_visible=True,
            directed=False,
            toolbar=None,
            edge_line_width=8
        )
    )
    output_dir = workspace
    output = os.path.join(output_dir, "chord.html")
    hv.save(chord, output)

    bokeh_obj = hv.render(chord, backend='bokeh')
    export_png(bokeh_obj, filename="chord_diagram.png")
    im = Image.open("chord_diagram.png")

    image_file = os.path.join(output_dir, "chord_diagram.gif")
    im.save(image_file, format='GIF')

def show_radial():
    file = os.path.join(workspace, 'data.csv')
    df = pd.read_csv(file)


    ori_counts = df['ori'].value_counts().sort_index()
    tgt_counts = df['target'].value_counts().sort_index()

    all_types = sorted(set(ori_counts.index).union(tgt_counts.index))

    data = pd.DataFrame({
        'LITH': all_types,
        'ori_COUNT': [ori_counts.get(i, 0) for i in all_types],
        'target_COUNT': [tgt_counts.get(i, 0) for i in all_types],
    })
    data = data.sort_values(by='target_COUNT', ascending=True).reset_index(drop=True)

    node_colors = {
        1: '#f4f3c3',
        2: '#4c983a',
        5: '#a3d6f5',
        8: '#e56766',
    }

    node_labels = {
        1: 'Agricultural land',
        2: 'Vegetation',
        5: 'Water bodies',
        8: 'Built-up'
    }

    data['label'] = data['LITH'].map(node_labels)
    data['color'] = data['LITH'].map(node_colors)
    data['log_ori'] = np.log1p(data['ori_COUNT'])
    data['log_target'] = np.log1p(data['target_COUNT'])

    max_log_value = max(data['log_ori'].max(), data['log_target'].max())

    fig = plt.figure(figsize=(10, 10))
    rect = [0.1, 0.1, 0.8, 0.8]

    ax_polar_bg = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar_bg.set_theta_zero_location('N')
    ax_polar_bg.set_theta_direction(1)
    ax_polar_bg.axis('off')

    for i in range(len(data)):
        ax_polar_bg.barh(i + 0.2, max_log_value * 1.2 * np.pi / max_log_value, color='grey', alpha=0.05)
        ax_polar_bg.barh(i - 0.2, max_log_value * 1.2 * np.pi / max_log_value, color='grey', alpha=0.05)

    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(1)

    ring_labels = [f'   {x}  ' for x in data['label']]
    ax_polar.set_rgrids([0, 1, 2, 3],
                        labels=ring_labels,
                        angle=0,
                        fontsize=30,
                        fontweight='bold',
                        color='black',
                        verticalalignment='center',
                        ha='left')

    for i in range(len(data)):
        ax_polar.barh(
            i - 0.2,
            data['log_ori'][i] * 1.2 * np.pi / max_log_value,
            color=data['color'][i],
            alpha=0.9,
            label='ori' if i == 0 else None
        )

    for i in range(len(data)):
        ax_polar.barh(
            i + 0.2,
            data['log_target'][i] * 1.2 * np.pi / max_log_value,
            color=data['color'][i],
            alpha=0.4,
            edgecolor='black',
            linewidth=1.2,
            label='target' if i == 0 else None
        )

    ax_polar.grid(False)
    ax_polar.tick_params(axis='both', left=False, bottom=False,
                         labelbottom=False, labelleft=True)


    handles, _ = ax_polar.get_legend_handles_labels()

    plt.savefig('./png/grouped_radial_log.png', bbox_inches='tight')
    im = Image.open('./png/grouped_radial_log.png')
    output_dir = workspace
    image_file = os.path.join(output_dir, 'grouped_radial_log.gif')
    im.save(image_file, format='GIF')

fishmap = "fishmap_labelclip.shp"
pred = "predicted.csv"
save_raster(fishmap, pred)
show_pred()
show_chord()
show_radial()