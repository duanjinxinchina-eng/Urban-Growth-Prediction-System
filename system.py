import time
import threading
import arcpy
from arcpy.sa import *
import tkinter as tk
import ttk
from tkinter import messagebox
import tkFileDialog as filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
import scipy.sparse as sp
import numpy as np
from PIL import Image, ImageTk, ImageSequence, ImageDraw
import os
import pandas as pd
import subprocess
import sys


python3_path = r"D:\Anaconda\envs\urbangrowth\python.exe" # Change to your own Python 3 path
def update_status(text, message):
    """updata work record"""

    current_time = time.strftime("%Y/%m/%d %H:%M:%S")  # get time
    text.insert(tk.END, "{} - {}\n".format(current_time, message))  # record
    text.see(tk.END)  # Scroll to the latest record

def update_canvas(photo_number, file, file_name, workspace, ax, canvas):
    """updata canvas"""

    def change_environment():
        """ change into 3.11"""
        run_model_script = "./visualization.py"

        # get the "string" path
        command = [python3_path, run_model_script]
        name = file_name.split('.')[0]
        argvs = [photo_number, file, name, workspace]
        # run python 3.11
        subprocess.call(command + argvs, stdout=sys.stdout, stderr=sys.stderr)

    # update canvas
    ax.clear()

    image_file = os.path.join(workspace, file_name)
    if not os.path.exists(image_file):
        change_environment()

    image = imread(image_file)
    ax.imshow(image, aspect='equal')
    ax.set_axis_off()

    canvas.draw()


class CityGrowthSimulator:
    def __init__(self, root):
        self.root = root

        # show in center
        self.root.geometry("500x600+300+100")
        self.root.title("urban growth system")

        # entry dynamic text
        self.workspace_var = tk.StringVar()
        self.area_var = tk.StringVar()

        # dynamic text
        self.text_var = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.text_var)

        self.original_window()
    def original_window(self):
        def create_frame1():
            # workspace
            self.workspace_frame = ttk.Frame(self.frame1)
            self.workspace_frame.pack(fill='x')
            self.workspace_label = ttk.Label(self.workspace_frame, text="workspace")
            self.workspace_label.pack(
                anchor='nw', pady=5, padx=10
            )
            self.workspace_entry = ttk.Entry(self.workspace_frame)
            self.workspace_entry.config(textvariable=self.workspace_var)
            self.workspace_entry.pack(
                side='left',expand=True,fill='x',pady=8, padx=10
            )
            self.workspace_button = tk.Button(
                self.workspace_frame, text="read",relief="flat"
            )
            self.workspace_button.config(command=self.load_workspace)
            self.workspace_button.pack(side="right", pady=8, padx=6)

            # study area
            self.area_frame = ttk.Frame(self.frame1)
            self.area_frame.pack(fill='x')
            self.area_label = ttk.Label(self.area_frame, text="study area")
            self.area_label.pack(
                anchor='nw', pady=5, padx=10
            )
            self.area_entry = ttk.Entry(self.area_frame)
            self.area_entry.config(textvariable=self.area_var)
            self.area_entry.pack(
                side='left',expand=True, fill='x', pady=8, padx=10
            )
            self.area_button = tk.Button(
                self.area_frame, text="read",relief="flat"
            )
            self.area_button.config(command=self.load_study_area)
            self.area_button.pack(side="right", pady=8, padx=6)

            # operation
            self.op_frame = ttk.Frame(self.frame1)
            self.op_frame.pack(fill='x')
            self.op_label = ttk.Label(self.op_frame, text="operation")
            self.op_label.pack(
                anchor='nw', pady=5, padx=10
            )

            buttons = [
                ("load files", self.loadfiles_window),
                ("data process", self.data_preprocess),
                ("run model", self.runmodel_window),
                ("statistic", self.stas_window)
            ]

            for i,(text,cmd) in enumerate(buttons): # pack and grid do not mixed use, if do, no tk window
                tk.Button(self.op_frame, text=text, relief="flat", command=cmd).pack(
                    side='left', padx=2, pady=8, expand=True, fill='x'
                )

            self.status_label = ttk.Label(self.frame1, text="work record")
            self.status_label.pack(
                anchor='nw', pady=5, padx=10
            )
            self.status_text = tk.Text(
                self.frame1, width=8, height=30, relief='flat', font=("Times New Roman", 11)
            ) # show record
            self.status_text.pack(fill='x')

        def create_frame2():
            self.fig = Figure(figsize=(10, 6), dpi=300)
            self.ax = self.fig.add_subplot(111)

            self.ax.set_axis_off()
            self.fig.patch.set_facecolor('white')
            self.ax.patch.set_facecolor('white')

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame2)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            self.ax.plot([], [])
            self.fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.01)
            self.canvas.draw()

        # function frame
        self.frame1 = ttk.Frame(self.root)
        self.frame1.pack(side="left", fill="both", expand=True)
        create_frame1()
        # record frame
        self.frame2 = ttk.Frame(self.root, borderwidth=3, relief="flat")
        self.frame2.pack(side="right", fill="both", expand=True)
        create_frame2()


        # more like gundongtiao
    def load_workspace(self):
        """
        select workspace: the place which save file
        :return: self.workspace
        """
        file_path = filedialog.askdirectory(title="select workspace")
        if file_path:
            self.workspace_entry.delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.workspace_entry.insert(0, file_path)  # show file path,
            self.workspace = file_path

            arcpy.env.workspace = self.workspace
            arcpy.env.overwriteOutput = True

            update_status(self.status_text,"workspace has been defined")

    def load_study_area(self):
        """
        select study area
        :return: self.study_area
        """
        file_path = filedialog.askopenfilename(
            title="select study area (Shapefile)",
            filetypes=[("Shapefile", "*.shp")]
        )
        if file_path:
            self.area_entry.delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.area_entry.insert(0, file_path)  # show file path

            projected_path = 'JYUTM.shp'
            self.out_coordinate_system = arcpy.SpatialReference(32649)
            arcpy.Project_management(file_path, projected_path, self.out_coordinate_system)
            self.study_area = projected_path
            arcpy.env.extent = projected_path
            arcpy.env.mask = projected_path


            update_status(self.status_text,"study area has been defined")
            plotting_thread = threading.Thread(
                target=update_canvas, args=('1', file_path, 'study_area.gif', self.workspace, self.ax, self.canvas)
            )
            plotting_thread.start()
            update_status(self.status_text, "study area has been drawing...")
    def loadfiles_window(self):
        self.file_window_instance = LoadFiles(self.root,self.status_text)
        self.file_window_instance.loadfiles_window()
    def data_preprocess(self):
        print(self.file_window_instance.export())
        self.data_process_instance = DataProcess(self.status_text, self.file_window_instance.export())
        self.data_process_instance.main()
    def runmodel_window(self):
        self.model_window_instance = RunModel(self.root, self.status_text, self.workspace)
        self.model_window_instance.runmodel_window()

    def stas_window(self):
        self.stas_window_instance = Statistic(self.root, self.status_text, self.workspace)
        begin_year, end_year, years= self.model_window_instance.export()
        self.stas_window_instance.change_to_plot(begin_year, end_year, years)
class LoadFiles:
    def __init__(self, root, text):
        self.file_window = tk.Toplevel(root)
        self.file_window.geometry("700x550+350+100")
        self.file_window.title("load related data")

        self.workspace = arcpy.env.workspace
        self.text = text

        self.before_land_var = tk.StringVar()
        self.after_land_var = tk.StringVar()
        self.cityway_var = tk.StringVar()
        self.motorway_var = tk.StringVar()
        self.dem_var = tk.StringVar()
        self.restaurant_var = tk.StringVar()
        self.hospital_var = tk.StringVar()
        self.drugstore_var = tk.StringVar()
        self.hotel_var = tk.StringVar()
        self.shop_var = tk.StringVar()
        self.school_var = tk.StringVar()
        self.park_var = tk.StringVar()
        self.coach_var = tk.StringVar()
        self.government_var = tk.StringVar()
        self.bank_var = tk.StringVar()
        self.plant_var = tk.StringVar()


        self.layer = []

        # show
        self.show_buttons = []
        self.out_coordinate_system = arcpy.SpatialReference(32649) # Albers_Conic_Equal_Area no WKID
                                                                    # so use UTM
    def loadfiles_window(self):
        def create_frame1():
            files = [
                ("land use(before)", self.before_land_var, self.before_landuse_process),
                ("land use(after)", self.after_land_var, self.after_landuse_process),
                ("dem", self.dem_var, self.dem_process),
                ("cityway", self.cityway_var, lambda: self.poi_loaded('cityway')),
                ("motorway", self.motorway_var, lambda: self.poi_loaded('motorway')),
                ("restaurant", self.restaurant_var, lambda: self.poi_loaded('restaurant')),
                ("hospital", self.hospital_var, lambda: self.poi_loaded('hospital')),
                ("drugstore", self.drugstore_var, lambda: self.poi_loaded('drugstore')),
                ("hotel", self.hotel_var, lambda: self.poi_loaded('hotel')),
                ("shop", self.shop_var, lambda: self.poi_loaded('shop')),
                ("school", self.school_var, lambda: self.poi_loaded('school')),
                ("park", self.park_var, lambda: self.poi_loaded('park')),
                ("coach", self.coach_var, lambda: self.poi_loaded('coach')),
                ("government", self.government_var, lambda: self.poi_loaded('government')),
                ("bank", self.bank_var, lambda: self.poi_loaded('bank')),
                ("plant", self.plant_var, lambda: self.poi_loaded('plant')),
            ]

            self.entries =[]
            self.show_buttons = []
            for text, var, command in files:
                frame = ttk.Frame(self.frame1)
                frame.pack(fill='x')

                label = ttk.Label(frame, text=text)
                label.pack(side='left', pady=3, padx=5)

                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side='left', expand=True, fill='x', pady=3, padx=5)
                self.entries.append(entry)

                show_button = tk.Button(frame, text='show', relief='flat', command=self.warning)
                show_button.pack(side='right', pady=3, padx=5)
                self.show_buttons.append(show_button)

                read_button = tk.Button(frame, text="Read", relief="flat", command=command)
                read_button.pack(side="right", pady=3, padx=5)

        def create_frame2():
            self.fig = Figure(figsize=(10, 6), dpi=300)
            self.ax = self.fig.add_subplot(111)

            self.ax.set_axis_off()
            self.fig.patch.set_facecolor('white')
            self.ax.patch.set_facecolor('white')

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame2)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

            self.ax.plot([], [])
            self.fig.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.01)
            self.canvas.draw()

        # function frame
        self.frame1 = ttk.Frame(self.file_window)
        self.frame1.pack(side="left", fill="both", expand=True)
        create_frame1()

        # canvas frame
        self.frame2 = ttk.Frame(self.file_window, borderwidth=3, relief="flat")
        self.frame2.pack(side="right", fill="both", expand=True)
        create_frame2()

    def warning(self):
        messagebox.showwarning(title='warning', message='You need to read file firstly!')

    def draw(self, photo_number, file, file_name):
        update_status(self.text, "drawing...")
        plotting_thread = threading.Thread(
            target=update_canvas,
            args=(photo_number, file, file_name, self.workspace, self.ax, self.canvas)
        )
        plotting_thread.start()

    def before_landuse_process(self):
        """
        1. land use file
        2. process field
        :update: self.landuse
        """
        # 1.
        file_path = filedialog.askopenfilename(
            title="select former land use (Raster)",
            filetypes=[("Raster", "*.tif")]
        )
        if file_path:
            self.entries[0].delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.entries[0].insert(0, file_path)  # show file path,

            projected_path = 'landusebeforeproj.tif'
            re_path = 'landusebeforeprojre.tif'
            self.reclassification(file_path, projected_path, re_path)

            self.show_buttons[0].config(
                command=lambda: self.draw('2', re_path, 'Former_landuse.gif')
            )

            self.layer.append(re_path) # original file

        update_status(self.text, "Former land use has been loaded.")
    def after_landuse_process(self):
        """
        1. land use file
        2. process field
        :update: self.landuse
        """
        # 1.
        file_path = filedialog.askopenfilename(
            title="select latter land use (Raster)",
            filetypes=[("Raster", "*.tif")]
        )
        if file_path:
            self.entries[1].delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.entries[1].insert(0, file_path)  # show file path,

            projected_path = 'landusefterproj.tif'
            re_path = 'landuseafterprojre.tif'
            self.reclassification(file_path, projected_path, re_path)

            self.show_buttons[1].config(
                command=lambda: self.draw('3', re_path, 'Latter_landuse.gif')
            )

            self.layer.append(re_path) # original file

        update_status(self.text, "Latter land use has been loaded.")
    def reclassification(self, file_path, projected_path, re_path):
        # project
        arcpy.ProjectRaster_management(file_path, projected_path, self.out_coordinate_system)
        input_raster = projected_path
        output_raster = re_path

        # turn to numpy
        raster = arcpy.Raster(input_raster)
        arr = arcpy.RasterToNumPyArray(raster)
        # Modifying raster values
        # original landuse value
        arr[arr == 3] = 2  # vagetation
        arr[arr == 4] = 2
        arr[arr == 7] = 5  # hardly use
        # Getting raster space reference information
        lower_left = arcpy.Point(raster.extent.XMin, raster.extent.YMin)
        cell_size = raster.meanCellWidth
        # turn back to raster
        new_raster = arcpy.NumPyArrayToRaster(arr, lower_left, cell_size, cell_size, value_to_nodata=raster.noDataValue)
        # save
        new_raster.save(output_raster)

    def dem_process(self):
        """
        1. translate geo coordinate to project coordinate: WGS_1984_UTM_Zone_49N
        2. extract dem and slope value to point
        :return: points have dem and slope
        """
        # 2.
        file_path = filedialog.askopenfilename(
            title="select dem (Raster)",
            filetypes=[("Raster", "*.tif")]
        )
        if file_path:
            self.entries[2].delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.entries[2].insert(0, file_path)  # show file path,

            clipped_dem = 'clipped_dem.tif'
            arcpy.gp.ExtractByMask_sa(file_path, arcpy.env.mask, clipped_dem)

            projected_dem = "dem.tif"
            arcpy.ProjectRaster_management(clipped_dem, projected_dem, self.out_coordinate_system)
            self.layer.append(projected_dem)

            slope = "slope.tif"
            arcpy.gp.Slope_sa(projected_dem, slope, "DEGREE", "1", "PLANAR", "METER")
            self.layer.append(slope)

            self.show_buttons[2].config(
                command=lambda: self.draw('5',projected_dem,'dem_and_slope.gif')
            )

            update_status(self.text, "Dem and Slope have been loaded.")
    def poi_loaded(self, poi_name):
        files = {
            'cityway':(
                3, "select cityway(Shapefile)", "cityway_project.shp", "eu_cityway.tif", 'Distance_to_cityway.gif',
                "the cityway has been loaded."
            ),
            'motorway':(
                4,"select motorway(Shapefile)", "motorway_project.shp", "eu_motorway.tif", 'Distance_to_motorway.gif',
                "the motorway has been loaded."
            ),
            'restaurant': (
                5, "select restaurant(Shapefile)", "restaurant_project.shp", "eu_restaurant.tif", 'Distance_to_restaurant.gif',
                "the restaurant has been loaded."
            ),
            'hospital': (
                6,"select hospital(Shapefile)", "hospital_project.shp", "eu_hospital.tif", 'Distance_to_hospital.gif',
            "the hospital has been loaded."
            ),
            'drugstore': (
                7,"select drugstore(Shapefile)", "drugstore_project.shp", "eu_drugstore.tif", 'Distance_to_drugstore.gif',
                "the drugstore has been loaded."
            ),
            'hotel': (
                8,"select hotel(Shapefile)", "hotel_project.shp", "eu_hotel.tif", 'Distance_to_hotel.gif',
                "the hotel has been loaded."
            ),
            'shop': (
                9,"select shop(Shapefile)", "shop_project.shp", "eu_shop.tif", 'Distance_to_shop.gif',
                "the shop has been loaded."
            ),
            'school': (
                10,"select school(Shapefile)", "school_project.shp", "eu_school.tif", 'Distance_to_school.gif',
                "the school has been loaded."
            ),
            'park': (
                11, "select park(Shapefile)", "park_project.shp", "eu_park.tif", 'Distance_to_park.gif',
                "the park has been loaded."
            ),
            'coach': (
                12, "select coach(Shapefile)", "coach_project.shp", "eu_coach.tif", 'Distance_to_coach.gif',
                "the coach has been loaded."
            ),
            'government': (
                13, "select government(Shapefile)", "government_project.shp", "eu_government.tif", 'Distance_to_government.gif',
                "the government has been loaded."
            ),
            'bank': (
                14, "select bank(Shapefile)", "bank_project.shp", "eu_bank.tif", 'Distance_to_bank.gif',
                "the bank has been loaded."
            ),
            'plant': (
                15, "select plant(Shapefile)", "plant_project.shp", "eu_plant.tif", 'Distance_to_plant.gif',
                "the plant has been loaded."
            )
        }
        number, title, project_file, eu_file, draw_name, message = files[poi_name]
        self.poi_process(number, title, project_file, eu_file, draw_name, message)
    def poi_process(self, number, title, project_file, eu_file, draw_name, message):
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Shapefile", "*.shp")]
        )
        if file_path:
            self.entries[number].delete(0, tk.END)  # clean entry text
            # "index" indicates that the insertion position is the beginning of the input box
            self.entries[number].insert(0, file_path)  # show file path,

            # eu
            arcpy.Project_management(file_path, project_file, self.out_coordinate_system)
            arcpy.gp.EucDistance_sa(project_file, eu_file, "", "300", "", "PLANAR", "", "")
            self.layer.append(eu_file)

            # system
            self.show_buttons[number].config(
                command=lambda: self.draw('6', eu_file, draw_name)
            )
            update_status(self.text, message)

    def export(self):
        return self.layer
class DataProcess:
    def __init__(self, text, layer):
        self.text = text
        self.workspace = arcpy.env.workspace
        self.layer = layer
    def generate_point(self):
        extent = arcpy.env.extent
        # create fishmap
        fishmap = 'fishmap.shp'
        xmin = extent.XMin
        xmax = extent.XMax
        ymin = extent.YMin
        ymax = extent.YMax
        originCoordinate = "{} {}".format(xmin, ymin)
        yAxisCoordinate = "{} {}".format(xmin, ymin + 10)
        oppositeCorner = '{} {}'.format(xmax, ymax)
        arcpy.CreateFishnet_management(
            out_feature_class=fishmap,
            origin_coord=originCoordinate,
            y_axis_coord=yAxisCoordinate,
            cell_width="300",
            cell_height="300",
            number_rows="",
            number_columns="",
            corner_coord=oppositeCorner,
            labels="LABELS",
            template="",
            geometry_type="POLYGON"
        )

        # clip to study area
        inFeature = 'fishmap_label.shp'
        clipFeature = 'JYUTM.shp'
        outFetureClass = 'fishmap_labelclip.shp'
        arcpy.Clip_analysis(inFeature, clipFeature, outFetureClass)

        return outFetureClass
    def extract_data(self, point):
        fields = [
            'ori', 'target',
            'dem', 'slope', 'cityway', 'motorway', 'restaurant',
            'hospital', 'drugstore', 'hotel', 'shop', 'school',
            'park', 'coach', 'government', 'bank', 'plant'
        ]

        # extract data to point
        values = []
        i = 0
        for lay in self.layer:
            value = [lay, fields[i]]
            values.append(value)
            i = i + 1
        ExtractMultiValuesToPoints(point, values, "NONE")


        # clean miss value
        input_fc = point
        with arcpy.da.UpdateCursor(input_fc, fields) as cursor:
            for row in cursor:
                if any(val is None or val == -9999 for val in row):
                    cursor.deleteRow()  # DELETE

        # create dataset
        all_fields = ["FID"] + fields
        data = []
        with arcpy.da.SearchCursor(point, all_fields) as cursor:
            for row in cursor:
                data.append(row)
        df = pd.DataFrame(data, columns=all_fields)
        output_data = os.path.join(arcpy.env.workspace, "data.csv")
        df.to_csv(output_data, index=False)

        return input_fc, "data.csv"
    def generate_adj(self, point):
        # buffer
        inFeature = point
        outFeatureClass = 'buffer.shp'
        arcpy.Buffer_analysis(
            inFeature, outFeatureClass, "450 Meters")

        # interact to find neighbor
        outFeatureClass = 'neighbors.shp'
        arcpy.Intersect_analysis(
            [[point, 2], ['buffer.shp', 1]],
            outFeatureClass, "ALL", "", ""
        )

        return 'neighbors.shp'
    def extract_adjacency(self, neighbors, point_afterclean):
        """sparse saved"""
        T1 = time.time()

        output_file = os.path.join(arcpy.env.workspace,
                                   "adjacency_matrix.npz")  # sparse saved by .npz, the space of which using is much smaller for 80000 element

        row_indices = []
        col_indices = []
        data_values = []

        n = int(arcpy.GetCount_management(point_afterclean).getOutput(0))
        with arcpy.da.SearchCursor(neighbors, ['FID_fishma', 'ORIG_FID']) as cursor:  # two FID begin with 0
            for row in cursor:
                i, j = row[0], row[1]
                row_indices.append(i)  # indices
                col_indices.append(j)  # indptr
                data_values.append(1)  # value

        adjacency_matrix = sp.csr_matrix((data_values, (row_indices, col_indices)), shape=(n, n),
                                         dtype=np.int8)  # row compound
        np.savez(output_file,
                 data=adjacency_matrix.data,
                 indices=adjacency_matrix.indices,
                 indptr=adjacency_matrix.indptr,
                 shape=adjacency_matrix.shape)

        T2 = time.time()
        print('extract adj matrix using time :%s ms' % ((T2 - T1) * 1000))  # 2013 30s; 2018 60s

    def main(self):
        # create fishpoint
        point = self.generate_point()
        update_status(self.text, "fishmap has been saved at: \"fishmap.shp\"")

        # extract poi ,clean miss value and create dataset
        point_with_data, dataset = self.extract_data(point)
        update_status(self.text, "data has been extracted from raster layers, cleaned and saved at: \"data.csv\"")

        # define adj using buffer tools based on points
        neighbors = self.generate_adj(point_with_data)
        update_status(self.text, "finding neighborhoods with the Intersection tool and saved at: \"neighbors.shp\"")

        # extract adj
        self.extract_adjacency(neighbors, point_with_data)
        update_status(self.text, "adjacency matrix has been saved at: \"adjacency_matrix.npz\"")
class RunModel:
    def __init__(self, root, text, workspace):
        self.model_window = tk.Toplevel(root)
        self.model_window.geometry("250x130+350+100")
        self.model_window.title("run model")
        self.text = text
        self.workspace = workspace

        self.begin_var = tk.StringVar()
        self.after_var = tk.StringVar()
        self.iteration_var = tk.StringVar()
    def runmodel_window(self):
        # begin frame
        begin_frame = ttk.Frame(self.model_window)
        begin_frame.pack(fill='x')
        begin_label = ttk.Label(begin_frame, text='begin year')
        begin_label.pack(side='left', pady=3, padx=5)
        begin_entry = ttk.Entry(begin_frame, textvariable=self.begin_var)
        begin_entry.pack(side='right', expand=True, fill='x', pady=3, padx=5)

        # after frame
        after_frame = ttk.Frame(self.model_window)
        after_frame.pack(fill='x')
        after_label = ttk.Label(after_frame, text='after year')
        after_label.pack(side='left', pady=3, padx=5)
        after_entry = ttk.Entry(after_frame, textvariable=self.after_var)
        after_entry.pack(side='right', expand=True, fill='x', pady=3, padx=5)

        # iteration frame
        iteration_frame = ttk.Frame(self.model_window)
        iteration_frame.pack(fill='x')
        iteration_label = ttk.Label(iteration_frame, text='iteration times')
        iteration_label.pack(side='left', fill='x')
        iteration_entry = ttk.Entry(iteration_frame, textvariable=self.iteration_var)
        iteration_entry.pack(side='right', expand=True, fill='x', pady=3, padx=5)

        # function frame
        function_frame = ttk.Frame(self.model_window)
        function_frame.pack(fill='x')
        run_button = tk.Button(
            function_frame, text='run', command=self.run_model, relief='groove'
        )
        run_button.pack(fill='x', pady=3, padx=5)
    def run_model(self):
        def change_environment(before, after, iter, workspace):
            run_model_script = "stimulation.py"

            command = [python3_path, run_model_script]
            argvs = [before, after, iter, workspace]
            # run python 3.11
            subprocess.call(command + argvs, stdout=sys.stdout, stderr=sys.stderr)


        begin_year = self.begin_var.get()
        after_year = self.after_var.get()
        iteration = self.iteration_var.get()

        change_environment(begin_year, after_year, iteration, self.workspace)

        update_status(self.text, "Prediction is complete!")
        self.model_window.destroy()
    def export(self):
        return self.begin_var.get(), self.after_var.get(), self.iteration_var.get()
class Statistic:
    def __init__(self, root, text, workspace):
        self.plot_window = tk.Toplevel(root)
        self.plot_window.title("Presentation of simulation results")
        self.plot_window.geometry("1000x850")
        self.workspace = workspace
        self.text = text

        self.chord_canvas = None
        self.radial_canvas = None
        self.gif_frames_cache = {}
        self.all_gif_frames = {}

    def change_to_plot(self, begin_year, end_year, years):
        update_status(self.text, 'Drawing changes and statistics...')

        run_model_script = "statistic.py"

        command = [python3_path, run_model_script]
        argvs = [self.workspace, begin_year, end_year, years]
        subprocess.call(command + argvs, stdout=sys.stdout, stderr=sys.stderr)

        self.stas_window()
    def stas_window(self):
        # ================== map(auctual and pred) =================
        top_frame = ttk.LabelFrame(self.plot_window, text="stimulation", padding=(10, 10))
        top_frame.pack(fill="both", expand=True, ipady=10)


        map_titles = ["Former and Prediction", "Latter"] #'Latter_landuse.gif'
        gif_files = [os.path.join(self.workspace, r'raster_output\output_images\predicted_animation.gif'),
                     os.path.join(self.workspace, r'raster_output\output_images\Actual_end_year.gif')]
        for i, title in enumerate(map_titles):
            frame = ttk.Frame(top_frame)
            frame.pack(side='left', fill="both", expand=True)

            canvas = tk.Canvas(frame, bg="lightgray")
            canvas.pack(fill="both", expand=True)

            self.play_gif(canvas, gif_files[i])

        # =====================chart and table========================
        bottom_frame = ttk.LabelFrame(self.plot_window, text="Analyzing Charts and Accuracy Records", padding=(10, 10))
        bottom_frame.pack(fill="both", expand=True)

        # ======================statistic chart=============================
        chart_frame = ttk.Frame(bottom_frame)
        chart_frame.pack(side="left", fill="both", expand=True, padx=1, pady=1) # padx, pady inner margin

        # chord
        chord_frame = ttk.Frame(chart_frame)
        chord_frame.pack(side='left', fill="both", expand=True, padx=2) # Reduced left-right spacing
        self.chord_canvas = tk.Canvas(chord_frame, bg="#F0F0F0")
        self.chord_canvas.pack(fill="both", expand=True)

        self.add_chord()

        # bar
        radial_frame = ttk.Frame(chart_frame)
        radial_frame.pack(side='left', fill="both", expand=True, padx=1)
        self.radial_canvas = tk.Canvas(radial_frame, bg="#F0F0F0")
        self.radial_canvas.pack(fill="both", expand=True)

        self.add_radial()
        # ==========================table============================
        table_frame = ttk.Frame(bottom_frame)
        table_frame.pack(side="right", fill="both", expand=False, padx=10)
        table_frame.config(width=300) # Fixed form width

        table_label = ttk.Label(table_frame, text="history", font=("Arial", 16))
        table_label.pack()

        # create table
        acc_records = []
        file = os.path.join(self.workspace,'acc.txt')
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0].isdigit():
                    iteration, oa, kappa = parts
                    acc_records.append((iteration, oa, kappa))

        columns = ("year", "OA", "Kappa")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=60, anchor="center")
        tree.pack(fill="both", expand=True)

        for rec in acc_records: # record
            tree.insert("", tk.END, values=rec)

    def play_gif(self, canvas, gif_path):
        def resize_gif_frames(gif_path, new_width, new_height):
            gif = Image.open(gif_path)
            resized_frames = []
            for frame in ImageSequence.Iterator(gif):
                frame = frame.copy().convert("RGBA")
                frame = frame.resize((new_width, new_height), Image.LANCZOS)
                resized_frames.append(ImageTk.PhotoImage(frame))
            return resized_frames

        try:
            # Avoid loading the same GIF over and over again
            if gif_path in self.gif_frames_cache:
                frames = self.gif_frames_cache[gif_path]
            else:
                gif = Image.open(gif_path)
                frames = resize_gif_frames(gif_path, 500, 400)
                self.gif_frames_cache[gif_path] = frames  # Cache all frames

            def update(index):
                try:
                    frame = frames[index]
                    canvas.image = frame  # save current frame
                    canvas.delete("gif")  # clean old frame
                    canvas.create_image(0, 0, anchor=tk.NW, image=frame, tags="gif")
                    next_index = (index + 1) % len(frames)
                    canvas.after(1000, update, next_index)  # 1000ms
                except Exception as e:
                    print("Gif update failed: {}".format(e))

            update(0)  # begin

        except Exception as e:
            print("Gif load failed: {}".format(e))
            # Show Error Placeholder
            error_img = Image.new('RGBA', (450, 250), (255, 0, 0, 128))
            ImageDraw.Draw(error_img).text((50, 100), "failed to load\n{}".format(gif_path), fill="white")
            error_photo = ImageTk.PhotoImage(error_img)
            canvas.create_image(0, 0, anchor=tk.NW, image=error_photo)
            canvas.image = error_photo
    def add_chord(self):
        file = os.path.join(self.workspace, 'chord_diagram.gif')
        img = Image.open(file)
        img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(img)
        self.chord_canvas.image = photo  # save current
        self.chord_canvas.create_image(0, 0, anchor='nw', image=self.chord_canvas.image)
    def add_radial(self):
        file = os.path.join(self.workspace, 'grouped_radial_log.gif')
        img = Image.open(file)
        img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(img)
        self.radial_canvas.create_image(0, 0, anchor='nw', image=photo)
        self.radial_canvas.image = photo  # save current





