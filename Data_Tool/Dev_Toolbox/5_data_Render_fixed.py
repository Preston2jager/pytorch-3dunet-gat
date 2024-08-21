from vispy import app, visuals, scene
from vispy.color import Color 
import numpy as np
import h5py 

#=================================
#Loading data file
#file_path = 'C:/Users/prest/Downloads/predictions/data_bdc8ddb1_predictions_1.h5'
file_path = 'C:/Users/prest/Downloads/predictions/data_bdc8ddb1_predictions_3.h5'

with h5py.File(file_path, 'r') as h5_file:
    array = h5_file['predictions'][:]

#=================================
#Initialise render
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
canvas = scene.SceneCanvas(keys="interactive", show=True)
view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov = 45
view.camera.distance = 500

#=================================
#Points processing

colors = [
    "#FF5733",  "#33FF57",  "#5733FF",  "#FFFF33",  "#33FFFF",
    "#FF33FF",  "#FF8C00",  "#8C00FF",  "#008CFF",  "#FFD700",
    "#00FFD7",  "#D700FF",  "#00FF00",  "#0000FF",  "#FF0000"
]


for i in range(1, 8):  # 从 1 到 7，因为 0 可能表示背景
    mask = (array == i)
    x1, y1, z1 = np.where(mask)
    
    if x1.size > 0:
        points_obj = np.vstack((x1, y1, z1)).T
        color_index = i % len(colors)
        
        face_color = Color(colors[color_index]).rgba
        edge_color = Color(colors[color_index]).rgba
        
        p1 = scene.visuals.Markers(parent=view.scene)
        p1.set_data(
            points_obj,
            face_color=face_color,
            symbol="o",
            size=5,
            edge_width=0.1,
            edge_color=edge_color
        )

app.run()