from vispy import app, visuals, scene
from vispy.color import Color 
import numpy as np
import argparse 

#=================================
#Loading data file
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='file to load')
args = parser.parse_args()
array = np.load(args.file)

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

last_dim_len = array.shape[-1]
for i in range(last_dim_len):
    if i != 0:
        target_vector = np.zeros(last_dim_len)
        target_vector[i] = 1
        mask = (array == target_vector).all(axis=-1)
        x1, y1, z1 = np.where(mask)
        if x1.size > 0:
            points_obj = np.vstack((x1, y1, z1)).T
            color_index = i % len(colors)

            face_color = Color(colors[color_index]).rgba
            edge_color = Color(colors[color_index]).rgba

            p1 = Scatter3D(parent=view.scene)
            p1.set_gl_state("translucent", blend=True, depth_test=True)
            p1.set_data(
                points_obj, 
                face_color=face_color, 
                symbol="o", 
                size=5, 
                edge_width=0.1, 
                edge_color=edge_color
        )
app.run()