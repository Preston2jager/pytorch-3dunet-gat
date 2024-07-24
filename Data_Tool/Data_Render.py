from vispy import app, visuals, scene
import numpy as np
import argparse 

#=================================
#Loading data file
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='file to load')
args = parser.parse_args()
array = np.load(args.file)

#=================================
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
canvas = scene.SceneCanvas(keys="interactive", show=True)
view = canvas.central_widget.add_view()
view.camera = "turntable"
view.camera.fov = 45
view.camera.distance = 500

#=================================
x1, y1, z1 = np.where((array == [0, 1]).all(axis=-1))
points_obj = np.vstack((x1, y1, z1)).T

p1 = Scatter3D(parent=view.scene)
p1.set_gl_state("translucent", blend=True, depth_test=True)
p1.set_data(
points_obj, face_color="yellow", symbol="o", size=5, edge_width=0.1, edge_color="yellow"
)
app.run()