from vedo.utils import trimesh2vedo
from vedo import Plotter

def create_image(mesh, current_epoch, showx, showy, pred_value, offscreen=True, image_fname=None):

    vp = Plotter(shape=(2, 3), axes=0, offscreen=offscreen, size=[1920, 1080])
    bottom_crop = 0.00

    vp1 = trimesh2vedo(mesh)
    vp1.cmap(on='points', cname='jet', input_array=showx, vmax=0.3, vmin=-0.3)
    vp1.crop(top=0, bottom=bottom_crop)
    vp1.addScalarBar(title='Original\nscan(mm)', pos=(0.65, 0.05))

    vp2 = trimesh2vedo(mesh)
    vp2.cmap(on='points', cname='jet', input_array=pred_value, vmax=0.3, vmin=-0.3)
    vp2.crop(top=0, bottom=bottom_crop)
    vp2.addScalarBar(title='NN\nprediction(mm)', pos=(0.65, 0.05))

    vp3 = trimesh2vedo(mesh)
    vp3.cmap(on='points', cname='jet', input_array=showy, vmax=0.3, vmin=-0.3)
    vp3.crop(top=0, bottom=bottom_crop)
    vp3.addScalarBar(title='average\ndeviation(mm)', pos=(0.65, 0.05))

    vp4 = trimesh2vedo(mesh)
    diff = pred_value - showx
    vp4.cmap(on='points', cname='seismic', input_array=diff, vmax=0.15, vmin=-0.15)
    vp4.crop(top=0, bottom=bottom_crop)
    vp4.addScalarBar(title='pred - input\ndeviation(mm)', pos=(0.65, 0.05))

    vp5 = trimesh2vedo(mesh)
    diff = showy - showx
    vp5.cmap(on='point', cname='seismic', input_array=diff, vmax=0.15, vmin=-0.15)
    vp5.crop(top=0, bottom=bottom_crop)
    vp5.addScalarBar(title='truth - input\ndeviation(mm)', pos=(0.65, 0.05))

    vp6 = trimesh2vedo(mesh)
    diff = pred_value - showy
    vp6.cmap(on='point', cname='seismic', input_array=diff, vmax=0.15, vmin=-0.15)
    vp6.crop(top=0, bottom=bottom_crop)
    vp6.addScalarBar(title='pred - truth\ndeviation(mm)', pos=(0.65, 0.05))

    cam = dict(pos=(-127.2, -84.22, 103.2),
               focalPoint=(-3.760, -4.435, 32.66),
               viewup=(0.2635, 0.3777, 0.8877),
               distance=163.1,
               clippingRange=(68.64, 291.0))

    vp.show(vp1, at=0)
    vp.show(vp2, at=1)
    vp.show(vp3, at=2)
    vp.show(vp6, at=3)
    vp.show(vp4, at=4)
    vp.show(vp5, interactive=True, at=5, camera=cam)
    if image_fname is None:
        image_fname = 'images/' + str(current_epoch) + 'comparison.png'
    vp.screenshot(image_fname)
    vp.close()
    return image_fname

def plot3d(data, cmap=None):
    from matplotlib import  pyplot as plt
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    sc = ax.scatter3D(data[:,0], data[:,1], data[:,2], c=cmap)
    plt.title("simple 3D scatter plot")
    plt.colorbar(sc)

    # show plot
    plt.show()

