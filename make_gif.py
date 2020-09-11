from PIL import Image

def make_gif(name, Nsteps, folder, skip=1):
    # Create the frames
    frames = []
    for i in range(1, Nsteps, skip):
        new_frame = Image.open('%s/normalized_%03d.png' % (folder, i))
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(name, format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=10)