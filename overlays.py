import numpy as np
import cv2
from matplotlib import pyplot as plt


def initialize_hist_figure():
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([0,3])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    def_x_line = np.arange(0, 256, 1)
    r_plot = ax.plot(def_x_line, def_x_line, 'r', animated=True)[0]
    g_plot = ax.plot(def_x_line, def_x_line, 'g', animated=True)[0]
    b_plot = ax.plot(def_x_line, def_x_line, 'b', animated=True)[0]
    
    return fig, ax, background, r_plot, g_plot, b_plot


def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    fig.canvas.restore_region(background)        
    r_plot.set_ydata(r_bars)        
    g_plot.set_ydata(g_bars)        
    b_plot.set_ydata(b_bars)

    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    fig.canvas.blit(ax.bbox)


def plot_overlay_to_image(np_img, plt_figure):
    rgba_buf = plt_figure.canvas.buffer_rgba()
    (w, h) = plt_figure.canvas.get_width_height()
    imga = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h,w,4)[:,:,:3]
    
    plt_indices = np.argwhere(imga < 255)

    height_indices = plt_indices[:,0]
    width_indices = plt_indices[:,1]
    
    np_img[height_indices, width_indices] = imga[height_indices, width_indices]

    return np_img


def plot_strings_to_image(np_img, list_of_string, text_color=(255,0,0), right_space=400, top_space=50):
    y_start = top_space
    min_size = right_space
    line_height = 20
    (h, w, c) = np_img.shape
    if w < min_size:
        raise Exception('Image too small in width to print additional text.')
        
    if h < top_space + line_height:
        raise Exception('Image too small in height to print additional text.')
    
    y_pos = y_start
    x_pos = w - min_size

    for text in list_of_string:
        if y_pos >= h:
            break
        np_img = cv2.putText(cv2.UMat(np_img), text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        y_pos += line_height

    if type(np_img) is cv2.UMat:
        np_img = np_img.get()

    return np_img
