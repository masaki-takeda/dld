import numpy as np
from scipy import io
import os

def save_eeg_ch_data():
    """ EEGの各チャンネルの名前とPositinをnumpy .npzデータとしてdataディレクトリに保存する """
    
    file_path = "/Users/miyoshi/Desktop/kut_data/Data/EEG/TM_191008_01/TM_191008_01_01_Segmentation.mat"
    matdata = io.loadmat(file_path)
    chanlocs = matdata['ALLEEG']['chanlocs'][0][0][0]
    channel_raw_names = [chanloc[0][0] for chanloc in chanlocs]
    channel_xs_raw = [chanloc[6][0] for chanloc in chanlocs]
    channel_ys_raw = [chanloc[7][0] for chanloc in chanlocs]
    channel_zs_raw = [chanloc[8][0] for chanloc in chanlocs]

    # ECGを除いた各チャンネルの位置を得る
    ch_size = len(channel_raw_names)
    ch_positions = np.empty([ch_size-1, 3]) # (63, 3)
    ch_names = []
    ch_indices = []

    ch_index = 0 # ECGを抜いたindex (0~62)
    ECG_INDEX = 31

    for ch in range(ch_size):
        if ch != ECG_INDEX:
            x = channel_xs_raw[ch]
            y = channel_ys_raw[ch]
            z = channel_zs_raw[ch]
            ch_positions[ch_index] = [x,y,z]
            ch_name = channel_raw_names[ch]
            ch_names.append(ch_name)
            ch_indices.append(ch)
            ch_index += 1

    # ECGを抜いたx,y
    xs =  ch_positions[:,0]
    ys =  ch_positions[:,1]
    zs =  ch_positions[:,2]

    file_name = "eeg_ch_data"

    # .npzを省いたパス
    file_path = os.path.join("experiment_data", file_name)

    # 圧縮して保存
    np.savez(file_path,
             names=ch_names,
             positions=ch_positions)

    print("EEG channel data saved: {}".format(file_path))


def load_eeg_ch_data():
    """ EEGの各チャンネルの名前とPositinをdataディレクトリに置いてある.npzデータからロードする """
    
    file_name = "eeg_ch_data.npz"
    file_path = os.path.join("experiment_data", file_name)
    
    data = np.load(file_path)

    # strに変換しておく
    names = [str(name) for name in data["names"]]
    positions = data["positions"]
    
    return names, positions


from PIL import Image
from PIL import ImageDraw

def draw_circle(draw, x, y, radius, color):
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), 
                  outline=color)
    
def draw_text(draw, x, y, text, color=(0,0,0)):
    w, h = draw.textsize(text)
    draw.text((x-w//2, y-h//2), text, fill=color)


def normalize_signed_weights(weights):
    """ 正の最大値が1.0 または　負の最小値が-1.0になる様にNormalizeする """
    #weights = np.array(weights)
    
    max_plus = max(np.max(weights), 0.0)
    min_minus = min(np.min(weights), 0.0)

    max_range = max(max_plus, -min_minus)

    if max_range != 0.0:
        normalized_weights = weights / max_range
    else:
        normalized_weights = weights
        
    return normalized_weights


def draw_eeg_ch_weights(ch_names, ch_positions, weights, file_base):
    normalized_weights = normalize_signed_weights(weights)
    
    width = 512

    image_xy = Image.new('RGB', (width, width), (255, 255, 255))
    draw_xy = ImageDraw.Draw(image_xy)

    image_xz = Image.new('RGB', (width, width), (255, 255, 255))
    draw_xz = ImageDraw.Draw(image_xz)
    
    # x,y,zは(-1~1, -1~1, -0.4~1)の範囲なので、外側に余裕少し持たせた範囲
    max_range = 1.2
    
    for name, position, weight in zip(ch_names, ch_positions, normalized_weights):
        if weight > 0.0:
            color = (255,0,0)
        else:
            color = (0,0,255)
        radius = int(32 * abs(weight))
        
        ix = int(position[0] / max_range * width/2 + width/2)
        iy = int(position[1] / max_range * width/2 + width/2)
        iz = int(position[2] / max_range * width/2 + width/2)

        x0 = width - iy
        y0 = width - ix
        draw_text(draw_xy, x0, y0, str(name))
        draw_circle(draw_xy, x0, y0, radius, color)

        x1 = width - ix
        y1 = width - iz

        if position[1] >= -0.01:
            draw_text(draw_xz, x1, y1, str(name))
        draw_circle(draw_xz, x1, y1, radius, color)
        
    image_xy.save(file_base + "_xy.png")
    image_xz.save(file_base + "_xz.png")
    

if __name__ == '__main__':
    save_eeg_ch_data()
    
    #names, positions = load_eeg_ch_data()
    #print(len(names))
    #print(len(positions))
