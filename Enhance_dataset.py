import matplotlib.pyplot as plt
import numpy as np
import itertools

def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, train_pixels=None):
    
    if len(img.shape) == 2:
        rows, cols = img.shape
        bands = 1
        img = img[:,:,np.newaxis]
    else:
        rows, cols, bands = img.shape

    if train_pixels is None:
        h = np.arange(0, rows)
        w = np.arange(0, cols)
        train_pixels = np.asarray(list(itertools.product(h, w))).transpose()

    min_value, max_value = [], []
    for band in range(bands):
        hist, bins = np.histogram(img[train_pixels[0], train_pixels[1], band].ravel(), bins=bins) # select training pixels
        cum_hist = np.cumsum(hist) / hist.sum()

        # # See outliers cut values
        # plt.plot(bins[1:], hist)
        # plt.plot(bins[1:], cum_hist)
        # plt.stem(bins[len(cum_hist[cum_hist<bth])], 0.5)
        # plt.stem(bins[len(cum_hist[cum_hist<uth])], 0.5)
        # plt.title("band %d"%(band))
        # plt.show()

        min_value.append(bins[len(cum_hist[cum_hist<bth])])
        max_value.append(bins[len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]

def median_filter(img, clips, mask):
    kernel_size = 10

    outliers = ((img < clips[0]) + (img > clips[1]))
    if len(img.shape) == 3:
        outliers *= np.expand_dims(mask, axis=2)
    else: outliers *= mask
    # plt.imshow(outliers[:,:,0], cmap='gray')
    # plt.imshow(outliers[:,:,1], cmap='gray')
    # plt.show()
    out_idx = np.asarray(np.where(outliers))

    img_ = img.copy()
    for i in range(out_idx.shape[1]):
        x = out_idx[0][i]
        y = out_idx[1][i]
        a = x - kernel_size//2 if x - kernel_size//2 >=0 else 0
        c = y - kernel_size//2 if y - kernel_size//2 >=0 else 0
        b = x + kernel_size//2 if x + kernel_size//2 <= img.shape[0] else img.shape[0]
        d = y + kernel_size//2 if y + kernel_size//2 <= img.shape[1] else img.shape[1]
        win = img[a:b, c:d][mask[a:b, c:d]==True]
        img_[x, y] = np.median(win, axis=0)
        # img_[x, y] = np.mean(win, axis=0)
    
    return img_

def Enhance_image(img, land_nan_mask, output_folder='', clips=None):

    # fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    # hist, bins  = np.histogram(img[~land_nan_mask], bins=10000)
    # axs[0].plot(bins[1:], hist/(hist.sum())); axs[0].set_title("hist")

    if clips is None:
        clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                                train_pixels=np.asarray(np.where(~land_nan_mask)))
    img = median_filter(img, clips, ~land_nan_mask)

    # hist, bins  = np.histogram(img[~land_nan_mask], bins=10000)
    # axs[1].plot(bins[1:], hist/(hist.sum())); axs[1].set_title("(no outliers)-hist")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(output_folder + 'histogram.png')
    # plt.close()

    min_ = img[~land_nan_mask].min(0)
    max_ = img[~land_nan_mask].max(0)
    img = np.uint8(255*((img - min_) / (max_ - min_)))
    
    img[land_nan_mask] = 255

    return img