# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 21-03-2025
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py
import lzma
import argparse
from pathlib import Path
from sklearn.metrics import root_mean_squared_error as rmse

def main():
    # Parse arguments:
    parser = argparse.ArgumentParser(
        description="distributions"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        help="input bands files, output from some calculations",
    )
    parser.add_argument(
        "--title",
        default="xxx",
        help="title for the graph",
    )

    parser.add_argument(
        "--fmin",
        type=float,
        help="min frequency",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        help="max frequency",
    )

    parser.add_argument("--dft", help="input dft bands file for comparison", default=None)
    parser.add_argument("--ml_labels", nargs="+", help="labels for ml bands", default=None)
    parser.add_argument("--dft_label", help="label for dft bands", default="DFT")
    parser.add_argument("--save", help="File to save the plot", default=None)
    args = parser.parse_args()

    title = args.title
    save_file = args.save
    bands = args.bands
    fmin = args.fmin
    fmax = args.fmax
    dft_file = args.dft
    ml_labels = args.ml_labels
    dft_label_text = args.dft_label

    assert bands is not None and len(bands) > 0
    if ml_labels is not None:
        assert len(ml_labels) == len(bands), "Number of labels must match number of band files"

    data_list = []
    nqpoint = None
    labels = None
    sp = None

    for band_file in bands:
        p = Path(band_file)
        assert p.exists(), f"File {band_file} does not exist"
        
        ext = p.suffix
        print(f"{ext}")
        data = None
        if ext == '.xz':
            with lzma.open(p, 'r') as file:
                dc = file.read()
                data = yaml.safe_load(dc)
        elif ext == '.hdf5':
            data  = h5py.File(p, 'r')
            print(f"{list(data.keys())}")
        else:
            with open(p, 'r') as file:
                data = yaml.safe_load(file)

        if ext==".hdf5":
            if nqpoint is None:
                nqpoint = data["nqpoint"][:][0]
                labels = [ [ y.decode('utf-8') for y in list(x)] for x in data['label'][:] ]
                sp = data['segment_nqpoint'][:][0]
            num_modes = data["natom"][()]*3
            f = data['frequency'][:]
            frequencies = f.reshape(-1,f.shape[-1])
        else:
            if nqpoint is None:
                nqpoint = data["nqpoint"]
                labels = data['labels']
                sp = data['segment_nqpoint'][0]
            num_modes = data["natom"]*3
            frequencies = np.array([[band["frequency"] for band in phonon["band"]] for phonon in data["phonon"]])
            
        data_list.append({'frequencies': frequencies, 'num_modes': num_modes})

    dft_data = None
    if dft_file:
        p = Path(dft_file)
        assert p.exists(), f"DFT file {dft_file} does not exist"
        ext = p.suffix
        if ext == '.xz':
            with lzma.open(p, 'r') as file:
                dc = file.read()
                data = yaml.safe_load(dc)
        elif ext == '.hdf5':
            data = h5py.File(p, 'r')
        else:
            with open(p, 'r') as file:
                data = yaml.safe_load(file)
        
        if ext == ".hdf5":
            f = data['frequency'][:]
            dft_frequencies = f.reshape(-1, f.shape[-1])
            num_modes = data["natom"][()]*3
        else:
            dft_frequencies = np.array([[band["frequency"] for band in phonon["band"]] for phonon in data["phonon"]])
            num_modes = data["natom"]*3
        dft_data = {'frequencies': dft_frequencies, 'num_modes': num_modes}

    k_points = np.arange(nqpoint)

    npa= -1
    seg_labels = {}
    seg_tick = {}
    for i,seg in enumerate(labels):
        if i > 0 and seg[0] == labels[i-1][1]:
            seg_labels[npa] += [seg[1]]
        else:
            npa += 1
            seg_labels[npa] = seg

    for k in seg_labels:
        if k > 0:
            seg_tick[k] = [ seg_tick[k-1][-1]+i*sp for i in range(len(seg_labels[k]))]
        else:
            seg_tick[k] = [ i*sp for i in range(len(seg_labels[k]))]

    npa += 1
    fs=8
    fsize=40
    # Add constant 4 inches to height for title/legend, and 3 inches to width for massive Y-axis labels
    fig, axs = plt.subplots(nrows=1, ncols=npa, figsize=(npa*fs + 3, fs + 4), squeeze=False, subplot_kw=dict(box_aspect=1))

    colors = plt.cm.tab10.colors

    for i in range(npa):
        for idx, d in enumerate(data_list):
            frequencies = d['frequencies']
            num_modes = d['num_modes']
            c = colors[idx % len(colors)]
            for mode in range(num_modes):
                label = Path(bands[idx]).stem if mode == 0 and i == 0 else None
                axs[0,i].plot(k_points[seg_tick[i][0]:seg_tick[i][-1]], frequencies[seg_tick[i][0]:seg_tick[i][-1], mode], color=c, alpha=0.5, linewidth=3, label=label)
                
        if dft_data is not None:
            dft_frequencies = dft_data['frequencies']
            num_modes = dft_data['num_modes']
            for mode in range(num_modes):
                label = dft_label_text if mode == 0 and i == 0 else None
                axs[0,i].plot(k_points[seg_tick[i][0]:seg_tick[i][-1]], dft_frequencies[seg_tick[i][0]:seg_tick[i][-1], mode], color='red', alpha=0.5, linewidth=3, linestyle='--', label=label)

        axs[0,i].tick_params(axis='both', labelsize=fsize)
        axs[0,i].set_xticks(seg_tick[i], labels=seg_labels[i])
        if fmin is not None and fmax is not None:
            axs[0,i].set_ylim([fmin, fmax])
        axs[0,i].set_xlim([k_points[seg_tick[i][0]], np.max(k_points[seg_tick[i][0]:seg_tick[i][-1]])+1])
        if i == 0:
            axs[0,i].set_ylabel(f'Frequency [THz]', fontsize=fsize)
        else:
            axs[0,i].set_yticklabels([])

    if len(data_list) >= 1:
        total_items = len(data_list)
        for idx, d in enumerate(data_list):
            label = ml_labels[idx] if ml_labels is not None else Path(bands[idx]).stem
            c = colors[idx % len(colors)]
            
            srmse = ""
            if dft_data is not None:
                val_rmse = rmse(dft_data['frequencies'], d['frequencies'])
                srmse = f" (RMSE: {val_rmse:.4f})"
                print(f"{label} RMSE: {val_rmse:.4f}")

            axs[0,0].text(0.0, 1.05 + (total_items - 1 - idx)*0.08, label + srmse, color=c, fontsize=fsize//2, 
                          transform=axs[0,0].transAxes, verticalalignment='bottom')
        
        if dft_data is not None:
            axs[0,0].text(0.0, 1.05 + total_items*0.08, dft_label_text, color='red', fontsize=fsize//2, 
                          transform=axs[0,0].transAxes, verticalalignment='bottom')

    plt.suptitle(title,fontsize=fsize)
    plt.tight_layout(pad=1.5)
    if args.save is None:
        plt.show()
    else:
        plt.savefig(f"{save_file}",transparent=True, bbox_inches='tight')

if __name__ == "__main__":
    main()
