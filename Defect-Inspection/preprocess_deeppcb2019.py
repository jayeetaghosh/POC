import numpy as np
import pandas as pd
import os, csv
from PIL import Image
import time

id2_defect={'0':'background','1':'open','2':'short','3':'mousebite',\
	'4':'spur','5':'copper','6':'pin-hole'}

# PCB_ROOT='C:\dev\pcbdata\PCBData'    # root input folder
PCB_ROOT="C:/Users/JGhosh/Documents/Python_Scripts/DGX-1_POC/DefectInspection/UNet_Industrial/data/PCBData/"

# FRCNN_ROOT='C:\dev\pcbdata\PCBData'  # root output folder
UNET_ROOT="C:/Users/JGhosh/Documents/Python_Scripts/DGX-1_POC/DefectInspection/UNet_Industrial"

# --- create dictionaries with group:filepaths
not_d = {}
img_d = {}


def get_paths(verbose=False):
    ''' Parse file structure of PCBData files under PCB_ROOT '''

    d = PCB_ROOT
    folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

    # print(folders)
    for f in folders:
        # group = f.split('\\')[-1]
        group = f.split('/')[-1]

        # print(group)
        f = f + "/"
        subfolders = [os.path.join(f, o) for o in os.listdir(f) if os.path.isdir(os.path.join(f, o))]
        # print(subfolders)

        for f2 in subfolders:
            f2 = f2 + "/"
            if '_not' in str(f2):
                # --- annotation files
                not_files = [os.path.join(f2, o) for o in os.listdir(f2) if not os.path.isdir(os.path.join(f2, o))]
                not_d[group] = not_files
                if verbose: print('Group {}: \t {} annotation files'.format(group, len(not_files)))
            else:
                # --- image files
                img_files = [os.path.join(f2, o) for o in os.listdir(f2) if not os.path.isdir(os.path.join(f2, o))]
                img_d[group] = img_files
                if verbose: print('Group {}: {} image files'.format(group, len(img_files)))
    return img_d, not_d

def annotation_rename(iname):
    ''' Write new annotation file in PCBData folder with F-RCNN  input format '''
    # parts=iname.split('\\')
    head, tail = os.path.split(iname)
    head = head + "/"
    oname=tail.split('.')
    oname=os.path.join(head,oname[0]+'_new.'+oname[1])
    return oname

def parse_annotation_files(grp,d):
    ''' Parse PCBData annotation files for group; re-format for input to R-CNN
	    Write output files to existing PCBData folders in place.
    '''
    files=d[grp]
    # print(files)
    for fname in files:
        if '_new' in fname: continue
        # print('File: {}'.format(fname))
        with open(fname,'r') as f:
            # x1,y1,x2,y2,class ID
            recs=[l.strip('\n').replace(' ',',') for l in f.readlines()]

            outlines=[]
            for r in recs:
                toks=[t for t in r.split(',')]

               # replace ID with defect type name
                toks[-1]=id2_defect[str(toks[-1])]

                # Calculate semi-major and semi-minor axes, and center of the ellipse. Keeping 0.0 as rotational angles
                xdiff = float(toks[2]) - float(toks[0])
                ydiff = float(toks[3]) - float(toks[1])
                semimajor = abs(xdiff)/2
                semiminor = abs(ydiff)/2
                xcenter = float(toks[0]) + xdiff/2
                ycenter = float(toks[1]) + ydiff/2
                rotangle = 0.0
                outs = [semimajor, semiminor, rotangle, xcenter, ycenter, toks[-1]]
                outs = '{0}'.format(', '.join(map(str, outs)))

                outlines.append(outs)
                # print(outlines)
            #--- write records to UNet annotation format file
            oname=annotation_rename(fname)
            # print(oname)
            with open(oname,'w') as of:
                of.write('\n'.join(outlines))
            of.close()
        f.close()
        os.remove(fname)
        print("Original annotation file removed", fname)
    return None

def change_image_format(grp,d):
    ''' Parse PCBData annotation files for group; re-format for input to R-CNN
	    Write output files to existing PCBData folders in place.
    '''
    files=d[grp]
    # print(files)
    for fname in files:
        if 'PNG' in fname: continue
        # print('File: {}'.format(fname))

        file, ext = os.path.splitext(fname)
        # print(file, ext)
        im = Image.open(fname)
        rgb_im = im.convert('RGB')
        out = file + ".PNG"
        # print(out)
        rgb_im.save(out)
        os.remove(fname)
        print("Original image file removed", fname)


    return None

if __name__ == '__main__':

    img_d, not_d = get_paths()
    start_time = time.time()
    for grp in img_d.keys():

        print('Running Group: {}'.format(grp))
        print("**************************")
        parse_annotation_files(grp, not_d)
        # change_image_format(grp, img_d)

        # exit()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Done!")