#!/usr/bin/env python
 
"""sourcesub.py -- subtract the stars from a dragonfly image using a high resolution image (e.g. cfht) to identify and model stars

Usage:
    sourcesub [-h] [-v] [-m] [-u UPPERLIMIT] [-l LOCATION] [-p PSF] [-s LOC] <dragonflyimagename> <highresimagename> <paramfile>

Options:
    -h, --help                                  Show this screen
    -v, --verbose                               Show extra information [default: False]    

    -m, --usemodelpsf                           Use the model psf (produced via allison's code) [default: False]
    -l LOCATION, --locpsf LOCATION              Directory where the psf code is [default: /Users/deblokhorst/Documents/Dragonfly/git/]
    -u UPPERLIMIT, --upperlimit UPPERLIMIT      Uses a different upper limit for the final masking   [default: False]

    -p PSF, --givenpsf PSF                      PSF  name  [default: ./psf/_psf_g.fits]
    -s LOC, --sexloc LOC                    	Location of SExtractor executable						[default: /usr/local/bin/sex]

Examples:
    python sourcesub.py -v -s /opt/local/bin/sex -p ./psf/_psf_g_small.fits _df_g.fits _cfht_g.fits NGC4565_param.txt
    python sourcesub.py -v -s /opt/local/bin/sex -p ./psf/_psf_g.fits _df_g.fits _cfht_g.fits M101_param.txt

"""

import docopt

import pyraf
from pyraf import iraf
import numpy as np
from astropy.io import fits, ascii
from astropy.nddata.utils import block_replicate
from matplotlib import pyplot as plt
from iraf import stsdas
from iraf import analysis
try:
    from iraf import dither
except ImportError:
    print('trying again to import dither!')
    from iraf import dither
from iraf import fitting
from iraf import fourier
from iraf import fconvolve
import os 
import sys
import subprocess
import re



sextractor_params = """NUMBER
FLUX_AUTO 
FLUXERR_AUTO
X_IMAGE
Y_IMAGE
X_WORLD
Y_WORLD
ELLIPTICITY
FWHM_IMAGE
THETA_IMAGE
"""

sextractor_config = """
    ANALYSIS_THRESH 3
        BACK_FILTERSIZE 3
        BACKPHOTO_TYPE GLOBAL
        BACK_SIZE 128
        CATALOG_NAME test.cat
        CATALOG_TYPE ASCII_HEAD
        CLEAN Y
        CLEAN_PARAM 1.
        DEBLEND_MINCONT .005
        DEBLEND_NTHRESH 32
        DETECT_MINAREA 5
        DETECT_THRESH {detect_thresh}
        DETECT_TYPE CCD
        FILTER Y
        FILTER_NAME {filter_name}
        FLAG_IMAGE flag.fits
        GAIN 1.0
        MAG_GAMMA 4.
        MAG_ZEROPOINT 0.0
        MASK_TYPE CORRECT
        MEMORY_BUFSIZE 4096
        MEMORY_OBJSTACK 30000
        MEMORY_PIXSTACK 3000000
        PARAMETERS_NAME {parameters_name}
        PHOT_APERTURES 3
        PHOT_AUTOPARAMS 2.5, 3.5
        PIXEL_SCALE {pixel_scale}
        SATUR_LEVEL 50000.
        SEEING_FWHM 2.5
        STARNNW_NAME {starnnw_name}
        VERBOSE_TYPE {verbose_type}
"""


default_conv = """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels. 
1 2 1
2 4 2
1 2 1
"""

default_nnw = """NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs:       9 for profile parameters + 1 for seeing.
# outputs:      ``Stellarity index'' (0.0 to 1.0)
# Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
# Optimized for Moffat profiles with 2<= beta <= 4.
 3 10 10  1

-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00

-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00

-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00

 0.00000e+00
 1.00000e+00
"""


def mkdirp(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return None

def print_verbose_string(asverbose):
    print >> sys.stderr,'VERBOSE: %s' % asverbose

def cleanit(fname,iama='file'):
    if iama == 'file':
        if os.path.isfile(fname):
            os.remove(fname)
    elif iama == 'dir':
        shutil.rmtree(fname)
    return None

def run_SExtractor(imagename,detect_thresh=10):
    'Names and storage directory of required config files'
    sextractor_config_name = './pipetmp/scamp.sex'
    params_name = './pipetmp/scamp.param'
    nnw_name = './pipetmp/default.nnw'
    conv_name = './pipetmp/default.conv'
    mkdirp('./pipetmp')

    'Output ascii catalog name and seg map name'
    catname = re.sub('.fits','.cat',imagename)
    segname = re.sub('.fits','_seg.fits',imagename)
    rmsname = re.sub('.fits','_rms.fits',imagename)

    if verbose:
        verbose_type = 'NORMAL'
    else:
        verbose_type = 'QUIET'

    'Stick content in config files'
    configs = zip([sextractor_config_name,params_name,conv_name,nnw_name],[sextractor_config,sextractor_params,default_conv,default_nnw])
    for fname,fcontent in configs:
        fout = open(fname,'w')

        if 'scamp.sex' in fname:
            fout.write(fcontent.format(filter_name=conv_name,parameters_name=params_name,
                                       starnnw_name=nnw_name,verbose_type=verbose_type,
                                       detect_thresh=detect_thresh,pixel_scale=2.5))
        else:
            fout.write(fcontent)

        fout.close()

    if verbose:
        print_verbose_string('SExtracting... ')

    SExtract_command = sexloc +' -c {config} -CATALOG_NAME {catalog} -CHECKIMAGE_TYPE SEGMENTATION,BACKGROUND_RMS -CHECKIMAGE_NAME {seg},{rms} {image}'.format(config=sextractor_config_name,catalog=catname,seg=segname,rms=rmsname,image=imagename)
    print('command: ' + str(SExtract_command))
    subprocess.call(SExtract_command,shell=True)

    'Clean up'
    for fname in [sextractor_config_name,params_name,nnw_name,conv_name]:
        cleanit(fname)

    return catname


def run_SExtractor_dual(modelname,imagename,detect_thresh=10):
    'Names and storage directory of required config files'
    sextractor_config_name = './pipetmp/scamp.sex'
    params_name = './pipetmp/scamp.param'
    nnw_name = './pipetmp/default.nnw'
    conv_name = './pipetmp/default.conv'
    mkdirp('./pipetmp')

    'Output ascii catalog name and seg map name'
    catname = re.sub('.fits','.cat',imagename)
    segname = re.sub('.fits','_seg.fits',imagename)

    if verbose:
        verbose_type = 'NORMAL'
    else:
        verbose_type = 'QUIET'

    'Stick content in config files'
    configs = zip([sextractor_config_name,params_name,conv_name,nnw_name],[sextractor_config,sextractor_params,default_conv,default_nnw])
    for fname,fcontent in configs:
        fout = open(fname,'w')

        if 'scamp.sex' in fname:
            fout.write(fcontent.format(filter_name=conv_name,parameters_name=params_name,
                                       starnnw_name=nnw_name,verbose_type=verbose_type,
                                       detect_thresh=detect_thresh,pixel_scale=2.5))
        else:
            fout.write(fcontent)

        fout.close()

    if verbose:
        print_verbose_string('SExtracting... ')

    SExtract_command = sexloc +' -c {config} -CATALOG_NAME {catalog} -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME {seg} {model},{image}'.format(config=sextractor_config_name,catalog=catname,seg=segname,model=modelname,image=imagename)
    print(SExtract_command)
    subprocess.call(SExtract_command,shell=True)
    
    print('-------------------')
    print(SExtract_command)    
    print('catalog name :'+str(catname))
    print('-------------------')


    'Clean up'
    for fname in [sextractor_config_name,params_name,nnw_name,conv_name]:
        cleanit(fname)

    return catname



def getphotosc(model,df,xmin=100,xmax=500,ymin=100,ymax=500,numsources=50):
    print "\n************ Calculating the photometric scaling ************\n"

    # 'run source extractor on the Dragonfly image to get the noise (rms) map'
    # run_SExtractor(df,detect_thresh=3)
    # quit()
    # 'run in dual mode with the rms map from the dragonfly image'
    # catname_model = run_SExtractor_dual(model,model,detect_thresh=3)
    # catname_df = run_SExtractor_dual(model,df,detect_thresh=3)

    catname_df = run_SExtractor_dual(df,df,detect_thresh=3)
    catname_model = run_SExtractor_dual(df,model,detect_thresh=3)

    
    'read in the sextractor catalogue from the df run, to pick out some sources'
    cat_df = ascii.read(catname_df)
    flux_df = np.array(cat_df['FLUX_AUTO'])
    x_df = np.array(cat_df['X_IMAGE'])
    y_df = np.array(cat_df['Y_IMAGE'])

    'read in the sextractor catalogue from the model run, to pick out some sources'
    cat_model = ascii.read(catname_model)
    flux_model = np.array(cat_model['FLUX_AUTO'])
    
    # 'restrict the values to an inner section (since the cfht image is smaller than dragonfly)'
    # flux_df = flux_orig[(x_orig<500)&(x_orig>100)&(y_orig<500)&(y_orig>100)]
    # x = x_orig[(x_orig<500)&(x_orig>100)&(y_orig<500)&(y_orig>100)]
    # y = y_orig[(x_orig<500)&(x_orig>100)&(y_orig<500)&(y_orig>100)]
    # median_flux = np.median(flux)
    # #print np.transpose([flux,x,y]).tolist()
    
    'pick out some number (numsources) of sources with fluxes close to the median flux'
    diff = 0.002
    median_flux = np.median(flux_df)
    flux_df_nearby = flux_df[(flux_df < (median_flux+diff)) & (flux_df > (median_flux-diff))]
    while len(flux_df_nearby) < numsources:
        diff = diff+0.002
        flux_df_nearby = flux_df[(flux_df < (median_flux+diff)) & (flux_df > (median_flux-diff))]
    if verbose:
        print_verbose_string('Selected %s sources with flux close to the median (%s) -- within flux range of %s to %s'%
                                (len(flux_df_nearby),median_flux,median_flux-diff,median_flux+diff)) 
    x_nearby = x_df[(flux_df < (median_flux+diff)) & (flux_df > (median_flux-diff))]
    y_nearby = y_df[(flux_df < (median_flux+diff)) & (flux_df > (median_flux-diff))]
    flux_model_nearby = flux_model[(flux_df < (median_flux+diff)) & (flux_df > (median_flux-diff))]

    print(len(x_nearby))
    print(len(y_nearby))
    print(len(flux_model_nearby))

    #print np.transpose([flux_df_nearby,x_nearby,y_nearby])
    
    'write the locations to a region file (to display in ds9)'
    f = open('photoscsources.reg','w')
    f.write('global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    f.write('Image\n\n')
    for i in range(len(x_nearby)): 
        f.write('point('+str(x_nearby[i])+','+str(y_nearby[i])+') # point=circle\n')
    f.close()
    
    print('flux_df_nearby')
    print(flux_df_nearby)
    print('flux_model_nearby')
    print(flux_model_nearby)

    flux_ratio = flux_df_nearby/flux_model_nearby
    avgphotosc = np.mean(flux_ratio)
    medianphotosc = np.median(flux_ratio)
    
    plotchecks=True
    if plotchecks:
        #print min(flux_ratio)
        #print max(flux_ratio)
        numbins = len(flux_ratio)
        #print 'number of bins: %s'%numbins
        plt.hist(flux_ratio,bins=numbins)
        plt.axvline(x=avgphotosc,c='r',label='average')
        plt.axvline(x=medianphotosc,c='b',label='median')
        plt.title('Photosc Values')
        xmax = np.percentile(flux_ratio,95)
        #print 'plotting max: %s'%xmax
        plt.xlim(0,xmax)
        plt.legend()
        print "Saving histogram of photosc values to photosc_hist.pdf."
        plt.savefig('photosc_hist.pdf')
        #plt.show()

    # 'open the files'
    # df_data = fits.getdata(df)
    # model_data = fits.getdata(model)
    
    # 'find the standard deviations in 5x5 pixels centered on the sources we selected from the Dragonfly image in both that image and the model'
    # stdev_df = []
    # stdev_model = []
    # radius=2
    # for i in range(len(x_nearby)):
    #     bounds = [int(round(x_nearby[i])-radius),int(round(x_nearby[i])+radius+1),int(round(y_nearby[i])-radius),int(round(y_nearby[i])+radius+1)]
    #     stdev_df.append(np.std(df_data[bounds[2]:bounds[3],bounds[0]:bounds[1]]))
    #     stdev_model.append(np.std(model_data[bounds[2]:bounds[3],bounds[0]:bounds[1]]))
    # photosc = np.array(stdev_df)/np.array(stdev_model)
    # avgphotosc = np.mean(photosc)
    #print stdev_df
    #print stdev_model
    #print photosc
    
    'check for any outliers (in case, e.g. the cfht image is smaller than the cutout we took)'
    
    if verbose:
        print_verbose_string('The average photosc from %s sources is %s.'%(len(flux_ratio),avgphotosc))
        print_verbose_string('The median photosc from %s sources is %s.'%(len(flux_ratio),medianphotosc))

    return avgphotosc,medianphotosc


def writeFITS(im,header,saveas):
    if os.path.isfile(saveas):
        os.remove(saveas)
    hdu = fits.PrimaryHDU(data=im,header=header)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(saveas,overwrite=True)
    hdulist.close()

    return None

def prep(df_image,hi_res_image,width_mask=1.5):
    print "\n************ Running the preparation steps ************\n"
    
    'run SExtractor to get bright sources that are easily detected in the high resolution data'
    #####  Add in option to change sextractor threshold
    subprocess.call('sex %s' %hi_res_image,shell=True)
    'copy the segmentation map to a mask'
    iraf.imdel('_mask.fits')
    iraf.imdel('_fluxmod_cfht*.fits')
    iraf.imdel('_df_4*.fits')

    iraf.imcopy('seg.fits','_mask.fits')
    'replace the values in the segments in the segmentation map (i.e. stars) all to 1, all the background is still 0'
    iraf.imreplace('_mask.fits',1,lower=0.5)
    'multiply the mask (with 1s at the stars) by the high res image to get the star flux back - now have the flux model'
    iraf.imarith('_mask.fits','*','%s'%hi_res_image,'_fluxmod_cfht')
    
    'smooth the flux model'
    'increase size of mask, so more is subtracted'
    'the "1.5" in the next line should be a user-defined parameter that is given'
    'to the script; it controls how much of the low surface brightness'
    'emission in the outskirts of galaxies is subtracted. This choice'
    'depends on the science application'
    iraf.gauss('_fluxmod_cfht','_fluxmod_cfht_smoothed',width_mask,nsigma=4.)

    iraf.imreplace('_fluxmod_cfht_smoothed', -1, lower=0, upper=0)
    iraf.imreplace('_fluxmod_cfht_smoothed', 1, lower=-0.5)
    iraf.imreplace('_fluxmod_cfht_smoothed', 0, upper=-0.5)
    iraf.imarith('%s'%hi_res_image, '*','_fluxmod_cfht_smoothed', '_fluxmod_cfht_new')
    
    ' this is the key new step! we"re registering the CFHT image'
    ' to a frame that is 4x finer sampled than the Dragonfly image.'
    ' this avoids all the pixelation effects we had before.'
    '  (4x seems enough; but we could have it as a free parameter - need'
    ' to be careful as it occurs elsewhere in the scripts too) '
    iraf.blkrep('%s'%df_image,'_df_4',4,4)
    
    'register the flux model onto the same pixel scale as the dragonfly image'
    iraf.wregister('_fluxmod_cfht_new','_df_4','_fluxmod_dragonfly',interpo='linear',fluxcon='yes')

    return None

def psfconv(df_image,psf):
    print "\n************ Running the psf convolution steps ************\n"

    iraf.imdel('_model.fits')
    iraf.imdel('_model_4.fits')
    iraf.imdel('_res*.fits')
    iraf.imdel('_psf*.fits')
    iraf.imdel('_df_sub')
    
    'subtract the sky value from the dragonfly image header'
    try:
        df_backval = fits.getheader(df_image)['BACKVAL']
        iraf.imarith('%s'%df_image,'-','%s'%df_backval,'_df_sub')
    except:
        print "WARNING: No BACKVAL to subtract!  Skipping the background subtraction..."
        iraf.imcopy('%s'%df_image,'_df_sub.fits')
    
    ##### subtract the background from the cfht image?
    
    'convolve the model with the Dragonfly PSF'
    if usemodelpsf:
        makeallisonspsf()
        psf = './psf/psf_static_fullframe.fits'
        
    if verbose:
        print 'VERBOSE:  Using %s for the psf convolution.'%psf
        
    'resample the PSF by a factor of 4'
    iraf.magnify('%s'%psf,'_psf_4',4,4,interp="spline3")
    
    'this is just to retain the same total flux in the psf'
    iraf.imarith('_psf_4','*',16.,'_psf_4')

    iraf.stsdas.analysis.fourier.fconvolve('_fluxmod_dragonfly','_psf_4','_model_4')
    
    'now after the convolution we can go back to the Dragonfly resolution'
    iraf.blkavg('_model_4','_model',4,4,option="average")
    
    return None

# def subract(df_image,psf,shifts=None,photosc=3.70e-6,width_cfhtsm=0.45,upperlim=0.04,lowerlim=0.005):
def subract(df_image,shifts=99.99,width_cfhtsm=0.,upperlim=0.04,lowerlim=0.005,medphotosc=True,numsources=100):
    print "\n************ Running the subtraction steps ************\n"

    #iraf.imdel('_psf*.fits')
    #iraf.imdel('_model*.fits')
    #iraf.imdel('_df_sub')
    iraf.imdel('_res*.fits')
    iraf.imdel('cc_images')
    iraf.imdel('_model_sh')
    iraf.imdel('_model_sc')
    iraf.imdel('_model_mask')
    iraf.imdel('_model_maskb')
    
    'shift the images so they have the same physical coordinates'
    if shifts[0]==99.99:
        print('Calculating shifts...')
        iraf.stsdas.analysis.dither.crossdriz('_df_sub.fits','_model.fits','cc_images',dinp='no',dref='no')
        iraf.stsdas.analysis.dither.shiftfind('cc_images.fits','shift_values')
        x_shift=0
        y_shift=0
        with open('shift_values','r') as datafile:
            line = datafile.read().split()
            x_shift = float(line[2])
            y_shift = float(line[4])
        print('The shift in x and y are: '+str(x_shift)+','+str(y_shift))
         
        iraf.imshift('_model','_model_sh',0-x_shift,0-y_shift)
    else:
        iraf.imshift('_model','_model_sh',shifts[0],shifts[1])

    'scale the model so that roughly the same as the _df_sub image'
    if usemodelpsf:
        iraf.imarith('_model_sh','/',16.,'_model_sc')
    else:
        'the photometric step, matching the images to each other. '
        'in principle this comes from the headers - both datasets are calibrated,'
        'so this multiplication should be something like 10^((ZP_DF - ZP_CFHT)/-2.5)'
        '(perhaps with a correction for the difference in pixel size - depending'
        'on what wregister does - so another factor (PIX_SIZE_DF)^2/(PIX_SIZE_CFHT)^2'
        
        avgphotosc,medianphotosc = getphotosc('_model_sh.fits','_df_sub.fits',numsources=numsources)
        if medphotosc:
            photosc = medianphotosc
            print 'Using median calculated photosc: %s'%photosc
        else:
            photosc = avgphotosc
            print 'Using average calculated photosc: %s'%photosc

        iraf.imarith('_model_sh','*',photosc,'_model_sc')

    iraf.imdel('_df_ga.fits')
    
    ' correction for the smoothing that was applied to the CFHT'
    if width_cfhtsm != 0:
        iraf.gauss('_df_sub','_df_ga',width_cfhtsm) # iraf.gauss('%s'%df_image,'_df_ga',width_cfhtsm)
    else:
        print 'WARNING: No smoothing applied to the dragonfly image before subtracting the model.'
        iraf.imcopy('_df_sub.fits','_df_ga.fits')

    'subtract the model from the dragonfly cutout'
    iraf.imarith('_df_ga','-','_model_sc','_res')

    iraf.imcopy('_res.fits','_res_org.fits')

    iraf.imdel('_model_mask')
    iraf.imdel('_model_maskb')
    
    iraf.imcopy('_model_sc.fits','_model_mask.fits')
    iraf.imreplace('_model_mask.fits',0,upper=upperlim)
    iraf.imreplace('_model_mask.fits',1,lower=lowerlim)
    iraf.boxcar('_model_mask','_model_maskb',5,5)
    iraf.imreplace('_model_maskb.fits',1,lower=0.1)
    iraf.imreplace('_model_maskb.fits',0,upper=0.9)
    iraf.imarith(1,'-','_model_maskb','_model_maskb')
    iraf.imarith('_model_maskb','*','_res','_res_final')

    return None

def mask(res_org,upperlim=0.04):
    upperlim = float(upperlim)
    lowerlim_opt = upperlim/2

    iraf.imdel('_res_final.fits')
    iraf.imdel('_model_mask')
    iraf.imdel('_model_maskb')
    
    iraf.imcopy('_model_sc.fits','_model_mask.fits')
    iraf.imreplace('_model_mask.fits',0,upper=upperlim)
    iraf.imreplace('_model_mask.fits',1,lower=lowerlim_opt)
    iraf.boxcar('_model_mask','_model_maskb',5,5)
    iraf.imreplace('_model_maskb.fits',1,lower=0.1)
    iraf.imreplace('_model_maskb.fits',0,upper=0.9)
    iraf.imarith(1,'-','_model_maskb','_model_maskb')
    iraf.imarith('_model_maskb','*',res_org,'_res_final')

    return None


def makeallisonspsf():
    subprocess.call('mkdir -p ./psf',shell=True)
    if os.path.isfile('./psf/SloanG.CENTRALPSF.cube.fits') is False:
        print 'need to make psf data cube from the dragonfly image'
        subprocess.call('python %s/DragonflyPSF-v0.1/DragonflyPSF/build_dragonfly_psf.py -v -d %s -x 2 -y 2'%(locpsf,df_image),shell=True)
    if os.path.isfile('./psf/psf_static_fullframe.fits') is False:
        print 'grab the psf and save to a file'
        subprocess.call('python %s/DragonflyPSF-v0.1/DragonflyPSF/grab_dragonfly_psf.py -v -d -o psf/ --save --static psf/SloanG.CENTRALPSF.cube.fits'%locpsf,shell=True)
        
    return None
        

if __name__ == '__main__':
    
    arguments = docopt.docopt(__doc__)

    # Mandatory argument
    df_image = arguments['<dragonflyimagename>']
    hi_res_image = arguments['<highresimagename>']
    parameters_file = arguments['<paramfile>']
    
    # Non-mandatory options without arguments
    verbose = arguments['--verbose']
    usemodelpsf = arguments['--usemodelpsf']
    upperlim_opt = arguments['--upperlimit']
    
    # Non-mandatory options with arguments
    psf = arguments['--givenpsf']
    locpsf = arguments['--locpsf']
    sexloc = arguments['--sexloc']
    
    if verbose:
        print arguments
    
    if usemodelpsf:
        print '\nNOTE: Using model psf from Allisons code (assuming in path: %s)\n'%locpsf
    else:
        print '\nNOTE: Using cutout of star from dragonfly image as psf - need to have cutout named %s in directory or will crash.\n'%psf
        
    if psf is None and usemodelpsf is False:
        print 'ERROR: If not using a model psf from Allisons code, need to specify the name of the psf fits file to be used.\n'
        sys.exit()
    

    'Reading parameters from the user'
    user_parameters = []
    use_or_not = []
    with open(parameters_file,'r') as i:
        lines = i.readlines()
    for ind in range(7,len(lines)):
    	line = lines[ind]
    	user_parameters.append(float(line.split('\t')[0]))
    	use_or_not.append(int(line.split('\t')[1]))


    default_param = [0.04, 0.005, 0.15, 0.30, 0., 1.5]
    parameters_to_use = np.asarray(default_param)
    
    for i in range(len(default_param)):
    	if use_or_not[i] == 1:
    		parameters_to_use[i] = user_parameters[i]

    upperlim = parameters_to_use[0]
    lowerlim = parameters_to_use[1]
    shifts = [parameters_to_use[2],parameters_to_use[3]]
    width_cfhtsm = parameters_to_use[4]
    width_mask = parameters_to_use[5]
    
   # photosc = getphotosc('_model_sh.fits','_df_sub.fits',xmin=100,xmax=500,ymin=100,ymax=500,numsources=200)
#    print "Photosc: %s"%photosc
#    quit()
    
   # subract(df_image,shifts=shifts,width_cfhtsm=width_cfhtsm,upperlim=upperlim,lowerlim=lowerlim)
    #quit()
    
    if upperlim_opt=='False':
        print 'Running the entire source subtraction code'
        prep(df_image,hi_res_image,width_mask=width_mask)
        psfconv(df_image,psf)
        subract(df_image,shifts=shifts,width_cfhtsm=width_cfhtsm,upperlim=upperlim,lowerlim=lowerlim)
    else:
        print 'Only performing the masking on the residual image - _res_org.fits.\n'
        mask('_res_org.fits',upperlim=upperlim_opt)
        
        
        
