#!/usr/bin/env python
 
"""starsub.py -- subtract the stars from a dragonfly image using a high resolution image (e.g. cfht) to identify and model stars

Usage:
    starsub [-h] [-v] [-u] [-m] [-l LOCATION] [-p PSF] <dragonflyimagename> <highresimagename>

Options:
    -h, --help                                  Show this screen
    -v, --verbose                               Show extra information [default: False]    

    -u, --upsample                              Upsample the dragonfly image and psf [default: False]

    -m, --usemodelpsf                           Use the model psf (produced via allison's code) [default: False]
    -l LOCATION, --locpsf LOCATION              Directory where the psf code is [default: /Users/deblokhorst/Documents/Dragonfly/git/]

    -p PSF, --givenpsf PSF                      PSF  name  [default: ./psf/_psf_g.fits]

Examples:

"""

import docopt

import pyraf
from pyraf import iraf
import numpy as np
from astropy.io import fits
from astropy.nddata.utils import block_replicate
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

def writeFITS(im,header,saveas):
    if os.path.isfile(saveas):
        os.remove(saveas)
    hdu = fits.PrimaryHDU(data=im,header=header)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(saveas,overwrite=True)
    hdulist.close()

    return None
    
def upsample(inimage,outimage,factor=2):
    data,header = fits.getdata(inimage,header=True)
    data_upsample = block_replicate(data,factor)
    header['NAXIS1'] = header['NAXIS1']*factor
    header['NAXIS2'] = header['NAXIS2']*factor
    header['comment'] = 'Data was upsampled from original image, %s, by factor of %s'%(inimage,factor)
    writeFITS(data_upsample,header,outimage)

    return None

def prep(df_image,hi_res_image,width=1.5):
    'run SExtractor to get bright sources that are easily detected in the high resolution data'
    
    #####  option to change sex threshold 
    subprocess.call('sex %s' %hi_res_image,shell=True)
    'copy the segmentation map to a mask'
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
    #####  option to provide width for the threshold 
    iraf.gauss('_fluxmod_cfht','_fluxmod_cfht_smoothed',width,nsigma=4.)
    
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

def subract(df_image,psf,shifts=None):
##    iraf.imdel('_model*.fits')
    iraf.imdel('_res*.fits')
##    iraf.imdel('_psf*.fits')
##    iraf.imdel('_df_sub')
    
    'subtract the sky value from the dragonfly image header'
  #####  df_backval = fits.getheader(df_image)['BACKVAL']
  #####  iraf.imarith('%s'%df_image,'-','%s'%df_backval,'_df_sub')

    'convolve the model with the Dragonfly PSF'
    if usemodelpsf:
        makeallisonspsf()
        psf = './psf/psf_static_fullframe.fits'
        
    if verbose:
        print 'VERBOSE:  Using %s for the psf convolution.'%psf
        
    'resample the PSF by a factor of 4'
##    iraf.magnify('%s'%psf,'_psf_4',4,4,interp="spline3")
    
    'this is just to retain the same total flux in the psf'
##    iraf.imarith('_psf_4','*',16.,'_psf_4')

##    iraf.stsdas.analysis.fourier.fconvolve('_fluxmod_dragonfly','_psf_4','_model_4')
    
    # now after the convolution we can go back to the Dragonfly resolution
##    iraf.blkavg('_model_4','_model',4,4,option="average")
    
    iraf.imdel('cc_images')
    
    'shift the images so they have the same physical coordinates'
    if shifts is None:
        iraf.stsdas.analysis.dither.crossdriz('_df_g.fits','_model.fits','cc_images',dinp='no',dref='no')
        #####iraf.stsdas.analysis.dither.crossdriz('_df_sub.fits','_model.fits','cc_images',dinp='no',dref='no')
        iraf.stsdas.analysis.dither.shiftfind('cc_images.fits','shift_values')
        x_shift=0
        y_shift=0
        with open('shift_values','r') as datafile:
            line = datafile.read().split()
            x_shift = float(line[2])
            y_shift = float(line[4])
        print('The shift in x and y are: '+str(x_shift)+','+str(y_shift))
         
        iraf.imdel('_model_sh')
        iraf.imdel('_model_sc')
        iraf.imshift('_model','_model_sh',0-x_shift,0-y_shift)
    else:
        # for M101: 0.15,0.30
        iraf.imshift('_model','_model_sh',shifts[0],shifts[1])

    'scale the model so that roughly the same as the _df_sub image'
 ####   if usemodelpsf:
 ####       iraf.imarith('_model_sh','/',16.,'_model_sc')  ## just trying to match it to the original (below) approximately
 ####   else:
 ####       iraf.imarith('_model_sh','/',2422535.2,'_model_sc')  ## difference between the flux of the star used to make the psf in both frames

 ####   iraf.imarith('_model_sc','*',1.5,'_model_sc')

    #  And this is the photometric step, matching the images to each other
    # in principle this comes from the headers - both datasets are calibrated,
    #  to this multiplication should be something like 10^((ZP_DF - ZP_CFHT)/-2.5)
    # (perhaps with a correction for the difference in pixel size - depending
    #  on what wregister does - so another factor (PIX_SIZE_DF)^2/(PIX_SIZE_CFHT)^2
    #  we can also measure it
    # again, I think having this factor be part of the parameters that are
    #  used to call the script is probably best - as it really should follow
    # from known information (so have the 3.7e-6 in the next line be an
    #   input parameter)
    ##### FROM PIETER ## COMMENT THIS OUT UNLESS TESTING
    iraf.imarith('_model_sh','*',3.70e-6,'_model_sc')
    #####

    iraf.imdel('_df_ga.fits')
    
    # next is a correction for the smoothing that was applied to the CFHT
    #  data in the prep script -but I'm not actually sure this is right!
    iraf.gauss('%s'%df_image,'_df_ga',0.45)

    iraf.imarith('_df_ga','-','_model_sc','_res')

    'subtract the model frm the dragonfly cutout'
    ##### iraf.imarith('_df_sub','-','_model_sc','_res')


    # next is for g band
    iraf.imcopy('_res.fits','_res_org.fits')

    iraf.imdel('_model_mask')
    iraf.imdel('_model_maskb')
    
    iraf.imcopy('_model_sc.fits','_model_mask.fits')
    iraf.imreplace('_model_mask.fits',0,upper=0.04)
    iraf.imreplace('_model_mask.fits',1,lower=0.005)
    iraf.boxcar('_model_mask','_model_maskb',5,5)
    iraf.imreplace('_model_maskb.fits',1,lower=0.1)
    iraf.imreplace('_model_maskb.fits',0,upper=0.9)
    iraf.imarith(1,'-','_model_maskb','_model_maskb')
    iraf.imarith('_model_maskb','*','_res','_res_final')
    
    '????'
  #####  iraf.imcopy('_model_sc','_model_mask')
  #####  iraf.imreplace('_model_mask.fits',0,upper=50)
  #####  iraf.imreplace('_model_mask.fits',1,lower=0.01)
  #####  iraf.boxcar('_model_mask','_model_maskb',5,5)
  #####  iraf.imreplace('_model_maskb.fits',1,lower=0.1)
  #####  iraf.imreplace('_model_maskb.fits',0,upper=0.9)
  #####  iraf.imarith(1,'-','_model_maskb','_model_maskb')
  #####  iraf.imarith('_model_maskb','*','_res','_res_final')

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
    
    # Non-mandatory options without arguments
    verbose = arguments['--verbose']
    usemodelpsf = arguments['--usemodelpsf']
    doupsample = arguments['--upsample']
    
    # Non-mandatory options with arguments
    psf = arguments['--givenpsf']
    locpsf = arguments['--locpsf']
    
    if verbose:
        print arguments
    
    if usemodelpsf:
        print '\nNOTE: Using model psf from Allisons code\n'
    else:
        print '\nNOTE: Using cutout of star from dragonfly image as psf - need to have cutout named %s in directory or will crash.\n'%psf
        
    if psf is None and usemodelpsf is False:
        print 'ERROR: If not using a model psf from Allisons code, need to specify the name of the psf fits file to be used.\n'
        sys.exit()
    
    #####  read a file here with the parameters
    
  #  prep(df_image,hi_res_image,width=width)
    subract(df_image,psf)
