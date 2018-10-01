imdel "_psf*.fits,_model*.fits"
imdel "_res*.fits"


# the psf ("psf/_psf_g" in next line)
# should come from Allison now - but for this script, probably
#  good to have it be part of the arguments that call the script
# (so it can also be run with other PSFs, and is not explicitly tied
#  to allison's stuff)

#  the next step is key as we need to convolve the CFHT model with a
#  PSF that is on the same subsampled grid as the CFHT data
#   ideally we create subsampled Dragonfly PSFs properly - that may
# not be hard within Allison's code, as it just requires projecting all
#  the shifted individual stars onto a 4x finer grid.
#  the 'magnify' step is not ideal but it's an OK way to resolve this
#  issue right now
magnify "psf/_psf_g" _psf_g4 4 4 interp="spline3"

#  this is just to retain the same total flux in the psf
imarith _psf_g4 * 16 _psf_g4

# here's the convolution step: takes longer now as it's on the
#  subsampled grid
fconvolve _fluxmod_dragonfly_g _psf_g4 _model_g4

# now after the convolution we can go back to the Dragonfly resolution
blkavg _model_g4 _model_g 4 4 option="average"

# next step needs to be done properly - we either need to make sure that
#  the images we feed into this are on the exact same system AND the
# PSF is exactly centered, or we need to solve for the shift every time
# we run the script. My instinct is to do something in between:
# we can add shifts to the callable parameters, so we can
#  apply "hand-measured" shifts if we want to (or measure them in
# another script before running this one, and then feeding the parameters)
#  So I'd say let's make dx and dx input parameters (with 0 as defaults)
imshift _model_g _model_gsh 0.15 0.30


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
imarith _model_gsh * 3.70e-6 _model_gsc

imdel _df_gg

# next is a correction for the smoothing that was applied to the CFHT
#  data in the prep script -but I'm not actually sure this is right!

gauss _df_g _df_gg 0.45

imarith _df_gg - _model_gsc _res_g


# next is for the r band - same steps
#  this should of course all be removed as we want to run it on 1 image
# at a time

###imcopy _df_r[184:204,110:130] _psf_r
##magnify "psf/_psf_r" _psf_r4 4 4 interp="spline3"
##imarith _psf_r4 * 16 _psf_r4
###blkrep _psf_r _psf_r4 4 4
##fconvolve _fluxmod_dragonfly_r _psf_r4 _model_r4
###fconvolve _fluxmod_dragonfly_r _gauss_r4 _model_r4
##blkavg _model_r4 _model_r 4 4 option="average"
###imshift _model_r _model_rsh -0.09 -0.30
##imshift _model_r _model_rsh -0.24 0.07
##imarith _model_rsh * 1.57e-6 _model_rsc
###imarith _model_rsh * 6.524e-4 _model_rsc
##imdel _df_rg
###gauss _df_r _df_rg 0.5
##gauss _df_r _df_rg 0.52
##imarith _df_rg - _model_rsc _res_r



# next steps are to mask the brightest residuals (replace them with 0)

##imcopy _res_r _res_r_org

##imcopy _model_rsc _model_mask

# 0.02 in next line is limit for masking: all pixels where the model flux
# is higher than 0.02 are masked in the residual frame!  this should
# be a free parameter, and we may want to use some clever scaling for it
#  eventually. for now it should be part of the input parameters
#  (it's effectively a surface brightness limit - i.e. mask all residuals
#  for which the surface brightness in the model is higher than, say,
#  20 mag / arcsec^2 - so we might want to give this parameter as
#   a surface brightness limit rather than, say, "0.02" )
##imreplace _model_mask 0 upper=0.02
##imreplace _model_mask 1 lower=0.01
###boxcar _model_mask _model_maskb 5 5
##boxcar _model_mask _model_maskb 3 3
##imreplace _model_maskb 1 lower=0.1
##imreplace _model_maskb 0 upper=0.9
##imarith 1 - _model_maskb _model_maskb
##imarith _model_maskb * _res_r _res_r

# right now the masked pixels are set to 0. this assumes that the background
# is subtracted from the dragonfly data! 
#  We can adapt the script in the following way to make sure of this:
# - early in the script:subtract the header parameter BACKGROUND from the frame
# - run the script
# - add BACKGROUND back to the frame
# 
#  the output should be 2 things, by the way: _res_r_org  is the original
#  residual, and _res_r is the residual with the high surface brightness
#  stuff masked. we want both, I think  (with different names of course!)


# next is for g band
imcopy _res_g _res_g_org

imdel _model_mask
imdel _model_maskb
imcopy _model_gsc _model_mask
imreplace _model_mask 0 upper=0.04
imreplace _model_mask 1 lower=0.005
boxcar _model_mask _model_maskb 5 5
imreplace _model_maskb 1 lower=0.1
imreplace _model_maskb 0 upper=0.9
imarith 1 - _model_maskb _model_maskb
imarith _model_maskb * _res_g _res_g

