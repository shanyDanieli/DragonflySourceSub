imdel "_*.fits"
#imcopy "cfht/MegaPipe.251.287.G.MP9401.fits[5982:8982,5682:8682]" _cfht_g
#imcopy "cfht/MegaPipe.251.287.R.MP9601.fits[5982:8982,5682:8682]" _cfht_r
imcopy cfht/W3-1+1.R _cfht_r
imcopy cfht/W3-1+1.G _cfht_g

#imcopy "cfht/MegaPipe.251.287.G.MP9401.fits" _cfht_g
#imcopy "cfht/MegaPipe.251.287.R.MP9601.fits" _cfht_r

imcopy "dragonfly/M101_g_final.fits[1:1720,3620:5000]" _df_g
imcopy "dragonfly/M101_r_final.fits[1:1720,3620:5000]" _df_r

#imcopy "dragonfly/M101_clean_g.fits[1550:3300,1870:3714]" _df_g_cl
#imcopy "dragonfly/M101_clean_r.fits[1550:3300,1870:3714]" _df_r_cl

!sex _cfht_g.fits
imcopy seg.fits _maskg.fits
imreplace _maskg.fits 1 lower=0.5
imarith _maskg.fits * _cfht_g _fluxmod_cfht_g

!sex _cfht_r.fits
imcopy seg.fits _maskr.fits
imreplace _maskr.fits 1 lower=0.5
imarith _maskr.fits * _cfht_r _fluxmod_cfht_r

# increase size of mask, so more is subtracted
#  the "1.5" in the next line should be a user-defined parameter that is given
# to the script; it controls how much of the low surface brightness
#  emission in the outskirts of galaxies is subtracted. This choice
#  depends on the science application.
gauss _fluxmod_cfht_r _fluxmod_cfht_rs 1.5 nsigma=4.
### Pieter
imreplace _fluxmod_cfht_rs -1 lower=0 upper=0
imreplace _fluxmod_cfht_rs 1 lower=-0.5
imreplace _fluxmod_cfht_rs 0 upper=-0.5
### Pieter
imarith _cfht_r * _fluxmod_cfht_rs _fluxmod_cfht_r_new

gauss _fluxmod_cfht_g _fluxmod_cfht_gs 1.5 nsigma=4.
imreplace _fluxmod_cfht_gs -1 lower=0 upper=0
imreplace _fluxmod_cfht_gs 1 lower=-0.5
imreplace _fluxmod_cfht_gs 0 upper=-0.5
imarith _cfht_g * _fluxmod_cfht_gs _fluxmod_cfht_g_new

# this is the key new step! we're registering the CFHT image
#  to a frame that is 4x finer sampled than the Dragonfly image.
# this avoids all the pixelation effects we had before. 
#  (4x seems enough; but we could have it as a free parameter - need
# to be careful as it occurs elsewhere in the scripts too) 
blkrep _df_g _df_g4 4 4


wregister "_fluxmod_cfht_g_new" "_df_g4" "_fluxmod_dragonfly_g" interpo="linear" fluxcon=yes
wregister "_fluxmod_cfht_r_new" "_df_g4" "_fluxmod_dragonfly_r" interpo="linear" fluxcon=yes


