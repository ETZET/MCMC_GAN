import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import wandb
import sys
import os
from process_data import *
from generative_model import WGAN_SIMPLE
import scipy.io as sio
import seaborn as sns
from scipy.stats import kurtosis,skew
import matplotlib.colors as colors
from scipy.stats import wasserstein_distance
from matplotlib.gridspec import  GridSpec
import random
import pooch
import geopandas as gp
import regionmask
from mpl_toolkits.basemap import Basemap, cm, maskoceans
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy
from matplotlib.patches import Polygon as MplPolygon
import shapefile
import pdb
import cartopy.feature as cfeature

cratons = sio.loadmat('/scratch/tolugboj_lab/Prj6_AfrTomography/2_Data/geoData/cratons/AfricanCratons.mat')
cratons = [cratons['Congo'],cratons['Kala'],cratons['Sahara'],cratons['Tanza'],cratons['West'],cratons['Zim']]
period = 30



# load the saved model file and MCMC samples  
data = np.genfromtxt('./datasets/Rayleigh_P30_downsampled_flat_extended.csv',delimiter=',',skip_header=True)
model = WGAN_SIMPLE(ndim=data.shape[1])
checkpoint = torch.load('./model/WGAN_Simple_epoch199.model')
model.load_state_dict(checkpoint["model_state_dict"])

#Load lat long mappings, and scaler file..
cord = np.genfromtxt('downsampled_points.csv',delimiter=',',skip_header=True)
scaler_file = open('./datasets/whole_scaler_extended.pkl', 'rb')
scaler = pickle.load(scaler_file)

# generate 10,000 mcmc samples using trained WGAN
fake_data = np.zeros((100000,2382))
for i in range(1000):
    left_idx = 100 * i
    right_idx = 100 * (i+1)
    fake_data[left_idx:right_idx,:] = \
            model.gen(torch.randn(100, model.nlatent, device="cpu")).detach().numpy()

# scale the generated data back to data range
fake_data_scaled = scaler.inverse_transform(fake_data)




real_mean = np.mean(data,axis=0)
real_skew = skew(data,axis=0)
real_std = np.std(data,axis=0)
real_kurt = kurtosis(data,axis=0)

fake_mean = np.mean(fake_data_scaled,axis=0)
fake_std = np.std(fake_data_scaled,axis=0)
fake_skew = skew(fake_data_scaled,axis=0)
fake_kurt = kurtosis(fake_data_scaled,axis=0)

data_pandas = pd.DataFrame(data)
data_cov = data_pandas.cov()
fake_pandas = pd.DataFrame(fake_data_scaled)
fake_cov = fake_pandas.cov()


#Download Africa Mask
file = pooch.retrieve(
    "https://pubs.usgs.gov/of/2006/1187/basemaps/continents/continents.zip", None
)

continents = gp.read_file("zip://" + file)

lon = np.linspace(min(x),max(x),1000)
lat = np.linspace(min(y),max(y),1000)

#lon = np.arange(min(x), max(x))
#lat = np.arange(min(y), max(y))
Africa = continents[continents['CONTINENT']=='Africa']
mask = regionmask.mask_geopandas(Africa,lon,lat)
#mask.plot()

#Sort by LON and interpolate w/ Cubic Spline
cord_pd = pd.DataFrame(cord)
cord_pd['point_spread_real'] = point_spread_real
cord_pd['point_spread_fake'] = point_spread_fake
cord_pd['real_mean'] = real_mean
cord_pd['fake_mean'] = fake_mean
cord_pd['real_std'] = real_std
cord_pd['fake_std'] = fake_std
cord_pd['real_skew'] = real_skew
cord_pd['fake_skew'] = fake_skew
cord_pd['real_kurt'] = real_kurt
cord_pd['fake_kurt'] = fake_kurt






cord_pd.columns = ['lon','lat','ps_real','ps_fake','real_mean','fake_mean','real_std','fake_std','real_skew','fake_skew','real_kurt','fake_kurt']
cord_pd_sorted = cord_pd.sort_values('lon')
x = cord_pd_sorted.lon
y = cord_pd_sorted.lat
lllat = min(y)
lllon = min(x)
urlat =  max(y)
urlon = max(x)
#m = Basemap(llcrnrlat=lllat,
#urcrnrlat=urlat,
#llcrnrlon=lllon,
#urcrnrlon=urlon,
#resolution='h', projection='cyl')
#Try making continuous map
#mlon, mlat = m(*(lon, lat))
xi, yi = np.meshgrid(lon, lat)
mlon, mlat = xi,yi
zi_ps_real = griddata((x,y), cord_pd_sorted.ps_real, (xi, yi), method='cubic', rescale=False)
zi_ps_fake = griddata((x,y), cord_pd_sorted.ps_fake, (xi, yi), method='cubic', rescale=False)
zi_mean_real = griddata((x,y), cord_pd_sorted.real_mean, (xi, yi), method='cubic', rescale=False)
zi_mean_fake = griddata((x,y), cord_pd_sorted.fake_mean, (xi, yi), method='cubic', rescale=False)
zi_std_real = griddata((x,y), cord_pd_sorted.real_std, (xi, yi), method='cubic', rescale=False)
zi_std_fake = griddata((x,y), cord_pd_sorted.fake_std, (xi, yi), method='cubic', rescale=False)
zi_skew_real = griddata((x,y), cord_pd_sorted.real_skew, (xi, yi), method='cubic', rescale=False)
zi_skew_fake = griddata((x,y), cord_pd_sorted.fake_skew, (xi, yi), method='cubic', rescale=False)
zi_kurt_real = griddata((x,y), cord_pd_sorted.real_kurt, (xi, yi), method='cubic', rescale=False)
zi_kurt_fake = griddata((x,y), cord_pd_sorted.fake_kurt, (xi, yi), method='cubic', rescale=False)

for i in range(len(mask[0])):
    for j in range(len(mask[1])):
        if np.isnan(mask[i,j]):
            zi_ps_real[i,j] = None
            zi_ps_fake[i,j] = None
            zi_mean_real[i,j] = None
            zi_mean_fake[i,j] = None
            zi_std_real[i,j] = None
            zi_std_fake[i,j] = None
            zi_skew_real[i,j] = None
            zi_skew_fake[i,j] = None
            zi_kurt_real[i,j] = None
            zi_kurt_fake[i,j] = None
min_std = round(min(real_std),2)
max_std = round(max(real_std),2)
min_mean = round(min(real_mean),2)
max_mean = round(max(real_mean),2)
min_pt_cov = round(min(point_spread_real),2)
max_pt_cov = round(max(point_spread_real),2)
min_skew = round(min(real_skew),2)
max_skew = round(max(real_skew),2)
min_kurt = round(min(real_kurt),2)
max_kurt = round(max(real_kurt),2)

#Plot Proposal Summary Plot
#Big Plot

point_spread_real = np.array(data_cov[indx].iloc[1,:])
point_spread_fake = np.array(fake_cov[indx].iloc[1,:])
x=cord[:,0]
y=cord[:,1]
z_mean = [real_mean,fake_mean]
z_std = [real_std,fake_std]
z_skew = [real_skew,fake_skew]
z_kurt = [real_kurt,fake_kurt]

figure = plt.figure(figsize = (10,10),zorder = 699)
axes = figure.add_axes([0.0, 0.0, 1, 1],projection= ccrs.PlateCarree())
#gps = ax = plt.axes(projection= ccrs.PlateCarree())
gps = axes.pcolormesh(mlon, mlat, zi_ps_fake,cmap=plt.get_cmap('seismic'),norm=colors.Normalize(vmin=min_pt_cov,vmax=max_pt_cov),zorder=0,transform = ccrs.PlateCarree())
axes.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes.add_feature(cfeature.COASTLINE,zorder=4,linewidth=3)



axes.text(-20,38,'R = '+ str(period),fontsize=16)



for craton in cratons:
        axes.plot(craton[:,1],craton[:,0],color='black',linewidth=1)
        axes.plot(craton[:,1],craton[:,0],color='black',linewidth=1)
        
        
axes.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',s=150,zorder=99)
#plt.colorbar(gps,location='left')
#im=axes.imshow(point_spread_fake)
cbbox = inset_axes(axes, '2%', '85%', loc = 5)
#cb=figure.colorbar(im,cax=cbbox,cmap='seismic')
cbbox.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,     
    right=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False,
    labelright=False,
    labeltop=False,
    labelsize=12) # labels along the bottom edge are off
axes.set_title('GAN Point Covariance')
cb=figure.colorbar(gps,cax=cbbox,cmap='RdYlGn',ticks = [min_pt_cov,max_pt_cov])
cbbox.yaxis.set_ticks_position('left')
#cbbox.yticks(fontsize=12)
#cb.set_ticks([])

axes2 = figure.add_axes([.011, 0.0, 0.4, 0.4],projection= ccrs.PlateCarree()) # inset axes
axes2.pcolormesh(mlon, mlat, zi_ps_real,cmap=plt.get_cmap('seismic'),norm=colors.Normalize(vmin=min_pt_cov,vmax=max_pt_cov),zorder=0,transform = ccrs.PlateCarree())
axes2.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes2.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes2.add_feature(cfeature.COASTLINE,zorder=4,linewidth=2)
axes2.set_title('MCMC Point Covariance')
axes2.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False, 
    right=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
axes2.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',zorder=99)

for craton in cratons:
        axes2.plot(craton[:,1],craton[:,0],color='black',linewidth=.65)
        axes2.plot(craton[:,1],craton[:,0],color='black',linewidth=.65)

axes3 = figure.add_axes([.972,.5,.5,.5],projection= ccrs.PlateCarree())
GM = axes3.pcolormesh(mlon, mlat, zi_mean_fake,cmap=plt.get_cmap('RdBu'),norm=colors.Normalize(vmin=min_mean,vmax=max_mean),zorder=0,transform = ccrs.PlateCarree())
axes3.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes3.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes3.add_feature(cfeature.COASTLINE,zorder=4,linewidth=2)
axes3.set_title('GAN Mean',fontsize=12)
axes3.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
cbbox3 = inset_axes(axes3, '2%', '85%', loc = 5)
cbbox3.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,     
    right=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False,
    labelright=False,
    labeltop=False) # labels along the bottom edge are off
cb3=figure.colorbar(GM,cax=cbbox3,cmap='RdBu')

axes3.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',s=75,zorder=99)


for craton in cratons:
        axes3.plot(craton[:,1],craton[:,0],color='black',linewidth=1)
        axes3.plot(craton[:,1],craton[:,0],color='black',linewidth=1)
#plt.colorbar(GM,ax=axes3)
axes4 = figure.add_axes([.972,0,.5,.5],projection= ccrs.PlateCarree())
GSD = axes4.pcolormesh(mlon, mlat, zi_std_fake,cmap=cmap_std,norm=colors.Normalize(vmin=min_std,vmax=max_std),zorder=0,transform = ccrs.PlateCarree())
axes4.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes4.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes4.add_feature(cfeature.COASTLINE,zorder=4,linewidth=2)
axes4.set_xlabel('GAN Standard Deviation',fontsize=12)
axes4.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
cbbox4 = inset_axes(axes4, '2%', '85%', loc = 5)
cbbox4.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,     
    right=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False,
    labelright=False,
    labeltop=False) # labels along the bottom edge are off
cb4=figure.colorbar(GSD,cax=cbbox4,cmap=cmap_std)
axes4.set_title('GAN STD',y=-.06)
axes4.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',s=75,zorder=99)

for craton in cratons:
        axes4.plot(craton[:,1],craton[:,0],color='black',linewidth=1)
        axes4.plot(craton[:,1],craton[:,0],color='black',linewidth=1)

axes5 = figure.add_axes([.977,.5,.2,.2],projection= ccrs.PlateCarree())
MCMCM = axes5.pcolormesh(mlon, mlat, zi_mean_real,cmap=plt.get_cmap('RdBu'),norm=colors.Normalize(vmin=min_mean,vmax=max_mean),zorder=0,transform = ccrs.PlateCarree())
axes5.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes5.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes5.add_feature(cfeature.COASTLINE,zorder=4)
axes5.set_title('MCMC Mean',fontsize=8)
axes5.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
#plt.colorbar(GSD,ax=axes4,shrink=.8)
axes5.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',zorder=99)

for craton in cratons:
        axes5.plot(craton[:,1],craton[:,0],color='black',linewidth=.6)
        axes5.plot(craton[:,1],craton[:,0],color='black',linewidth=.6)
        
axes6 = figure.add_axes([.977,0,.2,.2],projection= ccrs.PlateCarree())
MCMCSD = axes6.pcolormesh(mlon, mlat, zi_std_real,cmap=cmap_std,norm=colors.Normalize(vmin=min_std,vmax=max_std),zorder=0,transform = ccrs.PlateCarree())
axes6.add_feature(cfeature.OCEAN,zorder=1,color='white')
axes6.add_feature(cfeature.BORDERS,zorder=4,alpha=.5)
axes6.add_feature(cfeature.COASTLINE,linewidth=1,zorder=4)
axes6.set_title('MCMC STD',fontsize=8)
axes6.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False, 
    left=False,# ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
axes6.scatter(x[indx][1],y[indx][1],label=None,marker='*',color='yellow',zorder=99)

for craton in cratons:
        axes6.plot(craton[:,1],craton[:,0],color='black',linewidth=.6)
        axes6.plot(craton[:,1],craton[:,0],color='black',linewidth=.6)
