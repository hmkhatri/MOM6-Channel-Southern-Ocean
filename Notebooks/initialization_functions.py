"""

Graeme MacGilchrist
15/01/2020
gmacgilchrist@gmail.com

Set of functions to calculate input files for MOM6.


"""

import xarray as xr
import numpy as np
import scipy as sc

def calc_XYmeters(grid,center_x=True):
    '''Calculate the locations of each T point in [grid] in units of meters'''
    Xval = np.append(np.zeros(shape=(grid['lath'].size,1)),
                     grid['dxT'].cumsum('lonh').values,
                     axis=1)
    Yval = np.append(np.zeros(shape=(1,grid['lonh'].size)),
                     grid['dyT'].cumsum('lath').values,
                     axis=0)
    Xval = 0.5*(Xval[:,:-1]+Xval[:,1:])
    Yval = 0.5*(Yval[:-1,:]+Yval[1:,:])
    X = xr.DataArray(Xval,dims=['lath','lonh'],coords={'lath':grid['lath'],'lonh':grid['lonh']})
    Y = xr.DataArray(Yval,dims=['lath','lonh'],coords={'lath':grid['lath'],'lonh':grid['lonh']})

    if center_x:
        X = X - X.mean(dim='lonh')

    return X,Y

def calc_XYmeters_SOSE(grid,center_x=True):
    '''Calculate the locations of each T point in [grid] in units of meters'''
    Xval = np.append(np.zeros(shape=(grid['YC'].size,1)),
                     grid['dxG'].cumsum('XC').values,
                     axis=1)
    Yval = np.append(np.zeros(shape=(1,grid['XC'].size)),
                     grid['dyG'].cumsum('YC').values,
                     axis=0)
    Xval = 0.5*(Xval[:,:-1]+Xval[:,1:])
    Yval = 0.5*(Yval[:-1,:]+Yval[1:,:])
    X = xr.DataArray(Xval,coords=[("lath",grid['YC']), ("lonh",grid['XC'])])
    Y = xr.DataArray(Yval,coords=[("lath",grid['YC']), ("lonh",grid['XC'])])

    if center_x:
        X = X - X.mean(dim='lonh')

    return X,Y


def calc_vgrid(nk,max_depth,min_depth=0,thkcello_topcell=1,method='powerlaw'):
    '''Calculate the locations and thickness of grid cells for the vertical ocean grid'''
    z0 = min_depth + thkcello_topcell
    H = max_depth
    k = np.linspace(1,nk,num=nk)

    # Defining INTERFACE locations (i.e. grid cell interfaces)
    if method=='powerlaw':
        # Power law
        B = np.log(H/z0)/np.log(nk)
        zw = z0*k**B
    elif method=='uniform':
        zw = np.linspace(z0,H,nk)
    elif method=='exponential':
        zw = z0*np.exp(np.log(H/z0)*(k/(nk)))

    # Add the free surface, z*=0, as an interface (saved until this point as z0=0 messes with power law scaling)
    zw = np.append(0,zw)

    # Central point is THICKNESS location
    zt = (zw[1:] + zw[:-1]) / 2

    # Place in data arrays
    zw = xr.DataArray(zw,coords=[zw],dims=['NKp1'])
    zt = xr.DataArray(zt,coords=[zt],dims=['NK'])
    # Calculate thickness
    dz = zw.diff(dim='NKp1')
    dz = xr.DataArray(dz,coords=[zt],dims=['NK'])

    # Combine arrays to one dataset
    zw.name='zw'
    zt.name='zt'
    dz.name='dz'
    vgrid = xr.merge([zw,zt,dz])

    return vgrid

def def_sponge_dampingtimescale_north(Y,sponge_width,idampval):
    '''Define a sponge grid at the north of the domain based on horizontal grid shape.
    hgrid is the horizontal grid dataset
    sponge_width is the degrees of lat to damp over [must be a list, progressively decreasing in width]
    idampval is the inverse damping rate (in s-1) [must be a list] '''
    idamp = xr.zeros_like(Y)
    for i in range(len(sponge_width)):
        sponge_region = Y>Y.max(xr.ALL_DIMS)-sponge_width[i]
        idamp=idamp+xr.zeros_like(Y).where(~sponge_region,idampval[i])
    return idamp

def def_sponge_interfaceheight(vgrid,Y):
    '''Define a 3D array of layer interface heights (eta), to which sponge will relax.'''
    eta = xr.DataArray(-vgrid['zw'],coords=[vgrid['zw']],dims='NKp1')
    eta = eta*xr.ones_like(Y)
    return eta

def make_zeroinsponge(variable,Y,sponge_width_max):
    '''Set the given variable to zero in the sponge'''
    sponge_region = (Y>Y.max(xr.ALL_DIMS)-sponge_width_max)
    variable_new = variable.where(~sponge_region,0)
    return variable_new

def calc_distribution(coordinate,function,**kwargs):
    '''Calculate the distribution of a variable, based on a given coordinate
    e.g. linear surface distribution of temperature, where
             coordinate = Y
             val_at_mincoord = SST at south
             val_at_maxcoord = SST at north
             function = 'linear'
        Independent variable required for functions can be passed at the end of the function
    '''
    if function=='linear':
        A = (kwargs["val_at_maxcoord"]-kwargs["val_at_mincoord"])/(coordinate.max(xr.ALL_DIMS)-coordinate.min(xr.ALL_DIMS))
        B = kwargs["val_at_mincoord"] - coordinate.min(xr.ALL_DIMS)*A
        distribution = A*coordinate+B

    if function=='exponential':
        distribution = kwargs["val_at_maxcoord"]*np.exp(coordinate/kwargs["efolding"])

    if function=='gaussian':
        distribution = np.exp(-np.power(coordinate - kwargs["center"], 2.) / (2 * np.power(kwargs["width"], 2.)))

    if function=='uniform':
        distribution = kwargs["uniform_value"]*xr.ones_like(coordinate)

    #if function=='tan_hyperbolic':

    return distribution

def calc_forcing_zonaluniform(Y,function,**kwargs):
    '''Define zonally uniform forcing with a particular shape defined by function'''
    
    if function=='doublesinusoid_gen':
		# General function form, f(x) = sin(x + sin(a*x)*b/2)**c or sin(x + sin(d*x)*e/2)**f
		# Recommended values  a, d = 1, 2; b, e = [-1, 1]; c, f = 1, 2
		
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = ((kwargs["max_north"]*(np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width + 
        np.sin(np.pi*kwargs["a"]*(Y-kwargs["northsouth_boundary"])/north_width)*kwargs["b"]/2.))**kwargs["c"]).where(condition_north,0))

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = ((-kwargs["max_south"]*(np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width +
        np.sin(np.pi*kwargs["d"]*(Y-kwargs["south_zeroregion"])/south_width)*kwargs["e"]/2.))**kwargs["f"]).where(condition_south,0) + forcing)
        
    if function=='doublesinusoid_squared':
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = (kwargs["max_north"]*np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width)**2).where(condition_north,0)

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = (-kwargs["max_south"]*np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width)**2).where(condition_south,0) + forcing

    if function=='doublesinusoid_squared_skew':
		# Skew function, f(x) = sin(x + sin(x)*alpha/2)**2
		
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = ((kwargs["max_north"]*(np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width + 
        np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width)*kwargs["alpha"]/2.))**2).where(condition_north,0))

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = (-kwargs["max_south"]*np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width)**2).where(condition_south,0) + forcing
        
    if function=='doublesinusoid_mix_skew':
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = ((kwargs["max_north"]*(np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width + 
        np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width)*kwargs["alpha"]/2.))).where(condition_north,0))

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = (-kwargs["max_south"]*np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width)**2).where(condition_south,0) + forcing
        
    if function=='doublesinusoid_skew':
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = ((kwargs["max_north"]*(np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width + 
        np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width)*kwargs["alpha"]/2.))).where(condition_north,0))

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = (-kwargs["max_south"]*np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width)).where(condition_south,0) + forcing

    if function=='doublesinusoid':
        domain_width = Y.max(xr.ALL_DIMS)-Y.min(xr.ALL_DIMS)
        north_width = domain_width-kwargs["sponge_width_max"]-kwargs["northsouth_boundary"]
        south_width = kwargs["northsouth_boundary"]-kwargs["south_zeroregion"]

        condition_north = (Y>=kwargs["northsouth_boundary"]) & (Y<=domain_width-kwargs["sponge_width_max"])
        forcing = (kwargs["max_north"]*np.sin(np.pi*(Y-kwargs["northsouth_boundary"])/north_width)).where(condition_north,0)

        condition_south = (Y>=kwargs["south_zeroregion"]) & (Y<=kwargs["northsouth_boundary"])
        forcing = (-kwargs["max_south"]*np.sin(np.pi*(Y-kwargs["south_zeroregion"])/south_width)).where(condition_south,0) + forcing

    if function=='uniform':
        forcing = kwargs["uniform_value"]*xr.ones_like(Y)

    return forcing
