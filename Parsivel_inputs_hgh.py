# Inputs for parsival scripts for Highland High School
# Michael Wasserstein and Dave Kingsmill
# To visualized parsivel data, this script is needed
# 11/20/2022

import numpy as np

############### Diameter ##############

# particle diameter in mm
parsivel_D = np.array([0.062,0.187,0.312,0.437,0.562,0.687,0.812,0.937,1.062,
1.1870,1.3750,1.625,1.875,2.125,2.375,2.75,3.25,3.75,
4.25,4.75,5.50,6.50,7.50,8.50,9.50,11,13,15,17,19,21.5,24.5])

# Range for diameter
parsivel_deltaD = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125,
                            0.250,0.250,0.250,0.250,0.250,0.500,0.500,0.500,0.500,0.500,
                            1,1,1,1,1,2,2,2,2,2,3,3])


############## Fall Speed ##################

# Velocity in m/s
parsivel_V = np.array([0.050,0.15,0.25,0.35,0.45,0.55,0.65,0.75,
0.85,0.95,1.1,1.3,1.5,1.7,1.9,2.2,2.6,3,3.4,
3.8,4.4,5.2,6,6.8,7.6,8.8,10.4,12,13.6,15.2,
17.6,20.8])

# Range for velocity
parsivel_deltaV = np.array([0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,
                            0.10,0.10,0.20,0.20,0.20,0.20,0.20,0.40,
                            0.40,0.40,0.40,0.40,0.80,0.80,0.80,0.80,
                            0.80,1.6,1.6,1.6,1.6,1.6,3.2,3.2])

palt=1363 # site altitude in m

Vaggr = 0.8 * (parsivel_D**0.16)   # Locatelli and Hobbs 1974 AGGR of dendrites
Vgr = 1.6 * (parsivel_D**0.46)     # Locatelli and Hobbs 1974 lump graupel

dm = parsivel_D/(10**3);

### Calculate eta
KelvTemp = 273.15;    ## Convert to Kelvin
eta = 1.72e-5*(393/(KelvTemp+120)*(KelvTemp/273) ** 1.5);
#
### Calculate rho (units = kg/m^3)
rho=1.225*np.exp(-1e-4*palt);
#rho = 800e2/(287*273);
#
### Calculate X (dimensionless)
### Make sure diameters are in meters
g = 9.81; ##  m/s^2
rhow = 1000;  ## density of water (kg/m^3)
X = ((dm**3)*(4/3)*g*rhow*rho)/(eta**2);
#
### Calculate Reynolds number
a20 = -0.312611e1;
a21 = 0.101338e1;
a22 = -0.191182e-1;
a23 = 0;

lnRe = a20+a21*(np.log(X))+a22*(np.log(X))** 2+a23*(np.log(X))**3;
lnRe[np.isnan(lnRe)]=0;

Re = np.exp(lnRe);

### Calculate V (m/s)
Vbp = ((Re)*eta)/(dm*rho);

# Apply Foote and du Toit fall speed density correction
Vaggr_corrected = Vaggr*(1.275/rho)**0.4;
Vgr_corrected = Vgr*(1.275/rho)**0.4;
Vbp_corrected = Vbp*(1.275/rho)**0.4;

#dimensions of the laser beam
lx213b=30   # beam width in mm
lx213l=180   # length between arms in mm