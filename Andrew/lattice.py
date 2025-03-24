import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
from gt import cascade
from gt import periodify  
from gt import adjify 
from multiprocessing import Pool

class lattice: 
    def __init__(self,numbers,flux: float,particles: int=1,template=None): 
        # TODO: impliment multiple particles
        self.particles = particles
        assert self.particles == 1 

        self.numbers  = numbers
        self.flux     = flux
        self.template = template
        self.unitcell = None

    def __adjify(self):
        if self.template is None:
            self.template = cascade(numbers)
        self.unitcell = adjify(self.template,self.flux)

    def bands(self,kvals: int):
        assert self.particles == 1
        self.__adjify()

        energies = []
        for k in range(kvals+1): 
            kmat = periodify(self.unitcell,2 * np.pi * k/kvals,1)
            energies.append(np.linalg.eigvalsh(kmat))
        return [2*np.pi*k/kvals for k in range(kvals+1)], np.array(energies)

def test():
    # TODO: write some tests lol
    return 0

def densify(ens,density_factor:int):
    """
        expect ens = list[(phi, energies)]
        does a linear interpolation that samples density_factor times
    """
    if density_factor==0:
        return ens
    ensn = []
    for x in range(len(ens) - 1):
        ensn.append(ens[x])
        for y in range(1,density_factor):
            tempphi = ens[x][0]*(y*1.0/density_factor) + ens[x + 1][0]*(density_factor - 1.0*y)/density_factor
            tempenergy = ens[x][1]*(y*1.0/density_factor) + ens[x + 1][1]*(density_factor - 1.0*y)/density_factor
            ensn.append((tempphi,tempenergy))
    ensn.append(ens[-1])
    return ensn

def flatify(ens): 
    x = []
    y = []
    for ev in ens: 
        for val in ev[1].flat:
            x.append(ev[0])
            y.append(val)

    return np.array(x), np.array(y)

def gen_labels(numbers):
    """
        produces horizontal ticks in a nice latex format for 
        the butterfly plots.. don't mess with this lol.
    """
    denom = np.product(numbers)
    labels = []
    for x in range(0,denom + 1):
        gcd = np.gcd(2 * x, denom)
        num = int(2*x/gcd)
        dnm = int(denom/gcd)
        labelstring = '0'
        if num == 1:
            if dnm == 1:
                labelstring = "$\pi$"
            else: 
                labelstring = r"$\frac{"+  r"\pi}" +"{" +str(dnm)+ "}$"
        elif num > 1: 
            if dnm == 1:
                labelstring = "$" + str(num) + r"\pi$"
            else:
                labelstring = r"$\frac{" + str(num)+ r"\pi}" +"{" +str(dnm)+ "}$"
        labels.append(labelstring)
    return labels

def colordiagram(numbers,phisteps,ksteps,density_factor=10):
    """
        numbers        : glued tree numbers
        phisteps       : number of steps to take in phi on the interval [0, 2 pi]
        ksteps         : number of steps to take in k, on the interval [0, 2 pi]
        density_factor : by what factor should the sampled points be made more dense
                         setting this to some number > 0 can lead to better looking plots
                         much cheaper to raise this than phisteps or ksteps

        calculates (flux, energy) pairs sampled over ksteps, with output formatted
        for colordiagramplot()
    """
    ens = []
    template = cascade(numbers)
    for phi in range(phisteps + 1):
        # initialize a lattice, and calculate band structure at fixed flux 
        # only calculates up to phi = pi since bands should be even about that point
        latt = lattice(numbers,phi*np.pi/phisteps,template=template)
        kvals, energies = latt.bands(ksteps)

        # densify the band structure in k, and then throw away k values
        # to densify in k, you have to get the values into the right format
        to_be_interpolated = [(kvals[x],energies[x]) for x in range(len(kvals))]
        _, energies = flatify(densify(to_be_interpolated,density_factor))
        ens.append((phi*np.pi/phisteps, energies))

    # densify values in flux, and then return after flattening into (phi, E) pairs
    ens = densify(ens,density_factor)
    x,y = flatify(ens)
    return np.concatenate((x,2*np.pi - x)), np.concatenate((y,y))



def topar(inp):
    template = inp[0]
    phi = inp[1]
    numbers = inp[2]
    ksteps = inp[3]
    density_factor = inp[4]
    latt = lattice(numbers,phi,template=template)
    kvals, energies = latt.bands(ksteps)
    to_be_interpolated = [(kvals[x],energies[x]) for x in range(len(kvals))]
    _, energies = flatify(densify(to_be_interpolated,density_factor))
    return (phi, energies)

def cdpar(numbers,phisteps,ksteps,density_factor=10,ncores=4):
    ens = []

    # append 
    template = cascade(numbers)
    fluxes = np.array(range(phisteps+1))*np.pi/phisteps
    inputs = [(template,flux,numbers,ksteps,density_factor) for flux in fluxes]
    pool=Pool(4)
    results = pool.map_async(topar,inputs)

    ens = results.get()
    
    # densify values in flux, and then return after flattening into (phi, E) pairs
    ens = densify(ens,density_factor)
    x,y = flatify(ens)
    return np.concatenate((x,2*np.pi - x)), np.concatenate((y,y))
    

    
def colordiagramplot(numbers,x,y,resolution=100):
    """
        numbers    : the numbers associated with a glued tree. needed for ticks
        x          : horizontal range of data. should be fluxes
        y          : vertical range of data. should be energies
        resolution : controls the number of hexagons used. default
                     is 100*(product of numbers)

        produces a butterfly plot by taking the output of colordiagram()
    """
    gs = resolution*np.product(numbers)

    # set domain and range of the plot
    xlim = 0, 2 * np.pi
    ylim = y.min(), y.max()

    # initialize an instance of a matplotlib figure
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(7, 6))

    # bin the (phi, E) pairs into hexagonal bins in the right range
    hb = ax.hexbin(x, y, gridsize=gs, bins='log', cmap='inferno',mincnt=1)
    ax.set(xlim=xlim, ylim=ylim)

    # set up plot labels and stuff
    ax.set_title("Density of states")
    ax.set_ylabel(r"$E$")
    ax.set_xlabel(r"$\Phi$")

    # ticks and tick labels
    ax.set_xticks(np.array(range(0, np.product(numbers)+1))*2 *np.pi/np.product(numbers))
    ax.set_xticklabels(gen_labels(numbers))
    cb = fig.colorbar(hb, ax=ax, label='counts')

    plt.savefig('color' + str(numbers) + 'bin.png',dpi=600)
    plt.close()


if __name__ == "__main__":
    print("Running")
    #numbers = [2,3,4,5,6]
    numbers = [2,3]
    x,y = cdpar(numbers,40,20,4)
    colordiagramplot(numbers,x,y,resolution=300)




