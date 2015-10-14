
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
import numpy as np
import numpy.random as rand


class AnimatedSpins(object):
    def __init__(self, l, J, B, Ramp, simul_step): #, spins, singweight, spinup, simul_step):
        """
        Class for Ising 2D lattice animation
        constructor takes as arguments                                                              
        -- l: the lattice size                                                        
        -- J: coupling constant
        -- B: field
        -- Ramp: speed at which J is increased/decreased
        -- spins: the array containing the spin values
        -- 
        --                                               
        -- pos: numpy array of shape [numpoints, 3] containing the position coordinates           
        -- mom: numpy array of shape [numpoints, 3] containing the momentum coordinates
        -- part_update: function which calculates the new particle positions
           This function must output the updated arrays pos and mom                      
        -- args: the function part_update takes as arguments numpoints, box_len, pos, mom, *args 
        Example: anim_md.AnimatedScatter(n, box_len, pos, simulate, mom, n_t, dt)
        where 'simulate' is defined as
        def simulate(n, box_len, pos, mom, part_update, args):
          ...
          ...
          return pos, mom
        """
        self.l = l
        self.J = J
        self.B = B
        self.Ramp = Ramp
        self.spins = np.ones((self.l, self.l),dtype=int)
        self.spins[0,0] = -1
        self.singweight = np.exp(-2*self.J*np.arange(-4,5,2))
        self.spinup = np.exp(-2*self.B*np.arange(-1,2,2))
        self.xdata = [J]
        self.MagPoint = np.sum(self.spins)/(self.l)**2
        self.ydata = [self.MagPoint]
        self.iter = 0
        self.suspend = False
        
        self.stream = self.data_stream()
        self.simul_step = simul_step
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        plt.axis('off')
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[1, 25], height_ratios=[2,1])
        self.font = FontProperties()
        self.font.set_size(20)
        self.ax1=self.fig.add_subplot(self.gs[0])
        self.ax1.set_ylim(-1.0, 1.0)
        self.ax1.tick_params(axis='x', which='both', bottom='off', top='off',labelbottom='off')
        self.ax2=self.fig.add_subplot(self.gs[1])
        self.ax2.set_axis_off()
        self.ax3=self.fig.add_subplot(self.gs[3])
        self.ax3.set_ylim(-1.0,1.0)
        self.ax3.set_xlim(0.0, 1.0)
        self.ax3.set_xlabel ('J',fontproperties = self.font)
        self.ax3.set_ylabel('<s>', fontproperties = self.font)
        self.ax3.axhline(color='red')
        self.ax3.axvline(x=0.44,color='red')
        self.line, = self.ax3.plot(self.xdata, self.ydata, lw=2)
        plt.subplots_adjust(left=0.05, bottom=0.30)
        axcolor = 'lightgoldenrodyellow'
        self.axg = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg=axcolor)
        self.axt = plt.axes([0.15, 0.10, 0.65, 0.03], axisbg=axcolor)
        self.axr = plt.axes([0.15, 0.15, 0.65, 0.03], axisbg=axcolor)
        self.axl = plt.axes([0.15, 0.20, 0.65, 0.03], axisbg=axcolor)
        self.Jslider = Slider(self.axg, 'J', 0.1, 1.0, valinit=0.3)
        self.Bslider = Slider(self.axt, 'B', -0.5, 0.5, valinit=0.0)
        self.Rslider = Slider(self.axr, 'Ramp', -0.01, 0.01, valinit=0.0,  valfmt=u'%1.5f')
        self.Lslider = Slider(self.axl, 'L', 1, 200, valinit=self.l,  valfmt=u'%3.0f')
        self.im1 = self.ax1.add_patch(patches.Rectangle((0.0, 0.0), 1.0, 0.0,))
        self.im1.set_height(0.0)
        self.im2 = self.ax2.imshow(0.5*(self.spins+1), cmap='Greys',  interpolation='nearest')
        self.ani = animation.FuncAnimation(self.fig, self.display, 
                        self.simul_step(self.l, self.spins, self.singweight, self.spinup), interval=40)
        self.Jslider.on_changed(self.update)
        self.Bslider.on_changed(self.update)
        self.Rslider.on_changed(self.update)
        self.Lslider.on_changed(self.update)
        self.rar = plt.axes([0.78, 0.75, 0.15, 0.15], axisbg=axcolor)
        self.radio_reset = RadioButtons(self.rar, ('$T=0$', '$T=\infty$'))
        self.radio_reset.on_clicked(self.reset_spins)
        self.susb = plt.axes([0.78,0.65,0.15,0.05], axisbg=axcolor)
        self.suspend_button = Button(self.susb, "Click to suspend", color=axcolor, hovercolor='white')
        self.suspend_button.on_clicked(self.suspend_MC)
        self.resb = plt.axes([0.78,0.55,0.15,0.05], axisbg=axcolor)
        self.reset_mag_button = Button(self.resb, "Clear mag plot", color=axcolor, hovercolor='white')
        self.reset_mag_button.on_clicked(self.reset_magplot)

    def update(self,val):
        """ This function is called whenever a slider has been changed """
        self.J = self.Jslider.val
        self.B = self.Bslider.val
        self.Ramp = self.Rslider.val
        self.singweight = np.exp(-2*self.J*np.arange(-4,5,2))
        self.spinup = np.exp(-2*self.B*np.arange(-1,2,2))
        OldL = self.l
        self.l = np.int(self.Lslider.val/2)
        self.l = self.l * 2
        if OldL != self.l:
          self.reset_spins("T=0")
        self.fig.canvas.draw()
        self.Switch = True

    def reset_spins(self,label):
        """ Resets the spin to either the T=0 (ordered) or T=infty (disordered) configuration """
        if label=='$T=0$':
          self.spins = np.ones((self.l, self.l),dtype=int)
        else:
          self.spins = rand.random_integers(0, 1, (self.l,self.l))
          self.spins = 2*self.spins-1
 
    def suspend_MC(self,label):
        """ MC simulation is suspended / resumed """
        self.suspend = not(self.suspend)
        if self.suspend:
            self.suspend_button.label.set_text('Click to resume')
        else:
            self.suspend_button.label.set_text('Click to suspend')

    def reset_magplot(self,label):
        """ Resets the plot of the magnetization """
        self.xdata = []
        self.ydata = []
        
    def data_stream(self):
        """ 
           Calls spin update routine, copies it to the relevant section of the 'data' array which 
           is then yielded
        """
        self.spins = self.simul_step(self.l, self.spins, self.singweight, self.spinup)
        data = self.spins
        while True:
            if not(self.suspend):
                if (self.Ramp!=0):
                    self.J = self.J + self.Ramp
                    self.J = np.minimum(self.J, MaxJ)
                    self.J = np.maximum(self.J, MinJ)
                    self.singweight = np.exp(-2*self.J*np.arange(-4,5,2))
                    self.spinup = np.exp(-2*self.B*np.arange(-1,2,2))
                self.spins = self.simul_step(self.l, self.spins, self.singweight, self.spinup)
            data = self.spins
            yield data



    def display(self, i):# , simul_step, l, spins, J, B, Ramp, singweight, spinup, xdata, ydata):
        """ function called by FuncAnimation. Plots all graphs """    

        if (self.iter % 10) == 0:
          self.Jslider.set_val(self.J)
        self.iter = self.iter+1
        data = next(self.stream)
        self.im2.set_data((data+1)/2)
        self.im1.set_height(np.sum(data)/(self.l)**2)
        self.MagPoint = np.sum(data)/(self.l)**2
        self.xdata.append(self.J)
        self.ydata.append(self.MagPoint)
        self.line.set_data(self.xdata,self.ydata)
        plt.draw()
        return self.im1, self.im2, self.line, self.xdata, self.ydata


    def show(self):
        plt.show()


def init_simul():
    """ Generates the variables l, J, B, Ramp and initialises them """
    l=100
    J = 0.3
    B = 0.0
    Ramp = 0.0
    MaxJ = 1.0
    MinJ = 0.1
    return l, J, B, Ramp, MaxJ, MinJ





def simul_step(l, spins, singweight, spinup):
    """ Main function: performs a MC step. Red-black updates are used. Therefore some 
        calculations are redundant. Maybe it helps to avoid those. """

    weights = np.zeros((l,l),dtype=float)
    ranlat = np.ndarray((l, l),dtype=float)
    temp = np.ndarray((l, l),dtype=int)
    neigh_sum = np.roll(spins,1,0) + np.roll(spins, 1, 1) + np.roll(spins, -1, 1) + \
                np.roll(spins, -1, 0)
    energies = spins*neigh_sum
    weights = singweight[(energies+4)//2]
    weights = weights*spinup[(spins+1)//2]
    ranlat = rand.random_sample(size=(l,l))
    temp = np.where(ranlat < weights, -spins, spins)
    spins[0:l:2,0:l:2] = temp[0:l:2,0:l:2]
    spins[1:l:2,1:l:2] = temp[1:l:2,1:l:2]
    neigh_sum = np.roll(spins,1,0) + np.roll(spins, 1, 1) + np.roll(spins, -1, 1) + \
                np.roll(spins, -1, 0)
    energies = spins*neigh_sum
    weights = singweight[(energies+4)//2]
    weights = weights*spinup[(spins+1)//2]
    ranlat = rand.random_sample(size=(l,l))
    temp = np.where(ranlat < weights, -spins, spins)
    spins[1:l:2,0:l:2] = temp[1:l:2,0:l:2]
    spins[0:l:2,1:l:2] = temp[0:l:2,1:l:2]
    return spins



# Start of main program
l, J, B, Ramp, MaxJ, MinJ = init_simul()
Switch = False
# Start of the simulation and animation
a = AnimatedSpins(l, J, B, Ramp, simul_step)
a.show()


