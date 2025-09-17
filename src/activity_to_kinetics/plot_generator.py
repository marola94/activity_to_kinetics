import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
from units import ureg
import activity_to_kinetics.src.activity_to_kinetics.kinetic_model_fitting as kmf
from matplotlib.lines import Line2D

class ScalarFormatterClass(mticker.ScalarFormatter):
    def _set_format(self):
        self.format = "%1.2f"



# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(math.floor(math.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

# Function to convert and format units
def format_unit(unit):
    unit_str = str(unit)
    if '1/' in unit_str:
        parts = unit_str.split('1/')
        return f"{parts[1]}$^{{-1}}$"
    elif '/' in unit_str:
        parts = unit_str.split('/')
        return f"{parts[0]} {parts[1]}$^{{-1}}$"
    else:
        return unit_str


def activity_plot(x, y, well, use_filter: bool = False, **kwargs):

    WELL = well
    TITLE =  kwargs.get('Experiment title')
    RATE = kwargs.get('Initial rate')
    INTERCEPT = kwargs.get('Intercept')
    FILTER = kwargs.get('Filtered curve')

    print(f'\n Plotting activity of {WELL}.')

    labels=[]

    # Plotting the raw data
    fig1, ax1 = plt.subplots(num = 1, clear = True)
    plot1_color = 'black'
    ax1.plot(x, y, '.', markersize = 3, color = plot1_color)
    if y.units == 'AU':
        ax1.set_ylabel(f'Absorbance [AU]')

    else:
        ax1.set_ylabel(f'Concentration [{y.units}]')
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))  

    if TITLE is not None:
        ax1.set_title(f'{TITLE}', fontweight = 'bold')
    else:
        ax1.set_title(f'{WELL}', fontweight = 'bold')
       
    # Creating plot including filtered data

    if FILTER is not None and np.any(FILTER > 0) and use_filter:
        fig2, ax2 = plt.subplots()
        ax2.plot(x, y, '.', markersize = 3, color = plot1_color)
        ax2.plot(x, FILTER, '-', color = 'forestgreen')
        if y.units == 'AU':
            ax2.set_ylabel(f'Absorbance [AU]')

        else:
            ax2.set_ylabel(f'Concentration [{y.units}]')

        ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))

        if TITLE is not None:
            ax2.set_title(f'{TITLE}', fontweight = 'bold')
        else:
            ax2.set_title(f'{WELL}', fontweight = 'bold')
    
    # ---------------- Y-AXIS SETTINGS ----------------
    # Setting the y-axis limits
    y_max = max(y.magnitude)
    y_min = min(y.magnitude)
    ax1.margins(y=1)    
    ax1.set_ylim(bottom=0.0)

    if FILTER is not None and np.any(FILTER > 0) and use_filter:
        ax2.margins(y=1) 
        ax2.set_ylim(bottom=0.0)

    # -------------------------------------------------

    #---------------- X-AXIS SETTINGS ----------------#
    # Setting the x-axis limits
    x_max = max(x.magnitude)
    x_min = min(x.magnitude)
    
    # Set the number of desired ticks
    num_ticks = 6

    # Calculate tick positions
    tick_interval = (x_max - x_min) / (num_ticks)
    tick_positions = np.arange(0, x_max + tick_interval, tick_interval)
    
    #Setting the tick position
    if tick_positions[-1] > x_max:
        tick_positions = tick_positions[:-1] # Exclude ticks above the maximum time stamp
    
    # Set ticks and labels on x-axis
    ax1.set_xticks(tick_positions)
    ax1.set_xlabel('Time'+f' [{x.units}]')
    
    if FILTER is not None and np.any(FILTER > 0) and use_filter:    
        ax2.set_xticks(tick_positions)
        ax2.set_xlabel('Time'+f' [{x.units}]')

    # -------------------------------------------------

    # ------- Plotting rate and appending label -------
    if RATE is not None and INTERCEPT is not None:

        # Formatting the number to scientific notation
        if RATE.magnitude < 0.1:
            rate_number = sci_notation(RATE.magnitude,2)
            rate_unit = RATE.units

        else:
            rate_number = np.round(RATE.magnitude,2)
            rate_unit = RATE.units

        # Create label for the plot
        labels.append(r'$\bf{Initial}$ '+ r'$\bf{rate:}$ ' + f'{rate_number} ' + f'{rate_unit}' )

        # Plotting the rate as a stright line 
        xmin, xmax = ax1.get_xlim() # current x-axis limits
        ymin, ymax = ax1.get_ylim() # current y-axis limits

        # Rescale rate to the corresponding unit of the plotted times
        # Retrieves concentration part of the rate
        c_rate_unit = (RATE*(1*RATE._REGISTRY.parse_units(f'{x.units}'))).to_reduced_units().units
        if c_rate_unit == RATE._REGISTRY.dimensionless: # If the concentration unit is absorbance
            c_rate_unit = ureg.parse_units("AU") 
        
        a = RATE.to(f'{c_rate_unit}/{x.units}').magnitude
        b = INTERCEPT.magnitude
        
        x0 = 0                   # start at the left edge (time > 0)
        dx = 0.5 * (xmax - xmin)   # target horizontal length = 50% of x-axis width

        # Also cap the vertical span to ≤ 50% of y-axis height (handles steep slopes)
        if RATE.magnitude != 0:
            dx = min(dx, 0.5 * (ymax - ymin) / abs(RATE.magnitude))

        x1 = x0 + dx
        y0 = a * x0 + b
        y1 = a * x1 + b

        ax1.plot([x0, x1], [y0, y1], color="r", linewidth=2)

        #ax1.axline(xy1 = (0, INTERCEPT.magnitude), slope = RATE.magnitude, color='r')

        if FILTER is not None and np.any(FILTER > 0) and use_filter:

            #ax2.axline(xy1 = (0, INTERCEPT.magnitude), slope = RATE.magnitude, color='r')
            ax2.plot([x0, x1], [y0, y1], color="r", linewidth=2)

        else:
            pass

        # ------------------ ADD LEGEND ------------------#
    if len(labels) > 0:
        
        legend_elements = []
        for i in range(0,len(labels)):
            legend_elements.append(Line2D([0], [0], linestyle = 'None', label=f'{labels[i]}'))
        #legend_elements = [Line2D([0], [0], linestyle = 'None', label=f'{labels[0]}'),
        #                    Line2D([0], [0], linestyle = 'None', label=f'{labels[1]}'),
        #                    Line2D([0], [0], linestyle = 'None', label=f'{labels[2]}')]
        
        ax1.legend(handles=legend_elements, loc='upper left', frameon=False)

        if FILTER is not None and np.any(FILTER > 0) and use_filter:
            legend_elements = []
            for i in range(0,len(labels)):
                legend_elements.append(Line2D([0], [0], linestyle = 'None', label=f'{labels[i]}'))
            #legend_elements = [Line2D([0], [0], linestyle = 'None', label=f'{labels[0]}'),
             #                   Line2D([0], [0], linestyle = 'None', label=f'{labels[1]}'),
              #                  Line2D([0], [0], linestyle = 'None', label=f'{labels[2]}')]

            ax2.legend(handles=legend_elements, loc='upper left', frameon = False)

    # Adjusting size of plot
    fig1.tight_layout()
  
    if FILTER is not None and np.any(FILTER > 0) and use_filter:
        fig2.tight_layout()

    return fig1, fig2 if 'fig2' in locals() else None


def kinetic_plot(model, filename):

    
    labels=[]
    s = model.get('s')
    r = model.get('rates')
    V_MAX = model.get('Vmax')
    std_VMAX = model.get('std. Vmax')
    KM = model.get('Km')
    std_KM = model.get('std. Km')
    KI = model.get('Ki')
    std_KI = model.get('std. Ki')
    KCAT = model.get('kcat')
    std_KCAT = model.get('std. kcat')
    enzyme_conc = model.get('Enzyme concentration')
    TITLE = model.get('Title')
    model_number = model.get('Model No.')
    model_type = model.get('Model type')

    fig1, ax1 = plt.subplots(num = 1, clear = True)
    

    s_fit = np.linspace(0, max(s.magnitude), 1000)

    if model_type == 'Michaelis-Menten':
        r_fit = kmf.Michaelis_Menten(s_fit, V_MAX.magnitude, KM.magnitude)
        #ax1.plot(s_fit, kmf.Michaelis_Menten(s_fit, V_MAX.magnitude, KM.magnitude), color = 'black')
        parameters = {'$V_{max}$': V_MAX,
                  '$std. V_{max}$': std_VMAX,
                  '$K_{M}$': KM,
                  '$std. K_{M}$':std_KM,
                  '$k_{cat}$': KCAT,
                  '$std. k_{cat}$': std_KCAT
                    }

        
    elif model_type == 'Substrate Inhibition':
        r_fit = kmf.Substrate_Inhibition(s_fit, V_MAX.magnitude, KM.magnitude, KI.magnitude)
        #ax1.plot(s_fit, kmf.Substrate_Inhibition(s_fit, V_MAX.magnitude, KM.magnitude, KI.magnitude), color = 'black')
        parameters = {'$V_{max}$': V_MAX,
                  '$std. V_{max}$': std_VMAX,
                  '$K_{M}$': KM,
                  '$std. K_{M}$':std_KM,
                  '$K_{I}$': KI,
                  '$std. K_{I}$':std_KI,
                  '$k_{cat}$': KCAT,
                  '$std. k_{cat}$': std_KCAT
                    }

    labels = figure_labels(parameters)

    s_fit = ureg.Quantity(s_fit, s.units)
    r_fit = ureg.Quantity(r_fit, r.units)

    # Plotting the raw data
    
    ax1.plot(s_fit, r_fit, color = 'black')
    ax1.scatter(s,r, edgecolors= 'crimson', alpha=0.8, c='white')

    # Axes
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,9))

    # y-axis
    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0,0))
    ax1.yaxis.set_major_formatter(yScalarFormatter)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9))  
    ax1.set_ylabel('Rate ' f'[{format_unit(r.units)}]')

    # x-axis
    # Get compact version of highest substrate concenttration
    s_max = max(s_fit).to_compact()

    # Format x-ticks to mM
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: '{0:g}'.format((x * s_fit.units).to(s_max.units).magnitude))
        )
    ax1.set_xlabel(f'Substrate concentration [{s_max.units}]')

    # ------------------ ADD LEGEND ------------------#
    if len(labels) > 0:    
        legend_elements = []
        for i in range(0,len(labels)):
            legend_elements.append(Line2D([0], [0], linestyle = 'None', label=f'{labels[i]}'))
        
        ax1.legend(handles=legend_elements, loc='lower right', frameon=False)
    if TITLE:
        ax1.set_title(f'{TITLE}', fontweight = 'bold')
        fig1.suptitle(f'{model_type}')
    else:
        ax1.set_title(f'{model_number}', fontweight = 'bold')
        fig1.suptitle(f'{model_type}')

    fig1.tight_layout()

    return fig1 

def figure_labels(parameters):
    labels = []

    for key, value in parameters.items():
        if key.startswith('$std.'):
            continue

        elif key == '$K_{I}$':
            if value.magnitude > 10**3:
                    value = value.to('mM')
                    std = parameters['$std. '+f'{key[1:]}'].to('mM')
                    value_number = np.round(value.magnitude,2)

                    if value.magnitude > 10**3:
                        value = value.to('M')
                        std = std.to('M')
                        value_number = np.round(value.magnitude,2)

                        if value.magnitude > 10**3:
                            value_number = sci_notation(value.magnitude,2)
                        
                    if std.magnitude < 0.01:
                        if std.magnitude == 0.0:
                            std_number = np.round(std.magnitude,2)
                        else:
                            std_number = sci_notation(std.magnitude,2)

                    else:
                        std_number = np.round(std.magnitude,2)
            
            else:
                if value.magnitude < 0.01:
                    value_number = sci_notation(value.magnitude,2)
                else:
                    value_number = np.round(value.magnitude,2)  
                
                std = parameters['$std. '+f'{key[1:]}']
                if std.magnitude < 0.01:
                    if std.magnitude == 0.0:
                        std_number = np.round(std.magnitude,2)
                    else:
                        std_number = sci_notation(std.magnitude,2)

                else:
                    std_number = np.round(std.magnitude,2)

        else:
            if value.magnitude < 0.01:
                value_number = sci_notation(value.magnitude,2)

            else:
                value_number = np.round(value.magnitude,2)

            std = parameters['$std. '+f'{key[1:]}']
            if std.magnitude < 0.01:
                if std.magnitude == 0.0:
                    std_number = np.round(std.magnitude,2)
                else:
                    std_number = sci_notation(std.magnitude,2)

            else:
                std_number = np.round(std.magnitude,2)

        labels.append(f'{key}: ' + f'{value_number} ' +u'\u00B1 '+ f'{std_number} ' + f'{format_unit(value.units)}')
    
    return labels

