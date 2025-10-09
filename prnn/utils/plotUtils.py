from matplotlib import font_manager
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def setPlotDefaults():

    # Add arial font to the font manager
    font_dirs = ['../utils']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    print(font_files)
    if len(font_files) == 0:
        print("No fonts found")
    else:
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)

            if Path(font_file).name == "arial.ttf":
                plt.rcParams['font.family'] = 'Arial'

    # this changes but for those plots I did this
    plt.rc('xtick', labelsize=11) 
    plt.rc('ytick', labelsize=11) 
    plt.rc('axes', labelsize=11) 
    plt.rcParams['lines.linewidth'] = 2
    #plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.figsize'] = 7.5, 6.
    #plt.rcParams['figure.constrained_layout.w_pad'] = 3./72. 
    #plt.rcParams['figure.constrained_layout.h_pad'] = 3./72. 

    plt.rc('legend',fontsize='x-small')

def setNiceAxes():
        #TODO: add kwargs for manually setting bounds
        #ax = plt.gca()
        plt.xticks([plt.xticks()[0][1],plt.xticks()[0][-2]])
        plt.yticks([plt.yticks()[0][1],plt.yticks()[0][-2]])
        sns.despine(offset= 2, trim=True) 