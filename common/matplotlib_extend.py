import matplotlib
from cycler import cycler

def set_matplotlib_style(mode=None):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams["text.usetex"] = True

    if(mode=='slide'):
        matplotlib.rcParams['axes.prop_cycle']=  cycler(color= ['#66FF99','#FFCC99','#FFFF99','#FF99FF'])
        for name in matplotlib.rcParams:
            if matplotlib.rcParams[name]=='black':
                matplotlib.rcParams[name] ='#B9CAFF'
            if matplotlib.rcParams[name]=='white':
                matplotlib.rcParams[name] ='#0C0C3A'

def plot_ma(series,lables,title,n):
    fig, ax = matplotlib.pyplot.subplots()
    _colors = ['#800080', '#008000', '#0000ff', 'orange', 'red', 'yellow', 'black']
    _c_idx = 0
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    for s,label in zip(series,lables):
        x=range(len(s))
        y_std = s.rolling(n).std() * 100
        y_mean = s.rolling(n).mean() * 100
        ax.plot(y_mean,label=label, color=_colors[_c_idx])
        ax.set_title(title)
        ax.fill_between(x,y_mean-y_std, y_mean+y_std, alpha=0.2, color=_colors[_c_idx])
        _c_idx+=1 if _c_idx < len(_colors)-1 else 0
        
    matplotlib.pyplot.legend()            