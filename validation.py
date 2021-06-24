import os, getopt, sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as mdates


epoch_target = 500
log_dir = './output/train_out_20210623_192855/_2'
file_name = 'weights_154_w1.14.csv'
returns_file_name = '../df_ret_val.csv'

STOCK_CHARGE_RATIO = 0.02


def plot_maa(series, lables,title, n):
    fig, ax = plt.subplots()
    _colors = ['#800080', '#e68e9d', '#0000ff']
    _c_idx = 0
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for s,label in zip(series,lables):
    
        ax.plot(s * 100, label=label, color=_colors[_c_idx])
        ax.set_title(title)
        _c_idx+=1 if _c_idx < len(_colors)-1 else 0
        
    plt.legend()


def post_epoch_func(self = None):
    n = int(epoch_target / 10)  #  for rollilog_dirng
    csv_file = os.path.join(log_dir, file_name)
    returns_file = os.path.join(log_dir, returns_file_name)
    df_weights = pd.read_csv(csv_file)
    df_returns = pd.read_csv(returns_file)

    wealth_now = 1
    wealth_charged = 1
    wealth_list = []
    wealth_charged_fee_list = []
    date_list = []

    last_one_weight_percentages = [0,0,0,0,0,0,0,0,0,0,0]

    for idx in range(len(df_weights['Date'])):

        _weights = df_weights.iloc[idx]
        _returns = df_returns.iloc[idx]
        _weight_percentages = []
        
        _length = len(_weights)
        _total_row_reward = 0
        _sum_val = 0
        _sum_ret = 0
        for _sub_idx in range(_length):
            _wei = _weights[_sub_idx]
            _ret = _returns[_sub_idx]
            _val = 0
            if isinstance(_wei, float):
                _val = float(_wei)
            elif '%' in _wei:
                _val = float(_wei.strip('%')) / 100
                if _val and _val > 0:
                    _sum_ret += _ret
                    _sum_val += _val
                    _reward = ((1+_ret) * _val) - _val
                    # print('_reward: ', _reward)
                    _total_row_reward += _reward
            
                
            
            _weight_percentages.append(_val)
            # print('_sum_ret: ', _sum_ret)
        # print('_total_row_reward: ', _total_row_reward)
        wealth_now = wealth_now * (1 + _total_row_reward)

        row_total_change_weight = 0
        
        for _i in range(len(last_one_weight_percentages)):
            _befire_weight = last_one_weight_percentages[_i]
            _now_weight = _weight_percentages[_i]
            change_weight = abs(_now_weight - _befire_weight)
            row_total_change_weight += change_weight

        last_one_weight_percentages = _weight_percentages
        charge_fee = row_total_change_weight * STOCK_CHARGE_RATIO
        wealth_charged = wealth_charged * (1 + _total_row_reward)
        wealth_charged = wealth_charged - (wealth_charged * charge_fee)
        wealth_charged_fee_list.append(wealth_charged)

        wealth_list.append(wealth_now)
        date_list.append(df_weights['Date'][idx])
        
    print('wealth_list: ', wealth_list)
    print('date_list: ', date_list)
    print('wealth_charged_fee_list: ', wealth_charged_fee_list)
    
    # series = map(lambda s: df[f'{s}/env_infos/final/{kpi} Mean'], srcs)
    plot_maa(
        series=[pd.DataFrame(wealth_charged_fee_list), pd.DataFrame(wealth_list)],
        lables=['Wealth Charged', 'Wealth'], title='[ Evalution Wealth ]',
        n=1)
    plt.savefig(os.path.join(log_dir, 'check_back_weights_wealth.png'))
    plt.close()


def main(argv):
    file_name = None
    returns_file_name = None
    log_dir = None
    try:
        opts, args = getopt.getopt(argv,"d:i:r:",["dir=", "ifile=","rfile="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            file_name = arg
        elif opt in ("-r", "--rfile"):
            returns_file_name = arg
        elif opt in ("-d", "--dir"):
            log_dir = arg

    return log_dir, file_name, returns_file_name

if __name__ == '__main__':

    _dir, _file, _ret_file = main(sys.argv[1:])
    if _file:
        file_name = os.path.basename(_file)
        log_dir = os.path.dirname(_file)
    post_epoch_func()