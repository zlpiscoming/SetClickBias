import matplotlib.pyplot as plt
import numpy as np

def draw(y, save_dir='/home/leping_zhang/rl/torch1/plots/', another_dir='', bid=227):
    x = np.arange(len(y))
    y = np.array(y)
    
    color=['b','g','r','c','m','y','k','w']
    fig = plt.figure()
    ratio = 0.30
    y1 = [bid * (1+ratio)] * x.shape[0]
    y2 = [bid * (1-ratio)] * x.shape[0]
    y3 = [bid] * x.shape[0]
    plt.plot(x, y, color[0], label='Real Cost')
    plt.plot(x, y1, color[1], label='Upper Bound')
    plt.plot(x, y2, color[2], label='Lower Bound')
    plt.plot(x, y3, color[3], label='Bid')
    plt.xlabel('time')
    plt.ylabel('cost')
    plt.title('PID Controller')
    plt.legend()
    plt.show()


    fig.savefig(save_dir+"pid_result_with_total_PPO"+another_dir+".jpg")

def draw2(y, pid, wlc, save_dir='/home/leping_zhang/rl/torch1/plots/', another_dir='', cusname='', bid=227):
    x = np.arange(len(y))
    y, pid, wlc = list(y), list(pid), list(wlc)
    while len(pid) > len(y):
        pid.pop()
    while len(wlc) < len(y):
        wlc.append(wlc[-1])
    while len(wlc) > len(y):
        wlc.pop()
    while len(wlc) < len(y):
        wlc.append(wlc[-1])
    y = np.array(y)
    pid = np.array(pid)
    wlc = np.array(wlc)
    color=['b','g','r','c','m','y','k','w']
    fig = plt.figure()
    ratio = 0.40
    y1 = [bid * (1+ratio)] * x.shape[0]
    y2 = [bid * (1-ratio)] * x.shape[0]
    y3 = [bid] * x.shape[0]
    plt.plot(x, wlc, color[5], label='Waterlevel-based Controller')
    plt.plot(x, pid, color[4], label='PID Controller')
    plt.plot(x, y, color[0], label='Real Cost')
    plt.plot(x, y1, color[1], label='Upper Bound')
    plt.plot(x, y2, color[2], label='Lower Bound')
    plt.plot(x, y3, color[3], label='Bid')
    
    plt.xlabel('time')
    plt.ylabel('cost')
    plt.title('Control with Customer '+cusname)
    plt.legend()
    plt.show()


    fig.savefig(save_dir+"best-"+cusname+another_dir+".jpg")

def draw3(y, baselines, save_dir='/home/leping_zhang/saferl/safety-starter-agents/plots/', another_dir='', cusname='', bid=[]):
    x = np.arange(len(y))
    y = list(y)
    for k, v in baselines.items():
        v = list(v)
        while len(v) > len(y):
            v.pop()
        while len(v) < len(y):
            v.append(v[-1])
        v = np.array(v)
    
    y = np.array(y)
    color=['b','g','r','c','m','y','k','w']
    fig = plt.figure()
    ratio = 0.2
    bid = np.array(bid)
    y1 = bid * (1+ratio)
    y2 = bid * (1-ratio)
    y3 = bid
    plt.plot(x, y, color[0], label='Real Cost')
    plt.plot(x, y1, color[1], label='Upper Bound')
    plt.plot(x, y2, color[2], label='Lower Bound')
    plt.plot(x, y3, color[3], label='Bid')
    ptr = 4
    for k, v in baselines.items():
        plt.plot(x, v, color[ptr], label=k)
        ptr += 1
    plt.xlabel('time')
    plt.ylabel('cost')
    plt.title('Control with Customer '+cusname)
    plt.legend()
    plt.show()


    fig.savefig(save_dir+"best-"+cusname+another_dir+'without_fixed_bid'+".jpg")

def draw4(baselines, save_dir='/home/leping_zhang/saferl/safety-starter-agents/plots/', another_dir='', cusname='', bid=[]):
    
    lens = -1
    for k, v in baselines.items():
        v = list(v)
        if lens==-1:
            lens = len(v)
        else:
            lens = min(lens, len(v))

    for k, v in baselines.items():
        while len(v) > lens:
            v.pop()
        v = np.array(v)
    x = np.arange(lens)
    
    color=['b','g','r','c','m','y','k','w']
    fig = plt.figure()
    ratio = 0.2
    bid = np.array(bid)
    y1 = bid * (1+ratio)
    y2 = bid * (1-ratio)
    y3 = bid
    
    plt.plot(x, y1, color[0], label='Upper Bound')
    plt.plot(x, y2, color[1], label='Lower Bound')
    plt.plot(x, y3, color[2], label='Bid')
    ptr = 3
    for k, v in baselines.items():
        plt.plot(x, v, color[ptr], label=k)
        print('label is ', k)
        ptr += 1
    plt.xlabel('time')
    plt.ylabel('cost')
    plt.title('Customer '+cusname)
    plt.legend()
    plt.show()


    fig.savefig(save_dir+"best-"+cusname+another_dir+'without_fixed_bid'+".jpg")

def draw5(baselines, save_dir='/home/leping_zhang/saferl/safety-starter-agents/plots/', another_dir='', cusname='', bid=[]):

    lens = 45
    for k, v in baselines.items():
        v = list(v)
        if lens==-1:
            lens = len(v)
        else:
            lens = min(lens, len(v))

    for k, v in baselines.items():
        while len(v) > lens:
            v.pop()
        v = np.array(v)
    x = np.arange(lens)
    while len(bid) > lens:
        bid.pop()
    color=['b','g','r','c','m','y','k','w']
    fig = plt.figure()
    ratio = 0.2
    bid = np.array(bid)
    y1 = bid * (1+ratio)
    y2 = bid * (1-ratio)
    y3 = bid
    
    plt.plot(x, y1, color[0], label='Upper Bound')
    plt.plot(x, y2, color[1], label='Lower Bound')
    plt.plot(x, y3, color[2], label='Bid')
    ptr = 3
    for k, v in baselines.items():
        plt.plot(x, v, color[ptr], label=k)
        print('label is ', k)
        ptr += 1
    plt.xlabel('time')
    plt.ylabel('cost-per-conversion')
    #plt.title('Customer '+cusname)
    plt.legend()
    plt.show()

    fig.savefig(save_dir+another_dir+".svg", format='svg')
    #fig.savefig(save_dir+"best-"+cusname+another_dir+'without_fixed_bid'+".jpg")


def draw6(baselines1, baselines2, save_dir='/home/leping_zhang/saferl/safety-starter-agents/plots/', another_dir='', cusname='',
          bid=[]):
    # 画双纵轴的图片
    lens = 45
    for k, v in baselines1.items():
        v = list(v)
        if lens == -1:
            lens = len(v)
        else:
            lens = min(lens, len(v))

    for k, v in baselines2.items():
        v = list(v)
        if lens == -1:
            lens = len(v)
        else:
            lens = min(lens, len(v))

    for k, v in baselines1.items():
        while len(v) > lens:
            v.pop()
        v = np.array(v)

    for k, v in baselines2.items():
        while len(v) > lens:
            v.pop()
        v = np.array(v)

    fig, ax1 = plt.subplots()
    #plt.xticks(rotation=45)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    x = np.arange(lens)
    y3 = np.array(bid)[:lens]
    ax1.plot(x, y3, color[0], label='estimated conversion value')
    ptr = 1
    ax2 = ax1.twinx()

    while len(bid) > lens:
        bid.pop()

    for k, v in baselines1.items():
        ax1.plot(x, v, color[ptr], label=k)
        ptr += 1

    for k, v in baselines2.items():
        ax2.plot(x, v, color[ptr], label=k)
        ptr += 1
    plt.show()

    ax1.set_xlabel('time')
    ax1.set_ylabel('cost-per-conversion')
    ax2.set_ylabel("bid")
    # # plt.title('Customer '+cusname)
    fig.legend(loc="lower left", bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes)
    # plt.show()

    fig.savefig(save_dir + another_dir + ".jpg", format='jpg')
    # fig.savefig(save_dir+"best-"+cusname+another_dir+'without_fixed_bid'+".jpg")

if __name__ == '__main__':
    a = np.ones(3)
    b = np.ones(3) * 1.5
    b1 = {'zlp': a}
    b2 = {'plz': b}
    bid = [1, 2, 3]
    draw6(b1, b2, bid=bid, another_dir='test')