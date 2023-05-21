import numpy as np
from gym import Env, spaces, wrappers
import safe_rl.pg.np_tools as np_tools

class PID():
    def __init__(self, kp=10, ki=0.01, kd=0.1):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.now_val = 0
        self.now_err = 0
        self.last_err = 0
        self.last_last_err = 0
        self.change_val = 0
        self.max_size = 100000
        self.Ival = [0] * self.max_size
        self.Iptr = 0

    def reset(self):
        self.now_val = 0
        self.now_err = 0
        self.last_err = 0
        self.last_last_err = 0
        self.change_val = 0
        self.max_size = 100000
        self.Ival = [0] * self.max_size
        self.Iptr = 0
 
    def cmd_pid(self, now_val, exp_val):
        self.now_val = now_val
        
        self.last_last_err = self.last_err
        
        self.now_err = exp_val - self.now_val
        self.Ival[self.Iptr] = self.now_err
        self.Iptr = (self.Iptr+1) % self.max_size
        Pval, Ival, Dval = self.now_err, sum(self.Ival), (self.now_err - self.last_err)
        self.change_val = self.KP * self.now_err + self.KI * sum(self.Ival) + self.KD * (self.now_err - self.last_err)
        #if change_val < 0:
        self.last_err = self.now_err
        self.now_val += self.change_val
        self.now_val = max(1e-3, self.now_val)
        #print('pid summary', sum(self.Ival), self.now_err, self.change_val)
        
        return Pval, Ival, Dval, self.now_val

class Temp:
    def __init__(self, timestep, price, log):
        self.timestep = timestep
        self.price = int(price)
        self.log = log
    
    def show(self):
        print(self.timestep, self.price, self.log)

def add(a, val=1):
    b = str(int(a)+val)
    while len(b) < len(a):
        b = '0'+b
    return b
    
class WLC():
    def __init__(self, lamb=10):
        self.lamb = lamb
        self.last = 0
        self.now_val = 0

    def cmd_pid(self, now_val, exp_val):
        self.now_val = now_val
        dis = exp_val - now_val
        self.change_val = self.lamb * dis + 0.3*self.now_val
        self.now_val += self.change_val
        self.last = self.change_val
        return dis, self.now_val
    
    def reset(self):
        self.last = 0
        self.now_val = 0
        
class ValueController:

    def __init__(self, basedir = '/home/leping_zhang/rl/torch1/data/', ad = '3358', isdiscrete=False, ratio=0.2, pid = WLC(), episode_len=3, bid=227, dat='0606', w=False, otherwarm=False):
        super(ValueController, self).__init__()
        '''
        Params:
        isdiscrete bool 动作空间是否是离散的 0 -> 2 0.01, 0.02, ..., 2
        data list 数据含有两种 log = 1表示回本, log = 2表示花钱
        pid PID pid控制器, 算是一个baseline, 并且也作为state
        ratio float 赔付比率
        bid float 商家的出价
        real_val float 目前的实际价格
        ptr int 现在到data哪里了
        max_size int data的长度
        total_imp int 总的展示次数
        total_cost int 现在一共花多少钱了
        episode_len int 状态间的距离
        step_size int 一共有多少step
        now_step int 现在到第几个step了
        ''' 
        def create(dir, log=1):
            dir = basedir + dir
            f1 = open(dir, 'r')
            rs = f1.readlines()
            datas = []
            for r in rs:
                data = r.split('\t')
                if data[-2] != ad:
                    continue 
                if log == 1:
                    p = Temp(data[1], int(data[-5]), log)
                else:
                    p = Temp(data[1], -1, log)
                # print(p.price)
                # break
                datas.append(p)
            f1.close()
            return datas
        
        self.last = 0
        sp = 0
        if w:
            sp = 150
        else:
            sp = 80
        self.action_space = spaces.Discrete(sp) # 动作空间
        self.observation_space =  spaces.Box(low=-np.inf, high=np.inf, shape=(4, )) # 状态空间
        # Define an action space range from 0 to 4
        
        # clicks = create('clk.20130610.txt', 1) + create('clk.20130611.txt', 1) + create('clk.20130612.txt', 1)
        # trains =  create('imp.20130610.txt', 2) + create('imp.20130611.txt', 2) + create('imp.20130612.txt', 2)
        # clicks = create('conv.20130610.txt', 1) + create('conv.20130611.txt', 1) + create('conv.20130612.txt', 1)
        # trains =  create('clk.20130610.txt', 2) + create('clk.20130611.txt', 2) + create('clk.20130612.txt', 2)

        clicks = create('conv.2013'+add(dat, 0)+'.txt', 1) + create('conv.2013'+add(dat, 1)+'.txt', 1) + create('conv.2013'+add(dat, 2)+'.txt', 1)
        trains =  create('clk.2013'+add(dat, 0)+'.txt', 2) + create('clk.2013'+add(dat, 1)+'.txt', 2) + create('clk.2013'+add(dat, 2)+'.txt', 2)
        self.data = clicks + trains
        self.isdiscrete = isdiscrete    
        self.pid = pid
        self.ratio = ratio
        self.bid = bid
        self.real_val = 0
        self.ptr = 0
        self.max_size = len(self.data)
        self.total_cost = 0
        self.total_imp = 0
        self.total_click = 0
        self.data.sort(key=lambda s: s.timestep)
        self.w = w
        self.otherwram = otherwarm # True表示该广告基本没有转化，需要特殊的方式重启
        self.windows = []
        self.windowslen = 30
        self.windowsptr = 0
        self.trackedtime = self.max_size*0.8
        #print(len(clicks), len(trains))
        
        # while self.ptr < self.max_size and self.data[self.ptr].log == 2:
        #     self.total_imp += 1
        #     self.ptr += 1
        #     if self.data[self.ptr].log == 1:
        #self.total_click = 1
        self.total_click = 0
        self.total_cost = 0
        self.num = 4 # 多少个click 重启 
        self.init_ptr = 0
        if self.otherwram:
            self.num = 0
        while self.ptr < self.max_size and self.init_ptr < self.num:
            self.total_imp += 1
            self.ptr += 1
            if self.data[self.ptr].log == 1:
                self.init_ptr += 1
                self.total_click += 1
            
        self.total_cost = self.num * self.bid
        assert self.ptr < self.max_size
        if self.otherwram:
            self.total_cost = self.bid
            self.total_imp = 2000.0
            self.total_click = 1
        self.beta = 3
        self.fw = 0
        #self.ptr += 1
        self.real_val = self.total_cost / self.total_imp
        self.last = self.ptr
        self.episode_len = episode_len
        self.step_size = (self.max_size-self.ptr) / self.episode_len
        self.now_step = 0
        self.usefixedbid = False # 使用固定的bid 减少难度
        #print(self.bid)

    def f(self, vs):
        if self.ptr <= self.trackedtime:
            weight, lens = 0, len(vs)+1
            for i in range(len(vs)):
                weight += vs[i]*np.log(i+2) / np.log(lens)
            return weight
        else:
            return 0

    def step(self, a, baseline=False):
        '''
        params:
        a 动作，输入的是现在价格是WLC价格的倍数

        return:
        state: [kp, ki, kd] pid的三个状态 后面会加上时间流式数据
        reward: 这一次action一共赚了多少钱, 要是导致赔付的话就是一个非常大的负数
        done: 赔付或者完美ok都是done
        info: bool 为了后面嫁接上gym程序, False表示没引发赔付
        '''
        # a = a[0]+1.6
        # self.real_val *= a
        #self.real_val = a
        a = a[0]
        
        
        if baseline:
            self.real_val = a
        else:
            a = 1.0 + a*0.01
            ratio = 1.1
            if self.ptr >= 20:
                ratio = 1
            pid_value = self.pid.cmd_pid(self.total_cost / self.total_click, self.bid / ratio)[1]*self.total_click / self.total_imp
            self.real_val = a * pid_value
        state, reward, done, info = None, 0, False, 0
        # if a < 0:
        #     print(self.ptr)
        #     done = True
        self.now_step += 1
        dis = 0
        #print('now action is', self.real_val)
        self.change = False
        clicknum = 0
        convnum = 0
        c = 0
        can = True
        for i in range(self.ptr, min(self.ptr+self.episode_len, self.max_size)):
            if self.data[i].log == 1:
                self.total_click += 1
                clicknum += 1
                if not self.usefixedbid:
                    self.bid = self.data[i].price
                    #self.bid = self.data[i].price
                    self.change = True
                '''
                
                '''
                self.last = self.ptr
            else:
                convnum += 1
                self.total_cost += self.real_val
                dis += self.real_val
                self.total_imp += 1
            #reward += -abs(self.total_cost / self.total_click - self.bid)
            #print('env output', self.real_val, self.total_cost / self.total_click, self.bid)
            #print(self.total_cost, self.total_cost / self.total_click, self.bid)
            #print(self.total_cost, self.total_click)
            # if self.total_cost / self.total_click > self.bid * (1+self.ratio) or self.total_cost / self.total_click < self.bid * (1-self.ratio):
            #     print(self.ptr, self.max_size)
            #     done = True
            #     info = True
            #     reward = -100
            # print(self.ptr)
            #print(self.bid, self.total_cost / self.total_click)
            #print(self.bid, self.beta*self.f(self.windows))
            if self.w:
                if self.total_cost / self.total_click > (self.bid+self.beta*self.f(self.windows)) * (1+self.ratio) and not self.change:
                #print(self.ptr, self.max_size)
                # done = True
                # info = self.ptr
                # reward = -20000
                    c = 1
            else:
                if self.total_cost / self.total_click > (self.bid) * (1+self.ratio+0.02) and not self.change:
                #print(self.ptr, self.max_size)
                # done = True
                # info = self.ptr
                # reward = -20000
                    c = 1
        
        #print(self.step_size, dis)
        self.ptr += self.episode_len
        
        if self.now_step >= self.step_size-1 or self.ptr >= 40000:
            done = True 
            if self.total_cost / self.total_click > self.bid * (1+self.ratio) and not self.change:
                can = False
        ratio = 1.1
        if self.ptr >= 20:
            ratio = 1
        state = list(self.pid.cmd_pid(self.total_cost / self.total_click, self.bid/ratio))+[self.total_cost / self.total_click]+[self.ptr-self.last]
        state[1] *= self.total_click / self.total_imp
        #print('pid value is ', state[3])
        if reward == 0 or baseline:
            reward = dis
        clicknum, convnum = convnum, clicknum
        if self.windowsptr == self.windowslen:
            for i in range(1, self.windowslen):
                self.windows[i-1] = self.windows[i]
            self.windows[self.windowslen-1] = clicknum
        else:
            self.windows.append(clicknum)
            self.windowsptr = self.windowsptr+1

        info = {'clicknum': clicknum, 'convnum': convnum, 'bid': self.bid, 'cost': c, 'can': can}
        # if done:
        #     print(self.ptr, self.max_size)
        if self.w:
            self.fw = 0.2*self.f(self.windows)
        else:
            self.fw = 0
        return np_tools.norm(state), reward, done, info
        #return np.array(state), reward, done, info

    def reset(self):
        self.real_val = 0
        self.ptr = 0
        self.total_cost = 0
        self.total_imp = 0
        self.total_click = 0
        self.now_step = 0
        self.init_ptr = 0
        self.windows = []
        self.windowslen = 100
        self.windowsptr = 0
        self.already_gain = 0
        # while self.ptr < self.max_size and self.data[self.ptr].log == 2:
        #     self.total_imp += 1
        #     self.ptr += 1
        
        
        # assert self.ptr < self.max_size

        '''
        use a fixed bid 227
        '''
        self.total_click = 0
        self.total_cost = 0
        while self.ptr < self.max_size and self.init_ptr < self.num:
            if self.data[self.ptr].log == 1:
                self.init_ptr += 1
                self.total_click += 1
                self.already_gain += self.bid
            else:
                self.total_imp += 1
            self.ptr += 1

        self.total_cost = self.num * self.bid
        assert self.ptr < self.max_size
        if self.otherwram:
            self.total_cost = self.bid
            self.total_imp = 2000
            self.total_click = 1
        self.real_val = self.total_cost / self.total_imp
        
        self.last = self.ptr
        self.pid.reset()
        state = list(self.pid.cmd_pid(self.total_cost / self.total_click, self.bid/8.0)) + [self.total_cost / self.total_click] + [self.ptr-self.last]
        state = np.asarray(state)
        state[1] *= self.total_click / self.total_imp
        #print(self.total_cost)
        return np_tools.norm(state)
        #return state
    
    def pd(self, real = 0, target = 0, wc = False):
        if not wc:
            real = self.total_cost / self.total_click
            target = self.bid
        return self.pid.cmd_pid(real, target)[1]*self.total_click / self.total_imp

    def show_sequence(self):
        self.reset()
        self.real_val = self.pd()
        self.usefixedbid = True
        seq = 0
        backconv = 0
        backclick = 0
        a1, a2, a3, b = [], [], [], []
        click, imp, dis, last = 0, 0, self.total_cost - self.already_gain, 0
        
        for i in range(self.ptr, self.max_size):
            if self.data[i].log == 1:
                click += 1
            else:
                imp += 1
        self.last = -1
        for i in range(self.ptr, self.max_size):
            if seq % 3 == 0:
                if self.last == -1:
                    self.real_val = self.pd()
                else:
                    self.real_val = self.pd()*0.2+self.last*0.8
                self.real_val = max(self.real_val, 1e-2)
                self.last = self.real_val
            if self.data[i].log == 1:
                self.total_click += 1 
                click -= 1
            else:
                self.total_cost += self.real_val
                self.total_imp += 1
                imp -= 1
                
            

            a3.append(self.real_val)
            b.append(self.bid)
            #print(self.bid)
            a1.append(self.total_cost / self.total_click)    
            seq += 1
            a2.append((self.total_cost + imp * self.real_val) / (self.total_click + click))    

        from safe_rl.pg.draw import draw6
        total1, total2 = {'cost-per-conversion':a1, 're-estimated cost-per-conversion':a2}, {'PID bids':a3}
        draw6(total1, total2, another_dir='1', bid = b)

    def show_bound(self):
        self.usefixedbid = True
        click, imp = 0, 0
        #print(len(self.data), self.max_size)
        for i in range(self.max_size):
            if self.data[i].log == 1:
                click += 1
            else:
                imp += 1
        return 1.20*self.bid*click