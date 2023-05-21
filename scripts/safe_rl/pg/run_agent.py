import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import safety_gymnasium
import time
import os
import safe_rl.pg.trust_region as tro
from safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from safe_rl.pg.buffer import CPOBuffer
from safe_rl.pg.draw import draw3 as draw
from safe_rl.pg.draw import draw4
from safe_rl.pg.network import count_vars, \
                               get_vars, \
                               mlp_actor_critic,\
                               placeholders, \
                               placeholders_from_spaces
from safe_rl.pg.utils import values_as_sorted_list
from safe_rl.utils.logx import EpochLogger
from safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
import osqp
from scipy.sparse import csc_matrix
from scipy.stats import norm
import random

# Multi-purpose agent runner for policy optimization algos 
# (PPO, TRPO, their primal-dual equivalents, CPO)
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
        
 
    def run(self, now_val, exp_val):
        self.now_val = now_val
        
        self.last_last_err = self.last_err
        
        self.now_err = exp_val - self.now_val
        self.Ival[self.Iptr] = self.now_err
        self.Iptr = (self.Iptr+1) % self.max_size
         
        self.change_val = self.KP * self.now_err + self.KI * sum(self.Ival) + self.KD * (self.now_err - self.last_err)
        #if change_val < 0:
        self.last_err = self.now_err
        self.now_val += self.change_val
        
        #print('pid summary', sum(self.Ival), self.now_err, self.change_val)
        return self.now_val

class WLC():
    def __init__(self, lamb=10):
        self.lamb = lamb
        self.last = 0
        self.now_val = 0

    def run(self, now_val, exp_val):
        self.now_val = now_val
        dis = exp_val - now_val
        self.change_val = self.lamb * dis
        self.now_val += self.change_val
        self.last = self.change_val
        return self.now_val
    
    def reset(self):
        self.last = 0
        self.now_val = 0

class mpc():
    def __init__(self, k=50, p=0.95, ita=0.2, w=False, env=None):
        self.k = k
        self.w = w
        self.p = p
        self.ita = ita
        self.maxptr = env.max_size
        env.reset()
        
        clicknums, convnums = [], []
        while True:
            a = 0
            if env.ptr >= 20:
                ratio = 1
        
            s, r, d, infos = env.step([a], baseline=True)
            clicknums.append(int(infos['clicknum'])), convnums.append(int(infos['convnum']))
            if d:
                break
        #print(len(clicknums), len(convnums))
        c1, c2 = [], []
        for i in range(1, len(clicknums)):
            clicknums[i] += clicknums[i-1]
            convnums[i] += convnums[i-1]
            if i >= k:
                c1.append(clicknums[i]-clicknums[i-k])
                c2.append(convnums[i]-convnums[i-k])
        # print(c1)
        # print(c2)
        self.Ecost, self.Varcost, self.Econv, self.Varconv =  np.mean(c1), np.std(c1), np.mean(c2), np.std(c2)
        print(self.Ecost, self.Varcost, self.Econv, self.Varconv)

    def cdf(self, p, l=0, var=1):
        return norm.ppf(p, loc=l,scale=var)

    def cost_cdf(self, p):
        return self.cdf(p, self.Ecost, self.Varcost)

    def conv_cdf(self, p):
        return max(0, self.cdf(p, self.Econv, self.Varconv))

    def mpc(self, bold, pold, bid, ptr):
        m = osqp.OSQP()
        l = 0
        u = (1+self.ita)*(bold+self.conv_cdf(1-self.p)*bid)-pold+60*(self.maxptr-ptr)/self.maxptr
        u = max(l, u)
        if self.w:
            u += 300*(self.maxptr-ptr)/self.maxptr
        #print((1+self.ita), (bold+self.conv_cdf(1-self.p)*bid), pold)
        m.setup(P=csc_matrix([[0]]), q=np.array([-self.Ecost]), A=csc_matrix([[self.cost_cdf(self.p)]]), l=np.array([0]), u=np.array([u]), verbose=False)
        res = m.solve()
        return res.x[0]


def run_polopt_agent(env_fn, 
                     agent=PPOAgent(),
                     actor_critic=mlp_actor_critic, 
                     ac_kwargs=dict(), 
                     seed=0,
                     render=False,
                     # Experience collection:
                     steps_per_epoch=4000, 
                     epochs=50, 
                     max_ep_len=2000,
                     # Discount factors:
                     gamma=0.99, 
                     lam=0.97,
                     cost_gamma=0.99, 
                     cost_lam=0.97, 
                     # Policy learning:
                     ent_reg=0.,
                     # Cost constraints / penalties:
                     cost_lim=25,
                     penalty_init=1.,
                     penalty_lr=5e-2,
                     # KL divergence:
                     target_kl=0.01, 
                     # Value learning:
                     vf_lr=1e-3,
                     vf_iters=80, 
                     # Logging:
                     logger=None, 
                     logger_kwargs=dict(), 
                     save_freq=1,
                     cus='3358', _bid=227,
                     address='0607', withF = True,
                     otherwarm = True,
                     constraint_type = '',
                     ):


    #=========================================================================#
    #  Prepare logger, seed, and environment in this process                  #
    #=========================================================================#
    print('future info is', cus, _bid, address, withF, otherwarm)
    logger = EpochLogger(**logger_kwargs) if logger is None else logger
    logger.save_config(locals())
    # bygym=False
    # bysafegym=True
    bygym=False
    bysafegym=False
    #print('len', max_ep_len)
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    #print(env_fn)
    if bygym:
        env = gym.make(env_fn)
    elif bysafegym:
        env = safety_gymnasium.vector.make(env_fn)
    else:
        from safe_rl.pg.EnvCD2 import ValueController
        env = ValueController(basedir='/home/leping_zhang/rl/torch1/data/', ad=cus, bid=_bid, dat=address, w=withF, otherwarm=otherwarm)
    
    max_ret = 0
    p = PID(10, 0.01, 0.1)
    baselines, bids, maxval = {}, [], {}
    def test(p, name):
        global max_ret
        global baselines
        env.reset()
        p.now_val = env.bid
        pold, bold, last = env.total_cost, env.total_click*env.bid, 0
        total = 0
        y, bs = [], []
        ratio = 1.1
        while True:
            
            if name=='MPC':
                a = max(p.mpc(pold, bold, env.bid, env.ptr), 0)
                if last != 0:
                    a = last*0.98 + a*0.02
                last = a
            else:
                a = p.run(env.total_cost / env.total_click , env.bid / ratio + max(0, env.fw)) 
                a = a * env.total_click / env.total_imp
                a = max(a, 0)
            if env.ptr >= 20:
                ratio = 1
            s, r, d, info = env.step([a], baseline=True)
            bs.append(info['bid'])
            y.append(env.total_cost / env.total_click)
            #print(env.real_val , env.bid * env.total_click / env.total_imp)
            total += r
            if d:
                break
        # print(name,' baseline is ', total)
        # max_ret = max(max_ret, total)
        # baselines[name] = y
        return bs, y, total
    
    bids, y, total = test(p, 'PID')
    #print('pid total is', total)
    max_ret = max(max_ret, total)
    maxval['PID'] = total
    if withF:
        baselines['PIDwithF'] = y
    else:
        baselines['PIDwithoutF'] = y
    p = PID(10)
    bids, y, total = test(p, 'WLC')
    maxval['WL'] = total
    print('total is', total)
    max_ret = max(max_ret, total)
    if withF:
        baselines['WLCwithF'] = y
    else:
        baselines['WLCwithoutF'] = y
    if withF:
        p = mpc(80, 0.8, 0.1, True, env)
    else:
        p = mpc(20, 0.95, 0.1, False, env)
    bids, y, total = test(p, 'MPC')
    total_c, over_c = 0, 0
    # constraint_type, total_c, over_c
    #print('lens', len(baselines['PIDwithF']), len(baselines['WLCwithF']), len(baselines['WLCwithF']))
    #max_ret = max(max_ret, total)
    maxval['MPC'] = total
    if withF:
        baselines['LPwithF'] = y
    else:
        baselines['LPwithoutF'] = y
    #k=50, p=0.95, ita=0.2, w=False, env=None
    agent.set_logger(logger)
    
    #=========================================================================#
    #  Create computation graph for actor and critic (not training routine)   #
    #=========================================================================#

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    s_space, a_space = env.observation_space, env.action_space
    #s_space[3]
    #from gym.spaces import Box, Discrete
    #print(isinstance(s_space, Box), type(s_space))
    # Inputs to computation graph from environment spaces
    x_ph, a_ph = placeholders_from_spaces(s_space, a_space)

    # Inputs to computation graph for batch data
    adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph = placeholders(*(None for _ in range(5)))

    # Inputs to computation graph for special purposes
    surr_cost_rescale_ph = tf.placeholder(tf.float32, shape=())
    cur_cost_ph = tf.placeholder(tf.float32, shape=())

    # Outputs from actor critic
    ac_outs = actor_critic(x_ph, a_ph, **ac_kwargs)
    pi, logp, logp_pi, pi_info, pi_info_phs, d_kl, ent, v, vc = ac_outs

    # Organize placeholders for zipping with data from buffer on updates
    buf_phs = [x_ph, a_ph, adv_ph, cadv_ph, ret_ph, cret_ph, logp_old_ph]
    buf_phs += values_as_sorted_list(pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    get_action_ops = dict(pi=pi, 
                          v=v, 
                          logp_pi=logp_pi,
                          pi_info=pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    if not(agent.reward_penalized):
        get_action_ops['vc'] = vc

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'vf', 'vc'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n'%var_counts)

    # Make a sample estimate for entropy to use as sanity check
    approx_ent = tf.reduce_mean(-logp)


    #=========================================================================#
    #  Create replay buffer                                                   #
    #=========================================================================#

    # Obs/act shapes
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in pi_info_phs.items()}
    buf = CPOBuffer(local_steps_per_epoch,
                    obs_shape, 
                    act_shape, 
                    pi_info_shapes, 
                    gamma, 
                    lam,
                    cost_gamma,
                    cost_lam,
                    bysafegym=bysafegym)

    
    #=========================================================================#
    #  Create computation graph for penalty learning, if applicable           #
    #=========================================================================#

    if agent.use_penalty:
        with tf.variable_scope('penalty'):
            # param_init = np.log(penalty_init)
            param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
            penalty_param = tf.get_variable('penalty_param',
                                          initializer=float(param_init),
                                          trainable=agent.learn_penalty,
                                          dtype=tf.float32)
        # penalty = tf.exp(penalty_param)
        penalty = tf.nn.softplus(penalty_param)

    if agent.learn_penalty:
        if agent.penalty_param_loss:
            penalty_loss = -penalty_param * (cur_cost_ph - cost_lim)
        else:
            penalty_loss = -penalty * (cur_cost_ph - cost_lim)
        train_penalty = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)


    #=========================================================================#
    #  Create computation graph for policy learning                           #
    #=========================================================================#

    # Likelihood ratio
    ratio = tf.exp(logp - logp_old_ph)

    # Surrogate advantage / clipped surrogate advantage
    if agent.clipped_adv:
        min_adv = tf.where(adv_ph>0, 
                           (1+agent.clip_ratio)*adv_ph, 
                           (1-agent.clip_ratio)*adv_ph
                           )
        surr_adv = tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    else:
        surr_adv = tf.reduce_mean(ratio * adv_ph)

    # Surrogate cost
    surr_cost = tf.reduce_mean(ratio * cadv_ph)

    # Create policy objective function, including entropy regularization
    pi_objective = surr_adv + ent_reg * ent

    # Possibly include surr_cost in pi_objective
    if agent.objective_penalized:
        pi_objective -= penalty * surr_cost
        pi_objective /= (1 + penalty)

    # Loss function for pi is negative of pi_objective
    pi_loss = -pi_objective

    # Optimizer-specific symbols
    if agent.trust_region:

        # Symbols needed for CG solver for any trust region method
        pi_params = get_vars('pi')
        flat_g = tro.flat_grad(pi_loss, pi_params)
        v_ph, hvp = tro.hessian_vector_product(d_kl, pi_params)
        if agent.damping_coeff > 0:
            hvp += agent.damping_coeff * v_ph

        # Symbols needed for CG solver for CPO only
        flat_b = tro.flat_grad(surr_cost, pi_params)

        # Symbols for getting and setting params
        get_pi_params = tro.flat_concat(pi_params)
        set_pi_params = tro.assign_params_from_flat(v_ph, pi_params)

        training_package = dict(flat_g=flat_g,
                                flat_b=flat_b,
                                v_ph=v_ph,
                                hvp=hvp,
                                get_pi_params=get_pi_params,
                                set_pi_params=set_pi_params)

    elif agent.first_order:

        # Optimizer for first-order policy optimization
        train_pi = MpiAdamOptimizer(learning_rate=agent.pi_lr).minimize(pi_loss)

        # Prepare training package for agent
        training_package = dict(train_pi=train_pi)

    else:
        raise NotImplementedError

    # Provide training package to agent
    training_package.update(dict(pi_loss=pi_loss, 
                                 surr_cost=surr_cost,
                                 d_kl=d_kl, 
                                 target_kl=target_kl,
                                 cost_lim=cost_lim))
    agent.prepare_update(training_package)

    #=========================================================================#
    #  Create computation graph for value learning                            #
    #=========================================================================#

    # Value losses
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    vc_loss = tf.reduce_mean((cret_ph - vc)**2)

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    if agent.reward_penalized:
        total_value_loss = v_loss
    else:
        total_value_loss = v_loss + vc_loss

    # Optimizer for value learning
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(total_value_loss)


    #=========================================================================#
    #  Create session, sync across procs, and set up saver                    #
    #=========================================================================#

    #session = tf.Session(config=config)

    config = tf.ConfigProto()
    
    # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.allow_growth = True
    
    # 配置可使用的显存比例
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v, 'vc': vc})


    #=========================================================================#
    #  Provide session to agent                                               #
    #=========================================================================#
    agent.prepare_session(sess)


    #=========================================================================#
    #  Create function for running update (called at end of each epoch)       #
    #=========================================================================#

    def update():
        cur_cost = logger.get_stats('EpCost')[0]
        c = cur_cost - cost_lim
        if c > 0 and agent.cares_about_cost:
            logger.log('Warning! Safety constraint is already violated.', 'red')

        #=====================================================================#
        #  Prepare feed dict                                                  #
        #=====================================================================#

        inputs = {k:v for k,v in zip(buf_phs, buf.get())}
        inputs[surr_cost_rescale_ph] = logger.get_stats('EpLen')[0]
        inputs[cur_cost_ph] = cur_cost

        #=====================================================================#
        #  Make some measurements before updating                             #
        #=====================================================================#

        measures = dict(LossPi=pi_loss,
                        SurrCost=surr_cost,
                        LossV=v_loss,
                        Entropy=ent)
        if not(agent.reward_penalized):
            measures['LossVC'] = vc_loss
        if agent.use_penalty:
            measures['Penalty'] = penalty

        pre_update_measures = sess.run(measures, feed_dict=inputs)
        logger.store(**pre_update_measures)

        #=====================================================================#
        #  Update penalty if learning penalty                                 #
        #=====================================================================#
        if agent.learn_penalty:
            sess.run(train_penalty, feed_dict={cur_cost_ph: cur_cost})

        #=====================================================================#
        #  Update policy                                                      #
        #=====================================================================#
        agent.update_pi(inputs)

        #=====================================================================#
        #  Update value function                                              #
        #=====================================================================#
        for _ in range(vf_iters):
            sess.run(train_vf, feed_dict=inputs)

        #=====================================================================#
        #  Make some measurements after updating                              #
        #=====================================================================#

        del measures['Entropy']
        measures['KL'] = d_kl

        post_update_measures = sess.run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        logger.store(KL=post_update_measures['KL'], **deltas)




    #=========================================================================#
    #  Run main environment interaction loop                                  #
    #=========================================================================#

    start_time = time.time()
    total = 0
    y = []
    if bygym:
        o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    elif bysafegym:
        #print(env.reset())
        o, info = env.reset()
        r, d, c, ep_ret, ep_cost, ep_len = 0, False, 0, 0, 0, 0
    else:
        o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    if bygym or bysafegym:
        o = o[0]
    cur_penalty = 0
    cum_cost = 0
    max_ret = 0
    for epoch in range(epochs):

        if agent.use_penalty:
            cur_penalty = sess.run(penalty)

        for t in range(local_steps_per_epoch):
            if bysafegym:
                o = np.squeeze(o)
                get_action_outs = sess.run(get_action_ops, 
                                       feed_dict={x_ph: o[np.newaxis]})
            else:
                #print(o)
                get_action_outs = sess.run(get_action_ops, 
                                        feed_dict={x_ph: o[np.newaxis]})
            a = get_action_outs['pi']
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']

            # Step in environment
            a = a[0]
            if bygym:
                o2, r, d, _, info = env.step(a)
                c = 0
            elif not bygym and not bysafegym:
                a = [a]
                o2, r, d, info = env.step(a)
                y.append(env.total_cost / env.total_click)
                c = info.get('cost', 0)
            else:
                o2, r, c, d, _, info = env.step(a)
                c = info.get('cost', 0)
            cum_cost += c

            # save and log
            if agent.reward_penalized:
                r_total = r - cur_penalty * c
                r_total = r_total / (1 + cur_penalty)
                buf.store(o, a, r_total, v_t, 0, 0, logp_t, pi_info_t)
            else:
                buf.store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)
            logger.store(VVals=v_t, CostVVals=vc_t)
            # if not bysafegym and not bygym:
            #     o = np.squeeze(o2, axis=0)
            # else:
            o = o2
            
            #print('o2', o2[np.newaxis].shape)
            ep_ret += r
            ep_cost += c
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                total_c += 1

                if not ep_cost and ep_ret > max_ret and info['can'] and t!=local_steps_per_epoch-1:
                    max_ret = ep_ret
                    over_c += 1
                    import os
                    #draw(y, baselines, cusname=cus, bid=bids)
                    # draw future
                    # if withF:
                    #     f= open("data.txt",encoding='utf-8')
                    #     a = f.read().split(',')
                    #     a = [float(s) for s in a]
                    #     #baselines = {'RLwithoutF': a, 'RLwithF': y}
                    #     #print('len is ', len(y), len(baselines['LPwithF']))
                    #     baselines['RLwithF'] = y
                    #     draw4(baselines, cusname=cus, bid=bids)
                    # else:
                    #     baselines['RLwithoutF'] = y
                    #     draw4(baselines, cusname=cus, bid=bids)
                    #     f=open('data.txt',"w")
                    #     f.write(",".join((str(s) for s in y)))
                    
                    # f.close()
                    # baselines = {'RLwithoutFuture': a}
                    # draw(y, baselines, cusname=cus, bid=bids)
                # If trajectory didn't reach terminal state, bootstrap value target(s)
                if d and not(ep_len == max_ep_len):
                    # Note: we do not count env time out as true terminal state
                    last_val, last_cval = 0, 0
                else:
                    feed_dict={x_ph: o[np.newaxis]}
                    if agent.reward_penalized:
                        last_val = sess.run(v, feed_dict=feed_dict)
                        last_cval = 0
                    else:
                        last_val, last_cval = sess.run([v, vc], feed_dict=feed_dict)
                buf.finish_path(last_val, last_cval)

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                else:
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                # Reset environment
                if bygym:
                    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
                elif bysafegym:
                    #print(env.reset())
                    o, info = env.reset()
                    r, d, c, ep_ret, ep_cost, ep_len = 0, False, 0, 0, 0, 0
                else:
                    y = []
                    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
                if bygym or bysafegym:
                    o = o[0]

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        #=====================================================================#
        #  Run RL update                                                      #
        #=====================================================================#
        update()

        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        #=====================================================================#
        #  Log performance and stats                                          #
        #=====================================================================#

        logger.log_tabular('Epoch', epoch)

        # Performance stats
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('CumulativeCost', cumulative_cost)
        # logger.log_tabular('CostRate', cost_rate)
        #
        # # Value function values
        # logger.log_tabular('VVals', with_min_and_max=True)
        # logger.log_tabular('CostVVals', with_min_and_max=True)
        #
        # # Pi loss and change
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        #
        # # Surr cost and change
        # logger.log_tabular('SurrCost', average_only=True)
        # logger.log_tabular('DeltaSurrCost', average_only=True)
        #
        # # V loss and change
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        #
        # # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        # if not(agent.reward_penalized):
        #     logger.log_tabular('LossVC', average_only=True)
        #     logger.log_tabular('DeltaLossVC', average_only=True)
        #
        # if agent.use_penalty or agent.save_penalty:
        #     logger.log_tabular('Penalty', average_only=True)
        #     logger.log_tabular('DeltaPenalty', average_only=True)
        # else:
        #     logger.log_tabular('Penalty', 0)
        #     logger.log_tabular('DeltaPenalty', 0)

        # Anything from the agent?
        agent.log()

        # Policy stats
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        #
        # # Time and steps elapsed
        # logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    if withF:
        maxval['UBCRL'] = max_ret
    else:
        maxval['RL'] = max_ret
    print('RL is', max_ret)
    import os
    with open("/home/leping_zhang/saferl/result.log", encoding="utf-8", mode="a") as file:
        file.write(cus + '---------------------------------------  '+constraint_type+'\n')

        file.write(str(round(over_c / total_c, 2)))
        file.write('\n')
        #logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ppo')
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cost_gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--len', type=int, default=1000)
    parser.add_argument('--cost_lim', type=float, default=10)
    parser.add_argument('--exp_name', type=str, default='runagent')
    parser.add_argument('--kl', type=float, default=0.01)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward_penalized', action='store_true')
    parser.add_argument('--objective_penalized', action='store_true')
    parser.add_argument('--learn_penalty', action='store_true')
    parser.add_argument('--penalty_param_loss', action='store_true')
    parser.add_argument('--entreg', type=float, default=0.)
    args = parser.parse_args()

    try:
        import safety_gym
    except:
        print('Make sure to install Safety Gym to use constrained RL environments.')

    mpi_fork(args.cpu)  # run parallel code with mpi

    # Prepare logger
    from safe_rl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Prepare agent
    agent_kwargs = dict(reward_penalized=args.reward_penalized,
                        objective_penalized=args.objective_penalized,
                        learn_penalty=args.learn_penalty,
                        penalty_param_loss=args.penalty_param_loss)
    if args.agent=='ppo':
        agent = PPOAgent(**agent_kwargs)
    elif args.agent=='trpo':
        agent = TRPOAgent(**agent_kwargs)
    elif args.agent=='cpo':
        agent = CPOAgent(**agent_kwargs)

    run_polopt_agent(lambda : gym.make(args.env),
                     agent=agent,
                     actor_critic=mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                     seed=args.seed, 
                     render=args.render, 
                     # Experience collection:
                     steps_per_epoch=args.steps, 
                     epochs=args.epochs,
                     max_ep_len=args.len,
                     # Discount factors:
                     gamma=args.gamma,
                     cost_gamma=args.cost_gamma,
                     # Policy learning:
                     ent_reg=args.entreg,
                     # KL Divergence:
                     target_kl=args.kl,
                     cost_lim=args.cost_lim, 
                     # Logging:
                     logger_kwargs=logger_kwargs,
                     save_freq=1
                     )