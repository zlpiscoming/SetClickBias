2023-03-18 20:19:37.031519: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-18 20:19:37.134976: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-03-18 20:19:37.138187: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:37.138208: I tensorflow/compiler/xla/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-03-18 20:19:37.648507: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:37.648590: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:37.648599: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
2023-03-18 20:19:39.941142: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-18 20:19:40.191983: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192090: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublas.so.11'; dlerror: libcublas.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192128: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublasLt.so.11'; dlerror: libcublasLt.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192162: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192196: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192230: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192273: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusparse.so.11'; dlerror: libcusparse.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192308: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/leping_zhang/.mujoco/mujoco210/bin:/usr/lib/nvidia
2023-03-18 20:19:40.192324: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2023-03-18 20:19:40.204012: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:357] MLIR V1 optimization pass is not enabled
2023-03-18 20:19:40.427640: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
/home/leping_zhang/saferl/safety-starter-agents/scripts/safe_rl/pg/network.py:34: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
  x = tf.layers.dense(x, units=h, activation=activation)
/home/leping_zhang/saferl/safety-starter-agents/scripts/safe_rl/pg/network.py:35: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
  return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
[32;1mLogging data to /home/leping_zhang/saferl/safety-starter-agents/scripts/data/2023-03-18_cpo_PointGoal1/2023-03-18_20-19-39-cpo_PointGoal1_s0/progress.txt[0m
[36;1mSaving config:
[0m
{
    "_bid":	246,
    "ac_kwargs":	{
        "hidden_sizes":	[
            256,
            256
        ]
    },
    "actor_critic":	"mlp_actor_critic",
    "address":	"0606",
    "agent":	{
        "<safe_rl.pg.agents.CPOAgent object at 0x7f26e785cf10>":	{
            "backtrack_coeff":	0.8,
            "backtrack_iters":	10,
            "damping_coeff":	0.1,
            "learn_margin":	false,
            "margin":	0,
            "margin_lr":	0.05,
            "params":	{
                "constrained":	true,
                "learn_penalty":	false,
                "objective_penalized":	false,
                "penalty_param_loss":	false,
                "reward_penalized":	false,
                "save_penalty":	true,
                "trust_region":	true
            }
        }
    },
    "cost_gamma":	0.99,
    "cost_lam":	0.97,
    "cost_lim":	25,
    "cus":	"3358",
    "ent_reg":	0.0,
    "env_fn":	"Hopper-v3",
    "epochs":	1666,
    "exp_name":	"cpo_PointGoal1",
    "gamma":	0.99,
    "lam":	0.97,
    "logger":	{
        "<safe_rl.utils.logx.EpochLogger object at 0x7f26e785ca60>":	{
            "epoch_dict":	{},
            "exp_name":	"cpo_PointGoal1",
            "first_row":	true,
            "log_current_row":	{},
            "log_headers":	[],
            "output_dir":	"/home/leping_zhang/saferl/safety-starter-agents/scripts/data/2023-03-18_cpo_PointGoal1/2023-03-18_20-19-39-cpo_PointGoal1_s0",
            "output_file":	{
                "<_io.TextIOWrapper name='/home/leping_zhang/saferl/safety-starter-agents/scripts/data/2023-03-18_cpo_PointGoal1/2023-03-18_20-19-39-cpo_PointGoal1_s0/progress.txt' mode='w' encoding='UTF-8'>":	{
                    "mode":	"w"
                }
            }
        }
    },
    "logger_kwargs":	{
        "exp_name":	"cpo_PointGoal1",
        "output_dir":	"/home/leping_zhang/saferl/safety-starter-agents/scripts/data/2023-03-18_cpo_PointGoal1/2023-03-18_20-19-39-cpo_PointGoal1_s0"
    },
    "max_ep_len":	1000,
    "penalty_init":	1.0,
    "penalty_lr":	0.05,
    "render":	false,
    "save_freq":	50,
    "seed":	0,
    "steps_per_epoch":	6000,
    "target_kl":	0.01,
    "vf_iters":	80,
    "vf_lr":	0.001
}
Hopper-v3
PID  baseline is  26797.934400275997
pid total is 26797.934400275997
WLC  baseline is  26853.078554500527
wlc total is 26853.078554500527
233.37142857142857 2.8444001555508964 66.62857142857143 2.8444001555508964
MPC  baseline is  28638.54957125396
mpc total is 28638.54957125396
[32;1m
Number of parameters: 	 pi: 169872, 	 v: 67329, 	 vc: 67329
[0m
(4,) () False
Traceback (most recent call last):
  File "exp.py", line 82, in <module>
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, args.cus, args.bid, args.address)
  File "exp.py", line 48, in main
    algo(env_fn=env_name,
  File "/home/leping_zhang/saferl/safety-starter-agents/scripts/safe_rl/pg/algos.py", line 59, in cpo
    run_polopt_agent(agent=agent, **kwargs)
  File "/home/leping_zhang/saferl/safety-starter-agents/scripts/safe_rl/pg/run_agent.py", line 564, in run_polopt_agent
    get_action_outs = sess.run(get_action_ops, 
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 968, in run
    result = self._run(None, fetches, feed_dict, options_ptr,
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1191, in _run
    results = self._do_run(handle, final_targets, final_fetches,
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1371, in _do_run
    return self._do_call(_run_fn, feeds, fetches, targets, options,
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1378, in _do_call
    return fn(*args)
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1361, in _run_fn
    return self._call_tf_sessionrun(options, feed_dict, fetch_list,
  File "/home/leping_zhang/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1454, in _call_tf_sessionrun
    return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict,
KeyboardInterrupt
