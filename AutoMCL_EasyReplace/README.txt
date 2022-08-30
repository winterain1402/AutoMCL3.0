setup.sh--------------------set up script

AutoMCL_EasyReplace
-------InitConfigTask: Tiling size initialization for a specific task size
-------AutoConfig: a optimized tuner for different dense and conv2d task
-------AS_OS: Optimal Schedule for conv2d and Automatic Schedule selection for dense

Source code replacement mapping:
----------------EasyReplace/AutoConfig     --> tvm/python/tvm/autotvm/tuner
----------------EasyReplace/InitConfigTask --> tvm/python/tvm/autotvm/task
----------------EasyReplace/AS_OS          --> tvm/topi/python/topi/x86