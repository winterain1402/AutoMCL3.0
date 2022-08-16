# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and Relay
99.9% copy-paste of implementation by @MerryMercy

"""
import threading
import logging

import tvm
from tvm.autotvm.task.dispatcher import DispatchContext, FallbackContext
from tvm.target import Target
from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger("autotvm")


# TODO(moreau89) find a more elegant way to lower for VTAs
def _lower(func,
           target,
           params):
    """ Helper to lower VTA properly.
    """

    from tvm import relay
    from tvm.relay.backend import graph_executor_codegen

    if hasattr(target, 'device_name') and target.device_name == "vta":
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            import vta
            with vta.build_config():
                mod, _ = relay.optimize(func, target, params)
                grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
                return grc.codegen(mod["main"])
    # default case
    mod, _ = relay.optimize(func, target, params)
    grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
    return grc.codegen(mod["main"])


def extract_from_program(func, params, target, target_host=None, ops=None,
                         template_keys=None):
    """ Extract tuning tasks from a relay program.

    This function is the single program version of extract_from_multiple_program.

    Parameters
    ----------
    func: relay.expr.Function
        The func to tune
    params: dict of str to numpy array
        The associated parameters of the program
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target
    template_keys: dict of topi op to str
        The tuning template keys map for schedules, default to None.
        Example: {topi.nn.conv2d: 'direct'}

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    return extract_from_multiple_program([func], [params], ops, target, target_host,
                                         template_keys=template_keys)


def extract_from_multiple_program(funcs, params, ops, target, target_host=None,):
    """ Extract tuning tasks from multiple relay programs.

    This function collects tuning tasks by building a list of programs
    with a "tracing" target and tracing all the calls to topi.

    Parameters
    ----------
    funcs: List of relay.expr.Function
        The list of functions to tune
    params: List of dict of str to numpy array
        The associated parameters of the programs
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target
    template_keys: dict of topi op to str
        The tuning template keys map for schedules, default to None.
        Example: {topi.nn.conv2d: 'direct'}

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm import topi

    env = TaskExtractEnv.get()

    # merge target and target host
    target, target_host = Target.canon_target_and_host(target, target_host)

    # run compiler to collect all TOPI calls during compilation
    env.reset(ops)
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        for func, param in zip(funcs, params):
            relay.backend.te_compiler.get().clear()
            # wrap build call in thread to avoid multiprocessing problems
            build_thread = threading.Thread(target=_lower, args=(mod, target, param))
            build_thread.start()
            build_thread.join()
            relay.backend.te_compiler.get().clear()
            # Clear the warning message cache in FallbackContext
            if isinstance(DispatchContext.current, FallbackContext):
                DispatchContext.current.memory = {}
                DispatchContext.warning_messages = set()

        logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            tsk = create(task_name, args, target=target)
            tasks.append(tsk)
        except topi.InvalidShapeError:
            logger.warning("Invalid shape during AutoTVM task creation")

    return tasks

# 首先找到tvm0.9.0对0.6.0的修改，并向tvm0.9.0进行同步
# 去除import warnings
# 新增26，27，28三行
# 同步了43行
# 同步了174-179行，删除原160-169行，新增157-161行
# 修改了134行为te.complier
