#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.utils.types import get_sim_dtype, get_bpe

from typing import TYPE_CHECKING
from loguru import logger
import math

LOG     = logger
INFO    = LOG.info
DEBUG   = LOG.debug
ERROR   = LOG.error
WARNING = LOG.warning

class Component:
    def __init__(self, name: str, atype: str, **kwargs):
        self.name = name
        self.type = atype
        return

    def __str__(self):
        return f"{self.name} type={self.type}"

class MEM(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'MEM', **kwargs)
        self.size_nbytes      = kwargs.get('size')
        self.bw_bytes_per_clk = kwargs.get('bpc')
        return

    def __str__(self):
        return f"MEM " + super().__str__() + \
               f" size={self.size_nbytes:,d}B" + \
               f" bw={self.bw_bytes_per_clk:,d}B/Clk"

class NOC(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'NOC', **kwargs)
        self.nrows, self.ncols = kwargs.get('grid') #type: ignore
        return

    def __str__(self):
        return f"NOC " + super().__str__() + \
               f" nrows={self.nrows}, ncols={self.ncols}"

class PE(Component):
    def __init__(self, name, **kwargs):
        super().__init__(name, 'COMPUTE', **kwargs)
        return

    def __str__(self):
        return f"PE " + super().__str__()

class TTDevice:
    def __init__(self, name, **kwargs):
        self.name     = name
        self.mem_size = kwargs.get('mem_size')
        self.l1_size  = kwargs.get('l1_size')
        self.reg_size = kwargs.get('reg_size')
        self.mem_bw   = kwargs.get('mem_bw')
        self.l1_bw    = kwargs.get('l1_bw')
        self.reg_bw   = kwargs.get('reg_bw')
        self.noc_grid = kwargs.get('noc_grid')

        #components
        self.mem    = MEM(name + '.mem', size=self.mem_size, bpc=self.mem_bw)
        self.l1     = MEM(name + '.l1',  size=self.l1_size,  bpc=self.l1_bw)
        self.reg    = MEM(name + '.reg', size=self.reg_size, bpc=self.reg_bw)
        self.noc    = NOC(name + '.noc', grid=self.noc_grid)
        self.pe     = PE (name + '.pe')
        self.arch   = [self.mem, self.noc, self.l1, self.reg, self.pe]
        self.levels = len(self.arch)
        return

    def __str__(self):
        return "\n".join(f"{a}" for a in self.arch)

    def __getitem__(self, i):
        return self.arch[i]

class Device:
    """
        Hard coded for now

        For resource utilizations: decent estimates can follow a Tiler heuristic (compute)
        and profiling mem-stream benchmark measurements (memory)

        For fusion overlap, and SW/Host overhead guardband, we should get a better
        estimate from LLK team and UMD/Runtime

    """
    DG_COMPUTE_UTIL_CONSTANT        = 0.6
    DG_MEMORY_UTIL_CONSTANT         = 0.8
    G_FUSE_OP_OVERLAP_COST_CONSTANT = 0.10
    G_GUARDBAND                     = 0.25

    def __init__(self, simcfg_obj):
        compute_ips = [ipg for ipg in simcfg_obj.ipgroups if ipg.iptype == 'compute']
        memory_ips  = [ipg for ipg in simcfg_obj.ipgroups if ipg.iptype == 'memory']
        assert len(compute_ips) == 1, "ERR-1"
        assert len(memory_ips)  == 1, "ERR-2"

        self.simconfig_obj  = simcfg_obj
        self.devname        = simcfg_obj.devname
        self.name           = simcfg_obj.name
        self.freqMHZ        = simcfg_obj.frequency('matrix', units='MHz')
        self.compute_ip     = compute_ips[0]
        self.memory_ip      = memory_ips[0]
        return

    def execute_graph(self, wlgraph, wlmapspec):
        graph_ordered_nodes = wlgraph.get_ordered_nodes()

        # 1) SET PRECISION FOR ALL OPS
        wlgraph.set_precision(wlmapspec.data_type_spec)

        # 2) SET RESOURCES FOR ALL OPS
        wlgraph.set_resources(wlmapspec.rsrc_spec)

        # 3) EXECUTE OPS RESOURCES FOR ALL OPS : FIND COMPUTE/MEM CYCLES
        for opname in graph_ordered_nodes:
            op = wlgraph.get_op(opname)
            self.execute_op(op)

        # 4) GRAPH OPTIMIZATION: REMOVE NODES IF POSSIBLE
        wlgraph.remove_nodes(wlmapspec.removal_spec)

        # 5) GRAPH OPTIMIZATION: FUSE NODES IF POSSIBLE
        fusion_candidates = wlgraph.fuse_nodes(wlmapspec.fusion_spec)

        #Now all out fusion candidates have been found, and we can apply the
        # op-fusion on the graph
        for fusion_nodes in fusion_candidates:
            """create a new fused node with combined operations"""
            pattern_len   = len(fusion_nodes)
            first_op_name = fusion_nodes[0]
            last_op_name  = fusion_nodes[-1]
            first_op      = wlgraph.get_op(first_op_name)
            last_op       = wlgraph.get_op(last_op_name)

            """
            #update fusion op cycles
            # TODO: add some checks to make sure that intermediate fused ops have
            #   only one input - one output

            #compute cycles = sum of all fused op compute cycles + overhead per operator overlap
            # TODO: should we add overlap cost only if COMPUTE PIPES CHANGE?
            # intermediate mem rd/wr are suppressed by fusion
            # mem rd cycles = first op mem rd cycles
            # mem wr cycles = last op mem rd cycles
            """
            fused_matrix_cycles  = first_op.compute_cycles if first_op.uses_compute_pipe == 'matrix' else 0
            fused_vector_cycles  = first_op.compute_cycles if first_op.uses_compute_pipe == 'vector' else 0
            fused_compute_cycles = first_op.compute_cycles
            fused_mem_rd_cycles  = first_op.mem_rd_cycles
            fused_mem_wr_cycles  = last_op.mem_wr_cycles
            for i in range(1, pattern_len):
                matched_op_name  = fusion_nodes[i]
                matched_op       = wlgraph.get_op(matched_op_name)

                matrix_cycles = matched_op.compute_cycles if matched_op.uses_compute_pipe == 'matrix' else 0
                vector_cycles = matched_op.compute_cycles if matched_op.uses_compute_pipe == 'vector' else 0

                fused_matrix_cycles += math.ceil(matrix_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                fused_vector_cycles += math.ceil(vector_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                fused_compute_cycles += math.ceil(matched_op.compute_cycles * (1.0 + self.G_FUSE_OP_OVERLAP_COST_CONSTANT))
                matched_op.fuse_op(first_op_name)

            first_op.fused_op_cycles = {
                    'compute_cycles': fused_compute_cycles,
                    'matrix_cycles' : fused_matrix_cycles,
                    'vector_cycles' : fused_vector_cycles,
                    'mem_rd_cycles' : fused_mem_rd_cycles,
                    'mem_wr_cycles' : fused_mem_wr_cycles,
                    }

        return

    def execute_op(self, op):
        if TYPE_CHECKING:
            assert op.perf_stats is not None, f"SimOp {op.name} has no perf_stats set, cannot execute"

        #find compute cycles
        op.compute_cycles = 0
        for instr,instr_count in op.perf_stats['instrs'].items():
            peak_ipc = self.simconfig_obj.peak_ipc(op.uses_compute_pipe, instr, op.precision)
            real_ipc = peak_ipc * self.DG_COMPUTE_UTIL_CONSTANT
            op.compute_cycles += math.ceil(instr_count / real_ipc)
        #find memory cycles
        mem_rd_GB     = op.perf_stats['inBytes'] / 1024 / 1024 / 1024
        mem_wr_GB     = op.perf_stats['outBytes'] / 1024 / 1024 / 1024
        freq_MHz      = self.simconfig_obj.frequency(op.uses_compute_pipe, units='MHz')
        peak_bw_GBps  = self.simconfig_obj.peak_bandwidth(freq_units="GHz")
        bw_GBps       = peak_bw_GBps * self.DG_MEMORY_UTIL_CONSTANT
        #convert to device clk cycles
        op.mem_rd_cycles = math.ceil((mem_rd_GB / bw_GBps) * freq_MHz * 1e6)
        op.mem_wr_cycles = math.ceil((mem_wr_GB / bw_GBps) * freq_MHz * 1e6)

        return

    def get_exec_stats(self, wlgraph, bs):
        graph_ordered_nodes = wlgraph.get_ordered_nodes()

        for opnum,opname in enumerate(graph_ordered_nodes):
            op           = wlgraph.get_op(opname)

            if op.fused_op_cycles is None:
                compute_cycles = op.compute_cycles
                mem_cycles     = op.mem_rd_cycles + op.mem_wr_cycles
                matrix_cycles  = op.compute_cycles if op.uses_compute_pipe == 'matrix' else 0
                vector_cycles  = op.compute_cycles if op.uses_compute_pipe == 'vector' else 0
            else:
                compute_cycles = op.fused_op_cycles['compute_cycles']
                mem_cycles     = op.fused_op_cycles['mem_rd_cycles'] + op.fused_op_cycles['mem_wr_cycles']
                matrix_cycles  = op.fused_op_cycles['matrix_cycles']
                vector_cycles  = op.fused_op_cycles['vector_cycles']

            ramp_penalty     = self.simconfig_obj.ramp_penalty()
            dev_freq_MHz     = self.simconfig_obj.frequency(op.uses_compute_pipe, units='MHz')
            ideal_cycles     = max(compute_cycles, mem_cycles) + ramp_penalty
            ideal_msecs      = ideal_cycles / dev_freq_MHz / 1e3
            cycles           = math.ceil((1 + self.G_GUARDBAND) * ideal_cycles)
            msecs            = cycles / dev_freq_MHz / 1e3

            assert ideal_cycles > 0, f"Error: ideal_cycles = {ideal_cycles}!!"
            assert ideal_msecs > 0, f"Error: ideal_msecs = {ideal_msecs}!!"

            matrix_pipe_util = matrix_cycles/ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
            vector_pipe_util = vector_cycles/ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
            mem_rd_util      = op.mem_rd_cycles / ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT
            mem_wr_util      = op.mem_wr_cycles / ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT

            if op.removed_in_optimization or op.fused_in_optimization:
                rsrc_bnck        = 'NA'
                ideal_cycles     = 0
                cycles           = 0
                ideal_msecs      = 0.0
                msecs            = 0.0
                matrix_cycles    = 0
                vector_cycles    = 0
                matrix_pipe_util = 0.0
                vector_pipe_util = 0.0
                mem_rd_util      = 0.0
                mem_wr_util      = 0.0
            elif compute_cycles >= mem_cycles:
                rsrc_bnck = 'COMP'
            else:
                rsrc_bnck = 'MEM'

            op.exec_stats = {
                    'ramp_penalty'     : ramp_penalty,
                    'rsrc_bnck'        : rsrc_bnck,
                    'ideal_cycles'     : ideal_cycles,
                    'ideal_msecs'      : ideal_msecs,
                    'cycles'           : cycles,
                    'matrix_cycles'    : matrix_cycles,
                    'vector_cycles'    : vector_cycles,
                    'msecs'            : msecs,
                    'matrix_pipe_util' : matrix_pipe_util,
                    'vector_pipe_util' : vector_pipe_util,
                    'mem_rd_util'      : mem_rd_util,
                    'mem_wr_util'      : mem_wr_util,
                    }

        #compute aggregate stats
        def op_stat_iter(statname, /, repeat=False, use_precision=False):
            """ utility: iterates over wlgraph for a specific op.stat """
            for opname in graph_ordered_nodes:
                op = wlgraph.get_op(opname)
                if hasattr(op, statname):
                    val = getattr(op, statname)
                elif statname in op.perf_stats:
                    val = op.perf_stats.get(statname)
                elif statname in op.exec_stats:
                    val = op.exec_stats.get(statname)
                else:
                    val = None
                    assert False, f"unable to find {statname} in {op}"

                if repeat:
                    val = val * op.repeat_count

                if use_precision:
                    val = val * get_bpe(get_sim_dtype(op.precision))

                yield val

        tot_ideal_cycles = sum(op_stat_iter('ideal_cycles', repeat=True))
        tot_ideal_msecs  = sum(op_stat_iter('ideal_msecs',  repeat=True))
        tot_inParamCount = sum(op_stat_iter('inParamCount', repeat=True))
        tot_inParamBytes = sum(op_stat_iter('inParamCount', repeat=True, use_precision=True))
        tot_inActCount   = sum(op_stat_iter('inActCount'))
        tot_inActBytes   = sum(op_stat_iter('inActCount',  repeat=False, use_precision=True))
        tot_outActCount  = sum(op_stat_iter('outActCount'))
        tot_outActBytes  = sum(op_stat_iter('outActCount', repeat=False, use_precision=True))
        tot_maxActCount  = max(x+y for x,y in zip(op_stat_iter('inActCount'), op_stat_iter('outActCount')))
        tot_maxActBytes  = max(x+y for x,y in zip(op_stat_iter('inActCount', repeat=False, use_precision=True),
                                                  op_stat_iter('outActCount', repeat=False, use_precision=True)))

        tot_comp_bound_cycles = sum(y for x, y in zip(op_stat_iter('rsrc_bnck'),
                                                      op_stat_iter('ideal_cycles', repeat=True)) if x == 'COMP')
        tot_mem_bound_cycles  = sum(y for x, y in zip(op_stat_iter('rsrc_bnck'),
                                                      op_stat_iter('ideal_cycles', repeat=True)) if x == 'MEM')


        assert tot_ideal_cycles > 0, f"Error: tot_ideal_cycles = {tot_ideal_cycles}!!"
        assert tot_ideal_msecs > 0, f"Error: tot_ideal_msecs = {tot_ideal_msecs}!!"

        tot_matrix_cycles     = sum(op_stat_iter('matrix_cycles', repeat=True))
        tot_vector_cycles     = sum(op_stat_iter('vector_cycles', repeat=True))
        tot_mem_rd_cycles     = sum(op_stat_iter('mem_rd_cycles', repeat=True))
        tot_mem_wr_cycles     = sum(op_stat_iter('mem_wr_cycles', repeat=True))

        tot_matrix_pipe_util  = tot_matrix_cycles / tot_ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
        tot_vector_pipe_util  = tot_vector_cycles / tot_ideal_cycles * self.DG_COMPUTE_UTIL_CONSTANT
        tot_mem_rd_util       = tot_mem_rd_cycles / tot_ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT
        tot_mem_wr_util       = tot_mem_wr_cycles / tot_ideal_cycles * self.DG_MEMORY_UTIL_CONSTANT

        #check if fits device memory...
        tot_mem_size_GB  = (tot_inParamBytes + tot_maxActBytes)/1024/1024/1024
        device_mem_GB    = self.simconfig_obj.mem_size(units='GB')
        fits_device      = tot_mem_size_GB <= device_mem_GB

        #total perf metrics
        tot_ideal_throughput = bs * 1000 / tot_ideal_msecs
        tot_throughput       = (1 - self.G_GUARDBAND) * tot_ideal_throughput
        tot_cycles           = math.ceil((1 + self.G_GUARDBAND) * tot_ideal_cycles)
        tot_msecs            = (1 + self.G_GUARDBAND) * tot_ideal_msecs

        summary_stats = {
                'inParams'              : tot_inParamCount,
                'inActs'                : tot_inActCount,
                'outActs'               : tot_outActCount,
                'maxActs'               : tot_maxActCount,
                'inParamBytes'          : tot_inParamBytes,
                'inActBytes'            : tot_inActBytes,
                'outActBytes'           : tot_outActBytes,
                'maxActBytes'           : tot_maxActBytes,
                'tot_ideal_cycles'      : tot_ideal_cycles,
                'tot_ideal_msecs'       : tot_ideal_msecs,
                'tot_cycles'            : tot_cycles,
                'tot_msecs'             : tot_msecs,
                'ideal_throughput'      : tot_ideal_throughput,
                'perf_projection'       : tot_throughput,

                'tot_matrix_cycles'     : tot_matrix_cycles,
                'tot_vector_cycles'     : tot_vector_cycles,
                'tot_mem_rd_cycles'     : tot_mem_rd_cycles,
                'tot_mem_wr_cycles'     : tot_mem_wr_cycles,

                'tot_matrix_pipe_util'  : tot_matrix_pipe_util,
                'tot_vector_pipe_util'  : tot_vector_pipe_util,
                'tot_mem_rd_util'       : tot_mem_rd_util,
                'tot_mem_wr_util'       : tot_mem_wr_util,

                'rsrc_comp'             : tot_comp_bound_cycles/tot_ideal_cycles,
                'rsrc_mem'              : tot_mem_bound_cycles/tot_ideal_cycles,

                'mem_size_GB'           : tot_mem_size_GB,
                'device_memsize_GB'     : device_mem_GB,
                'fits_device'           : fits_device,
                'device_peak_bw_GBps'   : self.simconfig_obj.peak_bandwidth(freq_units="GHz"),
                'device_peak_fp8_tflops': self.simconfig_obj.peak_flops('matrix', 'mac', 'fp8', mul_factor=2),
                }

        return summary_stats

    def __str__(self):
        prefix = " "*4

        xstr  = f"Device:\n"
        xstr += f"{prefix}devname: {self.devname}\n"
        xstr += f"{prefix}name   : {self.name}\n"

        xstr += f"{prefix}Compute:\n"
        xstr += f"{prefix*2}ipname      : {self.compute_ip.ipname}\n"
        xstr += f"{prefix*2}num_units   : {self.compute_ip.num_units}\n"
        xstr += f"{prefix*2}freq_MHz    : {self.compute_ip.freq_MHz}\n"
        xstr += f"{prefix*2}ramp_penalty: {self.compute_ip.ramp_penalty}\n"
        if self.compute_ip.ipobj.l2_cache:
            xstr += f"{prefix*2}L2:\n"
            xstr += f"{prefix*3}num_banks              : {self.compute_ip.ipobj.l2_cache.num_banks}\n"
            xstr += f"{prefix*3}bytes_per_clk_per_bank : {self.compute_ip.ipobj.l2_cache.bytes_per_clk_per_bank}\n"
        xstr += f"{prefix*2}Pipes:\n"
        for pipe in self.compute_ip.ipobj.pipes:
            xstr += f"{prefix*2}-   name     : {pipe.name}\n"
            xstr += f"{prefix*3}num_units: {pipe.num_units}\n"
            xstr += f"{prefix*3}freq_MHz : {pipe.freq_MHz}\n"
            xstr += f"{prefix*3}instructions:\n"
            for ins in pipe.instructions:
                xstr += f"{prefix*3}-   {{name: {ins.name}, tpt: {ins.tpt} }}\n"

        xstr += f"{prefix}Memory:\n"
        xstr += f"{prefix*2}ipname     : {self.memory_ip.ipname}\n"
        xstr += f"{prefix*2}num_units  : {self.memory_ip.num_units}\n"
        xstr += f"{prefix*2}freq_MHz   : {self.memory_ip.freq_MHz}\n"
        xstr += f"{prefix*2}technology : {self.memory_ip.ipobj.technology}\n"
        xstr += f"{prefix*2}data_bits  : {self.memory_ip.ipobj.data_bits}\n"
        xstr += f"{prefix*2}freq_MHz   : {self.memory_ip.ipobj.freq_MHz}\n"
        xstr += f"{prefix*2}size_GB    : {self.memory_ip.ipobj.size_GB}\n"
        xstr += f"{prefix*2}stacks     : {self.memory_ip.ipobj.stacks}\n"
        xstr += f"{prefix*2}data_rate  : {self.memory_ip.ipobj.data_rate}\n"
        return xstr

if __name__ == '__main__':
    dev_cfg = {
            'mem_size': 2**30, 'mem_bw': 64,
            'l1_size' : 2**20, 'l1_bw' : 256,
            'reg_size': 2**16, 'reg_bw': 2048,
            'noc_grid': (8,8),
            }
    wh = TTDevice('wh', **dev_cfg)
    print(wh)
