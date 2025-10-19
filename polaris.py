#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import cProfile
from loguru import logger
import time
import tracemalloc
from itertools import product
from pathlib import Path
from typing import Any, Set

from ttsim.config import TTSimHLRunSummary, get_arspec_from_yaml, get_wlmapspec_from_yaml, get_wlspec_from_yaml
from ttsim.front import onnx2graph
from ttsim.utils.common import get_ttsim_functional_instance, print_csv, str_to_bool
import ttsim.config.runcfgmodel as runcfgmodel
from ttsim.back.device import Device
from ttsim.stats import HLMStats, OutputFormat, save_data

""" Polaris top-level executable. """

LOG   = logger
INFO  = LOG.info
DEBUG = LOG.debug
ERROR = LOG.error
WARNING = LOG.warning

class RangeArgument:
    def __init__(self, name, arg, range_type='add'):
        self.vals = []
        if arg is not None:
            argx = arg[0]
            assert len(argx) == 3, f"range-arg: {name} = {argx} needs 3 numbers to specify a range"
            start, end, step = argx[0], argx[1], argx[2]

            assert start != end, f"Illegal RangeArgument: {name} [start({start}) == end({end})]"
            if range_type == 'mul' and step == 1:
                assert False, f"Illegal RangeArgument: {name} is of 'mul' type, step cannot be 1"
            elif range_type == 'add' and step == 0:
                assert False, f"Illegal RangeArgument: {name} is of 'add' type, step cannot be 0"
            else:
                pass

            if start > end:
                start, end, step = argx[1], argx[0], -1 * argx[2]
            assert start >= 1, f"range-arg: {name} = {argx} start cannot be < 1"

            assert range_type in ['add', 'mul'], \
                    f"range-arg: {name}, range_type({range_type}) can only be (add|mul)"
            x = start
            while (x <= end):
                self.vals.append(x)
                x = x + step if range_type == 'add' else x * step
            if self.vals[-1] < end:
                self.vals.append(end)
            assert len(self.vals) > 0, f"range-arg: {name} = {arg} specifies an empty range"

    def check(self):
        return len(self.vals) > 0

    def getvals(self):
        return self.vals

def check_args(args):
    assert args.inference != args.training, \
            f"Cannot run inference({args.inference}) & training({args.training}) together"
    return

def apply_filter(L, filter_csv_str, get_param_func):
    if filter_csv_str is not None:
        filter_fields = filter_csv_str.split(',')
        L = [x for x in L if get_param_func(x) in filter_fields]
    return L

def get_wlgraph(TBL, wlg, wln, wli, gcfg, wpath, enable_memalloc):
    xrows = [xrec for xrec in TBL if xrec[0] == wlg and xrec[1] == wln and xrec[2] == wli]
    wlb  = gcfg['bs']
    num_xrows = len(xrows)
    if num_xrows == 1 and xrows[0][3] is None:
        # we did not have the workload batch-size when we created TBL because batchsweep
        # was not set; now that the workload is instantiated, we have the batch-size
        # so, update the TBL accordingly
        del TBL[(wlg, wln, wli, None)]
        TBL[(wlg, wln, wli, wlb)] = None
    else:
        assert (wlg, wln, wli, wlb) in TBL, \
            f"Workload= {wlg}.{wln}.{wli}.b{wlb} missing in workloads graph table!!"

    if TBL[(wlg,wln,wli,wlb)] is None:
        if wlg == 'TTSIM':
            ttsim_wl  = get_ttsim_functional_instance(wpath, wln, gcfg) #<--- This is slow...

            if enable_memalloc:
                tracemalloc.start()
            ttsim_wl.create_input_tensors()
            ttsim_wl_out   = ttsim_wl() # noqa: F841 # we execute the graph and all the nodes are well formed
            if enable_memalloc:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                for stat in top_stats[:10]:
                    print("TRACEMALLOC:", stat)
                print("TRACEMALLOC: ttsim", "="*50, "\n")

            ttsim_wl_graph = ttsim_wl.get_forward_graph() # we should have a valid workload graph at this point
            TBL[(wlg,wln,wli,wlb)] = (ttsim_wl, ttsim_wl_graph)
            DEBUG(">>ttsim-wl analytical parameter count {}.{}.{}.b{}= {:,d}", wlg, wln, wli, wlb,
                    ttsim_wl.analytical_param_count())
        elif wlg == 'ONNX':
            print(f">>onnx-wl = {wlg}.{wln}.{wli}.b{wlb} = {wpath}")
            onnx_graph = onnx2graph(wli, wpath)
            TBL[(wlg,wln,wli,wlb)] = (None, onnx_graph)
            for _,op in onnx_graph._ops.items():
                itensors = [onnx_graph._tensors[x] for x in op.inList]
                otensors = [onnx_graph._tensors[x] for x in op.outList]
                print("CALLING get_perf_counts for:", op.name)
                #for x in itensors: print(x)
                #for x in otensors: print(x)
                op.get_perf_counts(itensors, otensors)
                op.update_tensor_counts(itensors,otensors)
                #print(op.perf_stats)
        else:
            assert False, f"Workload Group: {wlg} is not supported; Current Support only for (TTSIM,ONNX)"

    return TBL[(wlg,wln,wli,wlb)]

def do_instr_profile(_WLG, _ODIR):
    ITBL: dict[tuple[str, str, str, int], Any] = {}
    for _wlx, (wlobj, wlgraph) in _WLG.items():
        if _wlx not in ITBL: ITBL[_wlx] = {}  # noqa: E701
        for i,x in enumerate(wlgraph.get_ordered_nodes()):
            fwd_op = wlgraph._ops[x]
            for k,v in fwd_op.perf_stats['instrs'].items():
                if k not in ITBL[_wlx]: ITBL[_wlx][k] = 0  # noqa: E701
                ITBL[_wlx][k] += v

    profile_data = []
    for wlg, wln, wli, wlb in ITBL:
        instr_tbl =  ITBL[(wlg, wln, wli, wlb)]
        for instr, count in instr_tbl.items():
            profile_data.append({
                'group'      : wlg,
                'workload'   : wln,
                'instance'   : wli,
                'batchsize'  : wlb,
                'instruction': instr,
                'count'      : count
                })
    instr_profile_file = _ODIR / 'workload_instruction_profile.csv'
    print_csv(profile_data[0].keys(), profile_data, instr_profile_file)

def setup_cmdline_args(argv: list[str] | None = None) -> argparse.Namespace:
    logging_levels = [ 'debug', 'info', 'warning', 'error', 'critical' ]
    data_types     = [ 'fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'int32', 'int8' ]  # noqa: F841
    parser = argparse.ArgumentParser('polaris')

    parser.add_argument('--dryrun',    '-n', action='store_true', default=False, help='show but do not run')
    parser.add_argument('--instr_profile',   action='store_true', default=False, help='Collect Instruction Profile for Workloads')
    parser.add_argument('--dump_ttsim_onnx', action='store_true', default=False, help='Dump ONNX graph for TTSIM Workload')
    parser.add_argument('--enable_memalloc', action='store_true', default=False, help='Enable Memory Allocation Stats')
    parser.add_argument('--enable_cprofile', action='store_true', default=False, help='Enable CProfiler Stats ')

    parser.add_argument('--training',  '-t', type=str_to_bool, default='false', help='Training run')
    parser.add_argument('--inference', '-i', type=str_to_bool, default='true',  help='Inference run')
    parser.add_argument('--log_level', '-l', type=str,         default='info',  help="set logging level", choices=logging_levels)

    parser.add_argument('--odir',      '-o', required=True, help="Output Directory Name")
    parser.add_argument('--study',     '-s', required=True, help="Study Name")
    parser.add_argument('--wlspec',    '-w', required=True, help="Workloads Specification")
    parser.add_argument('--archspec',  '-a', required=True, help="Architecture Specification")
    parser.add_argument('--wlmapspec', '-m', required=True, help="Workload To Architecture Mapping Specification")
    parser.add_argument('--datatype', '-d', type=str, default=None, choices=data_types, help="Activation Data Type to use for the projection")

    parser.add_argument('--filterwlg',  default=None, help="use only workload-groups specified in filterwlg (comma sep list)")
    parser.add_argument('--filterwl',   default=None, help="use only workloads specified in filterwl (comma sep list)")
    parser.add_argument('--filterwli',  default=None, help="use only workload instances specified in filterwli (comma sep list)")
    parser.add_argument('--filterarch', default=None, help="use only architectures specified in filterarch (comma sep list)")
    parser.add_argument('--frequency', nargs=3, metavar=('start', 'end', 'step'), type=int,
                        action='append', help='frequency (in MHz) range specification (arith-seq)')
    parser.add_argument('--batchsize', nargs=3, metavar=('start', 'end', 'step'), type=int,
                        action='append', help='batchsize range specification (geom-seq)')
    parser.add_argument('--outputformat', choices=['none', 'yaml', 'json', 'pickle'], default='json', type=str.lower)
    parser.add_argument('--dump_stats_csv', dest='dump_stats_csv', action='store_true', 
                        default=False, help='Dump stats in CSV format')

    #cmdline args processing
    args = parser.parse_args(argv)
    check_args(args)

    #print cmdline for easy debug
    DEBUG("CMD=  python " + " ".join(sys.argv))

    return args

def dump_ttsim_onnx(TBL, _odir):
    onnx_dir = _odir / 'ONNX'
    os.makedirs(onnx_dir, exist_ok=True)

    #we pick the smallest batch to dump the ONNX...
    min_batchsize = min([wlb for _,_,_,wlb in TBL])
    for wlg, wln, wli, wlb in TBL:
        if wlb == min_batchsize:
            wlobj, wlgraph = TBL[(wlg, wln, wli, wlb)]
            onnx_ofilename = ".".join([wlg, wln, wli]) + f'.b{min_batchsize}' + '.onnx'
            wlgraph.graph2onnx(onnx_dir / onnx_ofilename)
    return

def get_devices(devspec, fsweep, filterarch):
    # Collect Device Specifications
    ipblks, devs = get_arspec_from_yaml(devspec)
    if fsweep.check():
        device_list = [(d,f) for d in devs for f in fsweep.getvals()]
    else:
        device_list = [(d,None) for d in devs]
    devlist = sorted(apply_filter(device_list, filterarch, lambda x: x[0]))

    INFO('reading device specification {}: found {:4d} #devices', devspec, len(devs))
    if fsweep.check():
        INFO('reading frequency sweep {}: found {:4d} #frequencies', " "*26, len(fsweep.getvals()))
    return devlist, devs

def get_workloads(wlspec, bsweep, filterwlg, filterwl, filterwli):
    # Collect Workload Specifications
    workload_specs = get_wlspec_from_yaml(wlspec)
    wl_list = []
    for wlname, wlist in workload_specs.items():
        for wl in wlist:
            wlapi = wl.api
            for wli_name, wli_cfg in wl.get_instances().items():
                if bsweep.check():
                    wl_list += [(wlapi, wl.name, wli_name, wli_cfg, b) for b in bsweep.getvals()]
                else:
                    wl_list += [(wlapi, wl.name, wli_name, wli_cfg, None)]
    INFO('reading workload specification {}: found {:4d} #workloads', wlspec, len(wl_list))
    #for ndx, tmp in enumerate(wl_list): INFO('{}: {}', ndx, tmp)
    wl_list = apply_filter(wl_list, filterwlg, lambda x: x[0])
    wl_list = apply_filter(wl_list, filterwl,  lambda x: x[1])
    wl_list = apply_filter(wl_list, filterwli, lambda x: x[2])
    wl_list = sorted(wl_list)
    num_batches   = len(bsweep.getvals()) if bsweep.check() else 1
    num_workloads = len(wl_list) // num_batches
    INFO('reading workloads specification {}: found {:4d} #workloads', wlspec, num_workloads)
    if bsweep.check():
        INFO('reading batch sweep                   : found {} #batch-sizes', num_batches)
    return wl_list, workload_specs

def create_uniq_workloads_tbl(WL_LIST):
    TBL = {(wlg, wln, wli, wlb): None for wlg, wln, wli, _, wlb in WL_LIST}
    return TBL

def do_dryrun(_wl, _dl):
    ALL_EXPS = product(_wl, _dl)

    #set field widths for diagnostic prints
    DEVNF = max([len(x) for x,_ in _dl])            + 1
    DEVFF = max([len(f"{x}") for _,x in _dl])       + 1
    WLGF  = max([len(x) for x,_,_,_,_ in _wl])      + 1
    WLNF  = max([len(x) for _,x,_,_,_ in _wl])      + 1
    WLIF  = max([len(x) for _,_,x,_,_ in _wl])      + 1
    WLB   = max([len(f"{x}") for _,_,_,_,x in _wl]) + 1

    for exp_no, (exp_wl, exp_dev) in enumerate(ALL_EXPS):
        wlg, wln, wli, wlicfg, wlb = exp_wl
        devname, devfreq           = exp_dev
        #diagnostics
        xstr  = f"  ..exp:{exp_no:3d} dev:{devname:{DEVNF}s} freq:"
        xstr += f"{devfreq}" if devfreq is None else f"{devfreq:{DEVFF}d}"
        xstr += f" wlg:{wlg:{WLGF}s} wl:{wln:{WLNF}s}"
        xstr += f"wli:{wli:{WLIF}s} wlb:"
        xstr += f"{wlb}" if wlb is None else f"{wlb:{WLB}d}"
        INFO(xstr)


    return

def execute_wl_on_dev(_wl, _dl, _wspec, _dspec, wlmapspec, _WLG,
                      _odir, study, _enable_memalloc, outputfmt, flag_dump_stats_csv):
    # TODO: Reduce number of arguments to this function
    study_dir = _odir / study
    stat_dir    = study_dir / 'STATS'
    config_dir  = study_dir / 'CONFIG'
    os.makedirs(stat_dir,    exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    saved_devices: Set[str] = set()
    _summary_stats = []
    num_failures = 0
    job_summaries: list[Any] = []
    ALL_EXPS = product(_wl, _dl)
    for exp_no, (exp_wl, exp_dev) in enumerate(ALL_EXPS):
        wlgroup, wlname, wlins_name, wlins_cfg, wlbatch = exp_wl
        devname, devfreq                                = exp_dev

        dev_obj   = _dspec[devname]
        if devfreq is not None:
            dev_obj.set_frequency(devfreq) #override device frequency is we have freqsweep

        wlpath    = wlins_cfg['path']
        wlcfg     = wlins_cfg['cfg']
        if wlbatch is not None:
            wlcfg['bs'] = wlbatch #override batch_size if we have batchsweep

        try:
            wlobj, wlgraph = get_wlgraph(_WLG, wlgroup, wlname, wlins_name, wlcfg, wlpath,
                                         _enable_memalloc)
        except Exception as e:
            num_failures += 1
            ERROR('workload {} failed with {}', exp_wl, e)
            raise

        try:
            #TODO: wlgroup, wlname, wlins_name == should be attributes of wlgraph, since these
            #      don't change w/ batchsize sweeps etc.
            wlinfo = dict(wlg=wlgroup, wln=wlname, wli=wlins_name, wlb=wlcfg['bs'])
            stats_info = {
                    'flag_dump_stats_csv'  : flag_dump_stats_csv,
                    'outputfmt'            : outputfmt,
                    'stat_dir'             : stat_dir,
                    'saved_devices'        : saved_devices,
                    'config_dir'           : config_dir,
                    'odir'                 : _odir,
                    }
            cur_device = Device(dev_obj)
            cur_device.execute_graph(wlgraph, wlmapspec)

            hlm_stats = HLMStats(cur_device, wlgraph, wlinfo, stats_info)
            summary_dict = hlm_stats.dump_stats(devfreq)
            _summary_stats.append(summary_dict)

        except Exception as e:
            num_failures += 1
            ERROR('workload {} failed with {}', exp_wl, e)
            continue

        INFO('ran job #{} instance={} device={} frequency={}', exp_no, wlins_name, devname, devfreq)

    return num_failures, _summary_stats

def polaris(args: argparse.Namespace | runcfgmodel.PolarisRunConfig) -> int:
    """Main entry point for the Polaris simulation."""
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    freqsweep = RangeArgument('frequency', args.frequency)
    batchsweep = RangeArgument('batchsize', args.batchsize, range_type='mul')
    if args.enable_cprofile:
        profiler   = cProfile.Profile()
        profiler.enable()

    outputformat = OutputFormat.enumvalue(args.outputformat)
    #create outdir & output book-keeping assets
    odir        = Path(args.odir)
    studydir    = odir / args.study
    summary_dir = studydir / 'SUMMARY'
    os.makedirs(odir,        exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    device_list, devspec  = get_devices(args.archspec, freqsweep, args.filterarch)
    workload_list, wlspec = get_workloads(args.wlspec, batchsweep, args.filterwlg, args.filterwl, args.filterwli)
    wlmapspec = get_wlmapspec_from_yaml(args.wlmapspec)

    if args.datatype is not None:
        # set the data type for the simulation
        wlmapspec.data_type_spec.update_global_type(args.datatype)
        INFO(f"Using data type: {wlmapspec.data_type_spec}")
    num_failures = 0
    if args.dryrun:
        do_dryrun(workload_list, device_list)
        tot_exp_run = 0
    else:
        workload_graphs = create_uniq_workloads_tbl(workload_list)

        INFO('simulation: workload+ --> device+')
        num_failures, summary_stats = execute_wl_on_dev(workload_list, device_list, wlspec, devspec,
                                                        wlmapspec,  workload_graphs, odir, args.study,
                                                        args.enable_memalloc, outputformat, args.dump_stats_csv)
        summary_stats = sorted(summary_stats, key=lambda x: (x['wlname'], x['devname'], x['freq_Mhz'], x['wlinstance'], x['bs']))

        summary_stat_filename = summary_dir / f'study-summary.{outputformat.cname}'
        save_data(TTSimHLRunSummary(**{'summary': summary_stats}), summary_stat_filename, outputformat)
        if args.dump_stats_csv:
            summary_stat_csv_filename = summary_dir / (summary_stat_filename.stem + '.csv')
            if summary_stats:
                print_csv(summary_stats[0].keys(), summary_stats, summary_stat_csv_filename)
            else:
                WARNING(f"No summary stats to dump to CSV file: {summary_stat_csv_filename} - creating empty file")
                print_csv([], [], summary_stat_csv_filename)


        if args.instr_profile:
            do_instr_profile(workload_graphs, odir)

        if args.dump_ttsim_onnx:
            dump_ttsim_onnx(workload_graphs, odir)

        tot_exp_run = len(summary_stats)

    if args.enable_cprofile:
        profiler.disable()
        profiler.dump_stats("polaris_cprofile_stats.prof")
    INFO("Polaris run completed with {} experiments.", tot_exp_run)
    return 0 if num_failures == 0 else 1

def main(argv: list[str] | None = None) -> int:
    # args, freqsweep, batchsweep = setup_cmdline_args()
    args = setup_cmdline_args(argv)
    return polaris(args)

if __name__ == '__main__':
    start_time = time.perf_counter()
    num_exps   = main()
    end_time   = time.perf_counter()
    del_time   = end_time - start_time

    if num_exps > 0:
        print(f"Completed {num_exps} jobs in {del_time:0.2f} seconds @ {num_exps/del_time:.0f} jobs per sec")
    else:
        print()
