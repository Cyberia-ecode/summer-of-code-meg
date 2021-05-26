from megengine.tools.load_network_and_run import *


def run_model_(args, graph, inputs, outputs, data):
    out_data = ''
    
    # must use level0 to avoid unintended opr modification
    graph.options.graph_opt_level = 0
    
    logger.info("input tensors: ")
    for k, v in data.items():
        logger.info("  {}: {}".format(k, v.shape))
        out_data = out_data + "  {}: {}".format(k, v.shape) + '\n'
    
    G.modify_opr_algo_strategy_inplace(outputs, get_execution_strategy(args))
    
    if args.optimize_for_inference:
        opt_kwargs = get_opt_kwargs(args)
        outputs = G.optimize_for_inference(outputs, **opt_kwargs)
    
    # embed inputs must be on the last, to avoid const fold
    if args.embed_input:
        outputs, inp_dict = tools.embed_inputs(outputs, data.values(), inputs=inputs)
    else:
        outputs, inp_dict = tools.convert_inputs(outputs, inputs=inputs)
    
    if args.dump_cpp_model:
        dump_content, _ = G.dump_graph(outputs, keep_var_name=2)
        with open(args.dump_cpp_model, "wb") as file:
            file.write(dump_content)
        logger.info("C++ model written to {}".format(args.dump_cpp_model))
    
    outputs, output_dict = tools.convert_outputs(outputs)
    
    if args.profile:
        profiler = tools.GraphProfiler(graph)
    
    func = graph.compile(outputs)
    
    def run():
        if not args.embed_input:
            for key in inp_dict:
                inp_dict[key].set_value(mge.Tensor(data[key])._dev_tensor())
        func.execute()
        func.wait()
        return [oup_node.get_value().numpy() for oup_node in output_dict.values()]
    
    if args.warm_up:
        logger.info("warming up")
        run()
    
    total_time = 0
    
    for i in range(args.iter):
        logger.info("iter {}".format(i))
        start_time = time.time()
        retval = run()
        cur_time = time.time() - start_time
        total_time += cur_time
        
        avg_speed = (i + 1) / total_time
        if "data" in data:
            avg_speed *= data["data"].shape[0]
            avg_speed_txt = "{:.3f}sample/s".format(avg_speed)
        else:
            avg_speed_txt = "{:.3f}batch/s".format(avg_speed)
        
        msg = (
            "iter {}: duration={:.4f}({:.4f})s average={:.4f}s "
            "avg_speed={} time={:.4f}s"
        ).format(
            i,
            cur_time,
            func.get_prev_exec_time(),
            total_time / (i + 1),
            avg_speed_txt,
            total_time,
        )
        if args.calc_output_rms:
            rms = []
            for v in retval:
                rms.append("{:.3g}".format(float(((v ** 2).mean()) ** 0.5)))
            msg += " output_rms=[{}]".format(", ".join(rms))
        if logger.level > logging.INFO:
            print(msg)
        else:
            logger.info(msg)
        out_data = out_data + msg + '\n'
    
    if args.focused_nvprof:
        if get_device_count("gpu") < 1:
            logger.warning(
                "No cuda device detected. ``focused_nvprof`` will be ignored."
            )
        else:
            try:
                import pycuda.driver as D
                
                D.start_profiler()
                func.execute()
                func.wait()
                D.stop_profiler()
            except ImportError:
                logger.error("`focused_nvprof need pycuda`", exc_info=True)
    
    if args.profile:
        with open(args.profile, "w") as fout:
            fout.write(profiler.get())
    
    return avg_speed, out_data


class Params:
    batchsize = None
    calc_output_rms = False
    device = None
    dump_cpp_model = None
    embed_input = False
    enable_chwn4 = False
    enable_fuse_conv_bias_nonlinearity = False
    enable_fuse_conv_bias_with_z = False
    enable_hwcd4 = False
    enable_io16xc32 = False
    enable_ioc16 = False
    enable_nchw32 = False
    enable_nchw4 = False
    enable_nchw44 = False
    enable_nchw44_dot = False
    enable_nchw88 = False
    fast_run = False
    focused_nvprof = False
    input_desc = None
    iter = 1
    load_input_data = None
    log = None
    net = None
    optimize_for_inference = False
    output_name = None
    profile = None
    reproducible = False
    rng = None
    seed = 0
    verbose = False
    warm_up = False


def run(in_net, in_data, in_iter):
    args = Params()
    
    ###
    # args.net = 'resnet50.mge'
    # args.load_input_data = 'data.pkl'
    # args.iter = 2
    
    args.net = in_net
    args.load_input_data = in_data
    args.iter = in_iter
    
    args.calc_output_rms = True
    
    ###
    
    if args.verbose:
        enable_debug_log()
    if args.log:
        set_log_file(args.log)
    
    if args.device:
        set_default_device(args.device)
    
    if args.dump_cpp_model:
        args.embed_input = True
    
    logger.info("loading model ...")
    graph, _, output_vars = G.load_graph(args.net)
    input_vars = tools.get_dep_vars(output_vars, "Host2DeviceCopy")
    
    if args.output_name is not None:
        output_vars = tools.find_vars_by_name(output_vars, args.output_name)
    
    data = make_data_given_desc(args, input_vars)
    
    _, out = run_model_(args, graph, input_vars, output_vars, data)
    
    return out


if __name__ == '__main__':
    run(in_net='/Users/lyleshaw/Work/summer-code/summer-of-code-meg/storage/file/5258fb4036b3436eac1ed79b2e8c7db7_resnet50.mge', in_data='/Users/lyleshaw/Work/summer-code/summer-of-code-meg/storage/file/cdceaeb16f844c4993c355c925e36adc_data.pkl', in_iter=1)
