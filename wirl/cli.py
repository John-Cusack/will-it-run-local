"""Command line interface."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from . import (compat, doctor, drafters, emit, gguf, membw, model, predict,
               probe, recommend, report, search, tune)
from .lock import BenchmarkBusy, benchmark_lock, foreign_gpu_users
from .report import c, gb, gib, head, para
from . import runner
from .runner import RunConfig, find_server


# --------------------------------------------------------------------------
# helpers shared by several commands
# --------------------------------------------------------------------------

def _bandwidth(args) -> tuple:
    """Return (bytes_per_sec, description). Measures unless told otherwise."""
    if args.mem_bandwidth:
        return args.mem_bandwidth * 1e9, f"{args.mem_bandwidth:.0f} GB/s (given on command line)"
    print("  measuring memory bandwidth (a few seconds)...", flush=True)
    r = membw.measure("stream", reps=args.bw_reps)
    warn = ""
    if r.spread_pct > 10:
        warn = c(f"  [unstable: {r.spread_pct:.0f}% spread across repetitions]", report.YEL)
    print(f"    stream: {r.best:.1f} GB/s peak, {r.median:.1f} median "
          f"({r.threads} threads, {r.gib} GiB){warn}")
    return r.best * 1e9, f"{r.best:.1f} GB/s (measured, {r.method})"


def _gpu_choice(args):
    gpus = probe.gpu_info()
    if not gpus and args.gpu == 0:
        return None, 0, 0.0
    g = next((g for g in gpus if g["index"] == args.gpu), None)
    if g is None:
        available = ", ".join(f"{g['index']}: {g['name']}" for g in gpus) or "none"
        sys.exit(f"error: GPU index {args.gpu} is not available; available GPUs: {available}")
    budget = args.vram * (1 << 30) if args.vram else g["vram_total"]
    bw = probe.gpu_bandwidth_hint(g) or 500e9
    return g, budget, bw


def _load(path, label="model"):
    try:
        g = gguf.read(path)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    except ValueError as e:
        sys.exit(f"error reading {label}: {e}")
    return g, model.build(g)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_probe(args):
    head("CPU")
    cpu = probe.cpu_info()
    print(f"  {cpu['model']}")
    print(f"  {cpu['physical']} physical cores / {cpu['logical']} logical, "
          f"{cpu['sockets']} socket(s), {cpu['numa_nodes']} NUMA node(s)")
    isa = ", ".join(k for k, v in cpu["isa"].items() if v) or "none detected"
    print(f"  ISA: {isa}")
    print(f"  governor: {cpu['governor']}")

    head("Memory")
    m = probe.mem_info()
    print(f"  {gib(m['total']):.1f} GiB total, {gib(m['available']):.1f} GiB available")
    print(f"  swap: {gib(m['swap_used']):.2f} GiB used of {gib(m['swap_total']):.1f} GiB")
    if m["edac_total"]:
        print(f"  EDAC reports {gib(m['edac_total']):.0f} GiB physically installed")

    head("GPU")
    gpus = probe.gpu_info()
    if not gpus:
        print("  none detected (CPU-only inference)")
    for g in gpus:
        bw = probe.gpu_bandwidth_hint(g)
        print(f"  [{g['index']}] {g['name']}  {gib(g['vram_total']):.1f} GiB "
              f"({gib(g['vram_free']):.1f} free), CC {g['compute_cap']}, "
              f"driver {g['driver']}")
        print(f"      PCIe gen{g['pcie_gen']} x{g['pcie_width']}"
              + (f", ~{bw/1e9:.0f} GB/s VRAM bandwidth" if bw else ""))

    if not args.no_bandwidth:
        head("Measured memory bandwidth")
        para("Nameplate figures assume every channel is populated and clocked "
             "at spec. This is what the machine actually delivers, and it is "
             "the number that determines decode speed.")
        print()
        for mode in ("stream", "gather"):
            r = membw.measure(mode, reps=args.bw_reps)
            note = "sequential (optimistic bound)" if mode == "stream" \
                else "random 64B (pessimistic bound)"
            print(f"  {mode:7} {r.best:6.1f} GB/s peak  {r.median:6.1f} median  "
                  f"spread {r.spread_pct:4.1f}%   {note}")


def cmd_inspect(args):
    g, mc = _load(args.model)
    head(f"{os.path.basename(g.path)}")
    shards = g.kv.get("_wirl.shards")
    print(f"  architecture   {g.arch}")
    print(f"  size           {gb(mc.total_bytes):.2f} GB"
          + (f" across {shards} shards" if shards else ""))
    print(f"  layers         {mc.n_layer}")
    print(f"  embedding      {g.n_embd}")
    if mc.is_moe:
        print(f"  experts        {mc.n_expert} routed, {mc.n_expert_used} active per token "
              f"({mc.n_expert_used / mc.n_expert * 100:.1f}%)")
    ctxlen = g.a("context_length")
    if ctxlen:
        print(f"  trained ctx    {ctxlen:,}")

    head("Quantisation mix")
    for t, b in list(g.quant_mix().items())[:8]:
        print(f"  {t:9} {gb(b):8.2f} GB  {b / mc.total_bytes * 100:5.1f}%")

    head("Decode cost")
    para("Generating one token reads every weight it needs exactly once. For a "
         "dense model that is the whole model. For a mixture of experts only "
         "the routed experts that fire are read, which is why a large MoE can "
         "decode faster than a much smaller dense model.")
    print()
    bpt = mc.bytes_per_token
    print(f"  bytes per token   {gb(bpt):.3f} GB "
          f"({bpt / mc.total_bytes * 100:.1f}% of the file)")
    if mc.is_moe:
        el = mc.layers[len(mc.layers) // 2]
        print(f"  per layer         {gb(el.expert_bytes):.3f} GB of experts resident, "
              f"{el.expert_bytes_per_tok / 1e6:.0f} MB read per token")
        print(f"                    {el.other_bytes / 1e6:.0f} MB attention/shared/norms, "
              "read every token")
    print(f"  KV cache @ {args.ctx:<6} {gib(model.kv_cache_bytes(g, args.ctx)):.2f} GiB (f16)")

    head("Roofline")
    para("Decode speed is bounded by how fast the bytes above can be read. "
         "These are ceilings, not forecasts -- real throughput lands somewhat "
         "below them.")
    print()
    for name, bw in (("DDR4-3200 8ch (ideal)", 200e9), ("typical server DRAM", 100e9),
                     ("typical desktop DRAM", 50e9), ("RTX 3090 VRAM", 936e9)):
        print(f"  {name:24} {bw / 1e9:6.0f} GB/s -> {bw / bpt:6.1f} tok/s "
              "if entirely resident")
    if not args.no_bandwidth:
        bw, desc = _bandwidth(args)
        print(f"  {'this machine (measured)':24} {bw / 1e9:6.0f} GB/s -> "
              f"{bw / bpt:6.1f} tok/s if entirely in RAM")


def cmd_plan(args):
    g, mc = _load(args.model)
    dg = dmc = None
    if args.draft:
        dg, dmc = _load(args.draft, "draft model")
        r = compat.compare(g.vocab_sig(), dg.vocab_sig())
        if not r["compatible"]:
            print(c("  draft model is NOT compatible:", report.RED))
            for p in r["problems"]:
                report.para(p)
            sys.exit(1)

    gpu, budget, bw_gpu = _gpu_choice(args)
    bw_cpu, bw_desc = _bandwidth(args)

    head("Inputs")
    print(f"  model            {os.path.basename(g.path)}  {gb(mc.total_bytes):.1f} GB")
    if dg:
        print(f"  draft            {os.path.basename(dg.path)}  {gb(dmc.total_bytes):.1f} GB")
    print(f"  context          {args.ctx:,}")
    print(f"  RAM bandwidth    {bw_desc}")
    if gpu:
        print(f"  GPU              {gpu['name']}, budget {gib(budget):.1f} GiB, "
              f"~{bw_gpu / 1e9:.0f} GB/s")
    else:
        print("  GPU              none -- CPU only")

    if not gpu:
        tps = predict.cpu_only_tps(mc, bw_cpu)
        head("Prediction")
        print(f"  CPU only: ~{tps:.1f} tok/s ceiling")
        para("Real throughput will be well below this; on the reference machine "
             "a CPU-only run reached about half its roofline because router and "
             "attention overhead stop being hidden once nothing is on a GPU.")
        return

    # A dense model has no routed experts, so --n-cpu-moe does nothing at all.
    # The knob there is --n-gpu-layers.
    if mc.is_moe:
        knob = "--n-cpu-moe"
        pts = predict.curve(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                            draft_mc=dmc, draft_g=dg)
        best, _ = predict.best_fit(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                                   headroom=args.headroom * 1e9,
                                   draft_mc=dmc, draft_g=dg)
    else:
        knob = "--n-gpu-layers"
        if dg:
            print(c("  note: draft-model VRAM is not modelled for dense offload; "
                    "subtract it from --vram yourself.", report.YEL))
        pts = predict.dense_curve(mc, g, bw_cpu, bw_gpu, budget, args.ctx)
        best, _ = predict.best_fit_dense(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                                         headroom=args.headroom * 1e9)
        pts = list(reversed(pts))          # most-offloaded first

    head("Predicted configurations")
    print(f"  {knob:^13} tok/s     VRAM     fits    of which CPU")
    shown = 0
    for p in pts:
        if not p.fits and shown > 6:
            continue
        mark = "  <-- recommended" if best and p.n_cpu_moe == best.n_cpu_moe else ""
        fit = "yes" if p.fits else c("NO", report.RED)
        share = p.cpu_ms / (p.cpu_ms + p.gpu_ms) * 100 if (p.cpu_ms + p.gpu_ms) else 0
        print(f"  {p.n_cpu_moe:^13d} {p.tps:6.2f}  {gib(p.vram_bytes):6.2f} GiB  "
              f"{fit:>4}    {share:4.0f}%{mark}")
        shown += 1
        if shown > 12:
            break

    head("Verdict")
    for line in predict.verdict(mc, best, bw_cpu, knob=knob):
        report.para(line)
        print()
    if best:
        flags = (f"--n-gpu-layers 99 --n-cpu-moe {best.n_cpu_moe}" if mc.is_moe
                 else f"--n-gpu-layers {best.n_cpu_moe}")
        print(f"  Start here:  {flags} --ctx-size {args.ctx}")
        para(f"Predicted {best.tps:.1f} tok/s using {gib(best.vram_bytes):.1f} GiB of "
             f"{gib(budget):.1f} GiB VRAM, leaving "
             f"{gib(budget - best.vram_bytes):.1f} GiB of headroom.", indent="  ")
        print()
        para("These are predictions. Run `wirl tune` to confirm them by "
             "measurement before trusting them.", indent="  ")


def run_cfg_measure(cfg, binary, args):
    from .runner import run_config
    return run_config(cfg, binary, reps=args.reps, n_tokens=args.tokens,
                      log_dir=args.log_dir)


def cmd_auto(args):
    """Probe, check, sweep for real, emit. The whole job, measured."""
    binary = find_server(args.llama_server)
    if not binary:
        sys.exit("error: could not find llama-server. Pass --llama-server PATH "
                 "or set WIRL_LLAMA_SERVER.")
    g, mc = _load(args.model)
    dg = dmc = None
    if args.draft:
        dg, dmc = _load(args.draft, "draft model")
        comp = compat.compare(g.vocab_sig(), dg.vocab_sig())
        if not comp["compatible"]:
            print(c("  draft model is NOT compatible:", report.RED))
            for prob in comp["problems"]:
                report.para(prob)
            return 1

    # ---- 1. hardware -----------------------------------------------------
    head("1. Hardware")
    cpu = probe.cpu_info()
    gpu, budget, bw_gpu = _gpu_choice(args)
    print(f"  {cpu['model']}, {cpu['physical']} physical cores")
    if gpu:
        print(f"  {gpu['name']}, {gib(gpu['vram_total']):.1f} GiB VRAM")
    else:
        sys.exit("error: no GPU detected; `wirl auto` tunes a CPU+GPU split.")
    bw_cpu, bw_desc = _bandwidth(args)

    # ---- 2. pre-flight ---------------------------------------------------
    head("2. Pre-flight")
    ram_need = predict.min_ram_needed(mc, g, budget, args.ctx, args.cache_type_k,
                                      draft_mc=dmc, draft_g=dg)
    checks = doctor.run_all(ram_need=ram_need, cache_type_k=args.cache_type_k,
                            llama_server=args.llama_server, gpu_index=args.gpu)
    report.print_checks(checks, show_ok=False)
    blocking = [ch for ch in checks if ch.status == doctor.FAIL]
    if blocking and not args.force:
        print()
        para("Refusing to sweep with blocking problems outstanding: any number "
             "measured now would be wrong. Fix them, or pass --force.")
        return 1
    if not blocking:
        print("  nothing blocking.")

    # ---- 3. narrow the search space --------------------------------------
    head("3. Search space")
    para("The cost model is used here and nowhere else: to pick which "
         "configurations are worth launching. Everything reported below this "
         "point is measured.")
    print()
    # A dense model has no routed experts, so --n-cpu-moe does nothing:
    # bisecting it would launch several identical servers and call the
    # resulting noise a result. The knob there is --n-gpu-layers.
    moe = mc.is_moe
    knob = "--n-cpu-moe" if moe else "--n-gpu-layers"
    if moe:
        best_pred, _ = predict.best_fit(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                                        headroom=args.headroom * 1e9,
                                        draft_mc=dmc, draft_g=dg)
        start = best_pred.n_cpu_moe if best_pred else mc.n_layer
        lo = max(0, start - args.span - 2)
        hi = min(mc.n_layer, start + args.span)
    else:
        best_pred, _ = predict.best_fit_dense(mc, g, bw_cpu, bw_gpu, budget,
                                              args.ctx,
                                              headroom=min(args.headroom * 1e9, 1e9))
        start = best_pred.n_cpu_moe if best_pred else 0
        lo = max(0, start - args.span)
        hi = min(mc.n_layer, start + args.span + 2)
    print(f"  model is {'MoE' if moe else 'dense'}, {mc.n_layer} layers, "
          f"{gb(mc.bytes_per_token):.2f} GB read per token")
    if not dg:
        # Speculative decoding was worth +33% on the reference machine. Worth
        # a pointer, but not worth a surprise multi-GB download.
        print("  no draft model given -- `wirl find-draft <model>` searches for "
              "a compatible one")
    print(f"  predicted starting point: {knob} {start} "
          f"(bracketing {lo}..{hi} empirically)")

    base = RunConfig(model=g.path, ctx=args.ctx, threads=args.threads,
                     cache_type_k=args.cache_type_k,
                     gpu_uuid=gpu["uuid"],
                     draft_model=dg.path if dg else None,
                     draft_n_max=args.draft_n_max, port=args.port)
    log = search.SearchLog()

    try:
        with benchmark_lock(wait=args.wait):
            # ---- 4. find the real VRAM edge ------------------------------
            head("4. VRAM boundary (measured)")
            para("Bisecting by actually launching the server. A predicted "
                 "boundary is a guess about allocator behaviour; the real one "
                 "is wherever it stops starting.")
            print()
            finder = search.find_vram_edge if moe else search.find_ngl_edge
            edge = finder(base, binary, lo, hi, log=log, log_dir=args.log_dir)
            if edge is None:
                print()
                para("Nothing in that range started. Reduce --ctx-size, drop "
                     "the draft model, or use a smaller quantisation.")
                return 1
            print(f"  {'smallest' if moe else 'largest'} {knob} that starts: {edge}")

            # ---- 5. measure ----------------------------------------------
            head("5. Throughput (measured)")
            para(f"{args.reps} repetitions each, one configuration at a time, "
                 "spread reported.")
            print()
            results = search.measure_around(base, binary, edge, span=args.span,
                                            reps=args.reps, n_tokens=args.tokens,
                                            log=log, log_dir=args.log_dir,
                                            max_layer=mc.n_layer, moe=moe)
            if dg and args.depth_sweep:
                ok = [r for r in results if r.ok and r.samples]
                if ok:
                    head("6. Draft depth (measured)")
                    para("Shallow first, stopping as soon as depth stops "
                         "paying -- on sparse MoE it usually does not.")
                    print()
                    b2 = copy.copy(max(ok, key=lambda r: r.mean).config)
                    results += tune.sweep_draft_depth(
                        b2, binary, (1, 2, 3), reps=args.reps,
                        n_tokens=args.tokens, log_dir=args.log_dir)
            if args.thread_sweep:
                ok = [r for r in results if r.ok and r.samples]
                if ok:
                    head("7. Thread count (measured)")
                    para("Usually a null result on a bandwidth-bound model. "
                         "If it is flat, that is the finding.")
                    print()
                    b3 = copy.copy(max(ok, key=lambda r: r.mean).config)
                    counts = tune.thread_counts(cpu["physical"])
                    results += tune.sweep_threads(
                        b3, binary, counts, reps=args.reps,
                        n_tokens=args.tokens, log_dir=args.log_dir)
    except BenchmarkBusy as e:
        sys.exit(f"error: {e}")

    # ---- results ---------------------------------------------------------
    head("Measured results")
    _RESULTS_MARK = None
    print(tune.summarise(results))
    rep = tune.repeatability(results)
    if rep:
        print()
        print(rep)
    unstable = tune.flag_unstable(results)
    if unstable:
        print()
        print(c("  " + unstable[0], report.YEL))
        for line in unstable[1:]:
            print(f"  {line}")

    winner, why = search.pick_recommended(results, gpu["vram_total"],
                                          int(args.headroom * 1e9))
    if winner is None:
        sys.exit("no configuration completed successfully.")

    head("Recommended")
    print(f"  {winner.config.label()}  ->  {winner.mean:.2f} tok/s "
          f"(spread {winner.spread_pct:.1f}%), {winner.peak_vram >> 20} MiB peak VRAM")
    print(f"  {why}")
    print()

    # ---- phase C: what a real conversation feels like ---------------------
    depth_points = []
    if not args.no_depth:
        head("Long-context behaviour (measured)")
        para("Everything above used an almost-empty context. These run against "
             "one already-running server, so they cost seconds rather than "
             "another startup.")
        print()
        wcfg = copy.copy(winner.config)
        depths = [0] + [d for d in (2048, 8192, 16384, 32768)
                        if d <= args.ctx * 0.85]
        try:
            with benchmark_lock(wait=args.wait):
                with runner.server(wcfg, binary, log_dir=args.log_dir) as _:
                    depth_points = search.profile_depth(
                        wcfg, wcfg.host, wcfg.port, depths=depths,
                        gen_tokens=args.depth_tokens)
        except (BenchmarkBusy, runner.ServerFailed) as e:
            print(f"  skipped: {e}")
        if depth_points:
            print()
            print(search.summarise_depth(depth_points, args.ctx))

    # ---- second context length -------------------------------------------
    if args.ctx2:
        head(f"Same config at --ctx-size {args.ctx2} (measured)")
        para("A longer context costs VRAM for the KV cache, which can force "
             "more experts back onto the CPU. This measures whether it does.")
        print()
        c2 = copy.copy(winner.config)
        c2.ctx = args.ctx2
        try:
            with benchmark_lock(wait=args.wait):
                r2 = run_cfg_measure(c2, binary, args)
        except BenchmarkBusy as e:
            r2 = None
            print(f"  skipped: {e}")
        if r2 and r2.ok:
            d = (r2.mean - winner.mean) / winner.mean * 100
            print(f"  {r2.mean:.2f} tok/s (spread {r2.spread_pct:.1f}%), "
                  f"{r2.peak_vram >> 20} MiB peak VRAM  -> {d:+.1f}% vs "
                  f"ctx {args.ctx}")
            results.append(r2)
        elif r2:
            print(f"  does not run at ctx {args.ctx2}: {r2.error}")
            para(f"Stay at --ctx-size {args.ctx}.")

    if best_pred:
        delta = (winner.mean - best_pred.tps) / best_pred.tps * 100
        para(f"For reference the cost model predicted {best_pred.tps:.2f} tok/s "
             f"at {knob} {best_pred.n_cpu_moe}; measured is {delta:+.0f}% "
             "against that. The measured number is the one to trust.")
        print()
    for line in predict.verdict(mc, best_pred, bw_cpu, measured_tps=winner.mean, knob=knob):
        report.para(line)
        print()

    out = args.emit or "./run-llama.sh"
    cfg = copy.copy(winner.config)
    cfg.host, cfg.port = args.emit_host, args.emit_port
    notes = (f"Chosen by measurement: {winner.mean:.2f} tok/s over {args.reps} runs "
             f"(spread {winner.spread_pct:.1f}%), {winner.peak_vram >> 20} MiB peak VRAM.\n"
             f"VRAM boundary found empirically at --n-cpu-moe {edge}.\n"
             f"Measured RAM bandwidth at the time: {bw_cpu/1e9:.0f} GB/s.")
    emit.write(out, emit.launch_script(cfg, binary,
                                       cpus=probe.physical_core_cpus(),
                                       extra_comment=notes), 0o755)
    print(f"  wrote launch script: {out}")
    unit_path = out + ".service"
    emit.write(unit_path, emit.systemd_unit(
        os.path.abspath(out), f"llama.cpp - {os.path.basename(g.path)}", notes))
    print(f"  wrote systemd unit:  {unit_path}")
    return 0


def cmd_recommend(args):
    """Which quantisation of a model should this machine download?"""
    try:
        files = compat.list_gguf(args.repo)
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"error: could not list {args.repo}: {e}")
    if not files:
        sys.exit(f"error: no GGUF files found in {args.repo}")

    cands = recommend.group_shards(files)
    if args.filter:
        f = args.filter.lower()
        cands = [c for c in cands if f in c.name.lower()]
        if not cands:
            sys.exit(f"error: no quantisation matching '{args.filter}'")

    gpu, budget, bw_gpu = _gpu_choice(args)
    ram = (args.ram * (1 << 30)) if args.ram else probe.mem_info()["available"]
    bw_cpu, bw_desc = _bandwidth(args)

    head("Your machine")
    cpu = probe.cpu_info()
    print(f"  {cpu['physical']} cores, {gib(ram):.0f} GiB RAM available, "
          f"{bw_desc} memory bandwidth")
    if gpu:
        print(f"  {gpu['name']}, {gib(budget):.1f} GiB VRAM")
    else:
        print("  no GPU -- everything runs on the CPU")
        budget, bw_gpu = 0, 1.0

    head(f"{args.repo}")
    print(f"  reading headers for {len(cands)} quantisation(s) over HTTP Range "
          "(no model download)...")
    ready = []
    for cnd in cands:
        try:
            recommend.load_remote(args.repo, cnd)
        except Exception as e:                               # noqa: BLE001
            print(f"    {cnd.name}: could not read header ({e})")
            continue
        recommend.evaluate(cnd, bw_cpu, bw_gpu, budget, ram, args.ctx,
                           headroom=args.headroom * 1e9)
        ready.append(cnd)
    if not ready:
        sys.exit("error: could not read any model headers.")

    best, why = recommend.choose(ready, min_tps=args.min_tps)

    head("What you can run")
    print(f"  {'quant':<12} {'size':>9}  {'tok/s':>7}  {'where':<22} config")
    for cnd in ready:
        if not cnd.fits_at_all:
            print(f"  {cnd.name:<12} {gb(cnd.size):8.1f} GB  {'--':>7}  "
                  + c(cnd.note, report.RED))
            continue
        where = ("all on GPU" if cnd.fits_vram
                 else f"{gib(cnd.ram_needed):.1f} GiB in RAM")
        cfg = ("-ngl 99" if cnd.fits_vram else f"{cnd.knob} {cnd.knob_value}")
        mark = "  <-- " + why if best is cnd else ""
        if cnd.confidence == "ceiling":
            # GPU-bound: the roofline is a loose upper bound and this tool has
            # no measurements in that regime. Do not state a figure at all.
            speed = c(f"{'fast':>7}", report.GRN)
        else:
            # "~" marks a number whose GPU term is uncalibrated.
            txt = (f"{cnd.tps:7.1f}" if cnd.confidence == "calibrated"
                   else f"{'~' + format(cnd.tps, '.0f'):>7}")
            speed = c(txt, report.GRN if cnd.tps >= args.min_tps else report.YEL)
        print(f"  {cnd.name:<12} {gb(cnd.size):8.1f} GB  {speed}  {where:<22} {cfg}{mark}")

    runnable = [x for x in ready if x.fits_at_all]
    if any(x.confidence == "ceiling" for x in runnable):
        print()
        para('"fast" means decode is GPU-bound. This tool is calibrated for the '
             "CPU-offload regime and has no measurements for that one, so it "
             "reports no number rather than a roofline figure that would be "
             "several times too optimistic. Expect comfortably interactive.")
    if any(x.confidence == "mixed" for x in runnable):
        print()
        para('"~" marks a figure where a meaningful share of each token is read '
             "from the GPU. The CPU side is calibrated against measurement; the "
             "GPU side is an uncalibrated roofline, so treat these as optimistic.")

    head("Recommendation")
    if best is None:
        para("Nothing in this repository fits this machine. Look for a smaller "
             "model, or a more aggressive quantisation than this repo offers.")
        return 1
    if best.fits_vram or best.confidence in ("ceiling", "mixed"):
        where = ("It fits on the GPU, so it will be comfortably fast."
                 if best.fits_vram else
                 f"About {gib(best.ram_needed):.1f} GiB sits in system RAM; "
                 f"expect somewhat under {best.tps:.0f} tok/s.")
        para(f"Download {c(best.name, report.BOLD)} ({gb(best.size):.1f} GB). {where}")
    else:
        para(f"Download {c(best.name, report.BOLD)} ({gb(best.size):.1f} GB). "
             f"Predicted {best.tps:.1f} tok/s, with "
             f"{gib(best.ram_needed):.1f} GiB of it in system RAM.")
    print()
    print(f"  hf download {args.repo} --include '*{best.name}*' --local-dir ./models")
    print()
    para("Then run `wirl plan` on the downloaded file to confirm, and "
         "`wirl tune` to verify by measurement.")
    return 0


def cmd_find_draft(args):
    """Search for a speculative-decoding drafter and vet it, without downloading."""
    g, mc = _load(args.model)
    sig = g.vocab_sig()
    head("Target")
    print(f"  {os.path.basename(g.path)}")
    print(f"  arch {sig['arch']}, vocab {sig['n_vocab']}, sha {sig['vocab_sha256_16']}")

    head("Searching HuggingFace")
    print("  " + ", ".join(f'"{q}"' for q in drafters.candidate_queries(g)))
    cands = drafters.find_candidates(g)
    if not cands:
        para("No candidate repositories found. That is not proof none exist -- "
             "search only sees public repos with GGUF files.")
        return 1
    print(f"  {len(cands)} candidate repositories; checking headers over HTTP "
          "Range (no downloads)")

    head("Results")
    good, checked = [], 0
    for cnd in cands[:args.max_repos]:
        rows = drafters.vet(sig, cnd["repo"], max_files=args.max_files,
                            max_gb=args.max_gb)
        for r in rows:
            checked += 1
            if r.get("error"):
                continue
            tag = "purpose-built" if cnd["purpose_built"] else "sibling"
            if r["compatible"] and not r["warnings"]:
                print("  " + c("OK  ", report.GRN)
                      + f" {r['size']/1e9:6.2f} GB  {r['repo']}/{r['file']}")
                print(f"        arch {r['sig']['arch']}, vocab matches, {tag}")
                good.append(r)
            elif r["compatible"]:
                print("  " + c("WARN", report.YEL)
                      + f" {r['size']/1e9:6.2f} GB  {r['repo']}/{r['file']}")
                for w in r["warnings"]:
                    report.para(w, indent="        ")
                good.append(r)
            elif args.verbose:
                print("  " + c("no  ", report.RED)
                      + f" {r['size']/1e9:6.2f} GB  {r['repo']}/{r['file']}")
                for pr in r["problems"]:
                    report.para(pr, indent="        ")

    head("Summary")
    print(f"  checked {checked} files, {len(good)} could pair with this target")
    if not good:
        para("Nothing compatible found. Speculative decoding needs an identical "
             "vocabulary, which is rarer than it sounds; run without a draft "
             "model.")
        return 1
    best = min(good, key=lambda r: r["size"])
    print()
    para(f"Smallest compatible: {best['repo']}/{best['file']} "
         f"({best['size']/1e9:.2f} GB). A drafter must be small to pay for "
         "itself, so prefer the smallest that still drafts well.")
    print()
    print(f"  hf download {best['repo']} {best['file']} --local-dir ./drafts")
    print()
    para("Then pass it to `wirl auto --draft`, which will sweep draft depth "
         "and tell you whether it is actually helping on your hardware.")
    return 0


def cmd_doctor(args):
    _, budget, _ = _gpu_choice(args)
    ram_need = None
    if args.model:
        g, mc = _load(args.model)
        ram_need = predict.min_ram_needed(mc, g, budget, args.ctx, args.cache_type_k)
    head("Pre-flight checks")
    checks = doctor.run_all(ram_need=ram_need,
                            cache_type_k=args.cache_type_k,
                            path=os.path.dirname(args.model) if args.model else ".",
                            llama_server=args.llama_server,
                            server_log=args.server_log, gpu_index=args.gpu)
    report.print_checks(checks)
    fails = [c_ for c_ in checks if c_.status == "fail"]
    warns = [c_ for c_ in checks if c_.status == "warn"]
    print()
    print(f"  {len(checks) - len(fails) - len(warns)} ok, {len(warns)} warnings, "
          f"{len(fails)} blocking")
    return 1 if fails else 0


def cmd_check_draft(args):
    if args.repo:
        if not args.file:
            head(f"GGUF files in {args.repo}")
            for f in compat.list_gguf(args.repo):
                print(f"  {f['size'] / 1e9:7.2f} GB  {f['path']}")
            print()
            para("Re-run with --file <name> to vet one without downloading it.")
            return 0
        print(f"  fetching header of {args.repo}/{args.file} over HTTP Range...")
        r = compat.check_remote(args.target, args.repo, args.file)
    else:
        r = compat.check_local(args.target, args.draft)

    head("Draft compatibility")
    print(f"  target  {r['target']['arch']:12} vocab {r['target']['n_vocab']:>7} "
          f"sha {r['target']['vocab_sha256_16']}")
    print(f"  draft   {r['draft']['arch']:12} vocab {r['draft']['n_vocab']:>7} "
          f"sha {r['draft']['vocab_sha256_16']}")
    print()
    if r["compatible"]:
        print("  " + c("COMPATIBLE", report.GRN) + " -- vocabularies match exactly.")
    else:
        print("  " + c("NOT COMPATIBLE", report.RED))
        for p in r["problems"]:
            report.para(p, indent="    ")
    for w in r["warnings"]:
        print()
        print("  " + c("warning:", report.YEL))
        report.para(w, indent="    ")
    return 0 if r["compatible"] else 1


def cmd_tune(args):
    binary = find_server(args.llama_server)
    if not binary:
        sys.exit("error: could not find llama-server. Pass --llama-server PATH "
                 "or set WIRL_LLAMA_SERVER.")
    g, mc = _load(args.model)
    dg = dmc = None
    if args.draft:
        dg, dmc = _load(args.draft, "draft model")

    gpu, budget, bw_gpu = _gpu_choice(args)
    foreign = foreign_gpu_users(gpu_uuid=gpu["uuid"] if gpu else None)
    if foreign and not args.force:
        print(c("  refusing to benchmark: other processes hold GPU memory:", report.RED))
        for p in foreign:
            print(f"    {p['name']} (pid {p['pid']}, {p['vram_mb']} MB)")
        para("Timings measured beside other GPU work are not comparable, and "
             "VRAM fitting will be wrong. Stop them, or pass --force if you "
             "genuinely want a contended number.")
        return 1

    bw_cpu, _ = _bandwidth(args)
    moe = mc.is_moe
    knob = "--n-cpu-moe" if moe else "--n-gpu-layers"
    if moe:
        best, _ = predict.best_fit(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                                   headroom=args.headroom * 1e9,
                                   draft_mc=dmc, draft_g=dg)
    else:
        best, _ = predict.best_fit_dense(mc, g, bw_cpu, bw_gpu, budget, args.ctx,
                                         headroom=min(args.headroom * 1e9, 1e9))
    if best is None:
        sys.exit(f"error: no {knob} configuration fits this GPU at this context length.")

    start = best.n_cpu_moe
    lo, hi = max(0, start - args.span), min(mc.n_layer, start + args.span)
    cands = list(range(hi, lo - 1, -1) if moe else range(lo, hi + 1))

    base = RunConfig(model=g.path, ctx=args.ctx, threads=args.threads,
                     gpu_uuid=gpu["uuid"] if gpu else None,
                     draft_model=dg.path if dg else None,
                     draft_n_max=args.draft_n_max, port=args.port)

    head(f"Measured sweep: {knob}")
    para(f"Prediction says {start}. Sweeping {knob} {cands[0]} to {cands[-1]} to "
         "confirm, with " + str(args.reps) + " repetitions each.")
    print()
    try:
        with benchmark_lock(wait=args.wait):
            results = tune.sweep_offload(base, binary, cands, moe, reps=args.reps,
                                       n_tokens=args.tokens, log_dir=args.log_dir)
            if args.draft and args.depth_sweep:
                ok = [r for r in results if r.ok]
                if ok:
                    win = max(ok, key=lambda r: r.mean)
                    head("Measured sweep: --spec-draft-n-max")
                    para("Shallow first. On sparse MoE models deeper drafting "
                         "usually loses, so this stops as soon as it stops paying.")
                    print()
                    b2 = copy.copy(win.config)
                    results += tune.sweep_draft_depth(b2, binary, (1, 2, 3),
                                                      reps=args.reps,
                                                      n_tokens=args.tokens,
                                                      log_dir=args.log_dir)
    except BenchmarkBusy as e:
        sys.exit(f"error: {e}")

    head("Results")
    print(tune.summarise(results))
    rep = tune.repeatability(results)
    if rep:
        print()
        print(rep)
    unstable = tune.flag_unstable(results)
    if unstable:
        print()
        print(c("  " + unstable[0], report.YEL))
        for line in unstable[1:]:
            print(f"  {line}")

    ok = [r for r in results if r.ok and r.samples]
    if not ok:
        sys.exit("no configuration completed successfully.")
    win = max(ok, key=lambda r: r.mean)
    head("Winner")
    print(f"  {win.config.label()}  ->  {win.mean:.2f} tok/s, "
          f"{win.peak_vram / (1 << 20):.0f} MiB peak VRAM")
    print()
    for line in predict.verdict(mc, best, bw_cpu, measured_tps=win.mean, knob=knob):
        report.para(line)
        print()

    if args.emit:
        cfg = win.config
        cfg.host = args.emit_host
        cfg.port = args.emit_port
        notes = (f"Chosen by measurement: {win.mean:.2f} tok/s over {args.reps} runs "
                 f"(spread {win.spread_pct:.1f}%), {win.peak_vram >> 20} MiB peak VRAM.\n"
                 f"Roofline for this model on this machine was "
                 f"{best.tps:.2f} tok/s at {bw_cpu / 1e9:.0f} GB/s measured RAM bandwidth.")
        script = emit.launch_script(cfg, binary,
                                    cpus=probe.physical_core_cpus(),
                                    extra_comment=notes)
        p = emit.write(args.emit, script, 0o755)
        print(f"  wrote launch script: {p}")
        unit = emit.systemd_unit(os.path.abspath(p),
                                 f"llama.cpp - {os.path.basename(g.path)}", notes)
        up = emit.write(args.emit + ".service", unit)
        print(f"  wrote systemd unit:  {up}")
        para(f"Install with: cp {up} ~/.config/systemd/user/ && "
             "systemctl --user daemon-reload && systemctl --user enable --now "
             + os.path.basename(up), indent="  ")
    return 0


def cmd_bandwidth(args):
    head("Memory bandwidth")
    for mode in ("stream", "gather"):
        r = membw.measure(mode, threads=args.threads, gib=args.gib, reps=args.bw_reps)
        print(f"  {mode:7} {r.best:6.1f} GB/s peak   {r.median:6.1f} median   "
              f"spread {r.spread_pct:4.1f}%   ({r.threads} threads, {r.gib} GiB)")
        if r.spread_pct > 10:
            para(c("Repetitions disagree by more than 10%. Something else is "
                   "using the machine, or the memory subsystem is unstable. "
                   "Do not benchmark inference until this settles.", report.YEL))
    return 0


# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="wirl",
        description="Predict and tune local LLM inference from measured hardware limits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  wirl auto model.gguf --draft draft.gguf     # measure everything, emit a launcher
  wirl probe
  wirl inspect model.gguf
  wirl plan model.gguf --draft draft.gguf --ctx 16384
  wirl doctor --model model.gguf
  wirl recommend unsloth/Qwen3-30B-A3B-GGUF
  wirl check-draft model.gguf --repo someone/drafter-GGUF
  wirl tune model.gguf --draft draft.gguf --emit ./run-llama.sh
""")
    p.add_argument("--version", action="version", version=f"%(prog)s {__import__('wirl').__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def bw_opts(sp):
        sp.add_argument("--mem-bandwidth", type=float, metavar="GB_S",
                        help="skip measurement and use this figure")
        sp.add_argument("--bw-reps", type=int, default=3)
        sp.add_argument("--no-bandwidth", action="store_true",
                        help="skip bandwidth measurement entirely")

    def gpu_opts(sp):
        sp.add_argument("--gpu", type=int, default=0, help="GPU index")
        sp.add_argument("--vram", type=float, metavar="GIB",
                        help="override usable VRAM budget")

    sp = sub.add_parser("auto",
                        help="THE MAIN COMMAND: probe, check, sweep for real, emit a launcher")
    sp.add_argument("model")
    sp.add_argument("--draft")
    sp.add_argument("--ctx", type=int, default=16384)
    sp.add_argument("--threads", type=int)
    sp.add_argument("--cache-type-k", default="f16")
    sp.add_argument("--draft-n-max", type=int, default=1)
    sp.add_argument("--depth-sweep", action="store_true",
                    help="also sweep speculative draft depth")
    sp.add_argument("--thread-sweep", action="store_true",
                    help="also sweep thread count at the winner")
    sp.add_argument("--no-depth", action="store_true",
                    help="skip the long-context prefill/decode profile")
    sp.add_argument("--depth-tokens", type=int, default=120,
                    help="tokens to generate per long-context probe")
    sp.add_argument("--ctx2", type=int,
                    help="also measure the winner at a second context length")
    sp.add_argument("--span", type=int, default=2,
                    help="how many configs above the VRAM edge to measure")
    sp.add_argument("--reps", type=int, default=3,
                    help="repetitions per config (spread is reported)")
    sp.add_argument("--tokens", type=int, default=400)
    sp.add_argument("--headroom", type=float, default=3.0, metavar="GB")
    sp.add_argument("--port", type=int, default=38080)
    sp.add_argument("--llama-server")
    sp.add_argument("--log-dir")
    sp.add_argument("--emit", metavar="PATH", default=None)
    sp.add_argument("--emit-host", default="127.0.0.1")
    sp.add_argument("--emit-port", type=int, default=8080)
    sp.add_argument("--force", action="store_true",
                    help="sweep even with blocking problems (results will be wrong)")
    sp.add_argument("--wait", action="store_true")
    bw_opts(sp)
    gpu_opts(sp)
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("probe", help="report hardware, including measured bandwidth")
    bw_opts(sp)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("inspect", help="analyse a GGUF: size, experts, bytes/token")
    sp.add_argument("model")
    sp.add_argument("--ctx", type=int, default=16384)
    bw_opts(sp)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("plan",
                        help="ESTIMATE a configuration without running anything (prefer `auto`, which measures)")
    sp.add_argument("model")
    sp.add_argument("--draft", help="speculative decoding draft model")
    sp.add_argument("--ctx", type=int, default=16384)
    sp.add_argument("--headroom", type=float, default=3.0, metavar="GB",
                    help="VRAM to leave free (default 3 GB)")
    bw_opts(sp)
    gpu_opts(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("recommend",
                        help="which quantisation should THIS machine download?")
    sp.add_argument("repo", help="HuggingFace repo id, e.g. unsloth/Qwen3-30B-A3B-GGUF")
    sp.add_argument("--ctx", type=int, default=8192)
    sp.add_argument("--min-tps", type=float, default=5.0,
                    help="slowest acceptable speed (default 5 tok/s)")
    sp.add_argument("--filter", help="only consider quants matching this substring")
    sp.add_argument("--ram", type=float, metavar="GIB",
                    help="pretend this much system RAM is free "
                         "(for 'what if I upgraded' questions)")
    sp.add_argument("--headroom", type=float, default=3.0)
    bw_opts(sp)
    gpu_opts(sp)
    sp.set_defaults(func=cmd_recommend)

    sp = sub.add_parser("find-draft",
                        help="search for a speculative-decoding drafter and vet it")
    sp.add_argument("model")
    sp.add_argument("--max-repos", type=int, default=6)
    sp.add_argument("--max-files", type=int, default=4)
    sp.add_argument("--max-gb", type=float, default=None,
                    help="ignore candidate files larger than this")
    sp.add_argument("--verbose", action="store_true",
                    help="also show incompatible candidates and why")
    sp.set_defaults(func=cmd_find_draft)

    sp = sub.add_parser("doctor", help="check for conditions that silently ruin results")
    gpu_opts(sp)
    sp.add_argument("--ctx", type=int, default=16384)
    sp.add_argument("--model")
    sp.add_argument("--llama-server")
    sp.add_argument("--server-log",
                    help="llama-server log to scan for silently disabled flags")
    sp.add_argument("--cache-type-k", default="f16")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("check-draft",
                        help="verify a draft model pairs with a target, before downloading it")
    sp.add_argument("target")
    sp.add_argument("--draft", help="local draft GGUF")
    sp.add_argument("--repo", help="HuggingFace repo id")
    sp.add_argument("--file", help="file within the repo (omit to list)")
    sp.set_defaults(func=cmd_check_draft)

    sp = sub.add_parser("bandwidth", help="measure memory bandwidth only")
    sp.add_argument("--threads", type=int)
    sp.add_argument("--gib", type=int)
    sp.add_argument("--bw-reps", type=int, default=3)
    sp.set_defaults(func=cmd_bandwidth)

    sp = sub.add_parser("tune", help="measure real configurations and emit a launcher")
    sp.add_argument("model")
    sp.add_argument("--draft")
    sp.add_argument("--ctx", type=int, default=16384)
    sp.add_argument("--threads", type=int)
    sp.add_argument("--draft-n-max", type=int, default=1)
    sp.add_argument("--depth-sweep", action="store_true",
                    help="also sweep draft depth")
    sp.add_argument("--span", type=int, default=2,
                    help="how far either side of the prediction to sweep")
    sp.add_argument("--reps", type=int, default=3)
    sp.add_argument("--tokens", type=int, default=400)
    sp.add_argument("--headroom", type=float, default=3.0)
    sp.add_argument("--port", type=int, default=38080)
    sp.add_argument("--llama-server")
    sp.add_argument("--log-dir", default=None)
    sp.add_argument("--emit", metavar="PATH", help="write a launch script here")
    sp.add_argument("--emit-host", default="127.0.0.1")
    sp.add_argument("--emit-port", type=int, default=8080)
    sp.add_argument("--force", action="store_true",
                    help="benchmark even if the GPU is busy (results will be invalid)")
    sp.add_argument("--wait", action="store_true",
                    help="wait for the benchmark lock instead of failing")
    bw_opts(sp)
    gpu_opts(sp)
    sp.set_defaults(func=cmd_tune)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
