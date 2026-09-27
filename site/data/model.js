window.MODEL_DATA = {
 "accelerators": {
  "l40s": {
   "achieved_bandwidth": 0.7,
   "hourly_rate": 0.99,
   "hourly_rate_provenance": "RunPod console, 2026-08-15, On-Demand, 1x L40S 48 GB (48 GB VRAM, 62 GB RAM, 16 vCPU)",
   "measured": false,
   "memory_bytes": 48000000000.0,
   "mfu": 0.45,
   "name": "NVIDIA L40S",
   "peak_bandwidth": 864000000000.0,
   "peak_flops": 362050000000000.0,
   "provenance": "prior, unvalidated; the spec-sheet card, kept for contrast"
  },
  "l40s-run1": {
   "achieved_bandwidth": 0.83,
   "hourly_rate": 0.99,
   "hourly_rate_provenance": "RunPod console, 2026-08-15, On-Demand, 1x L40S 48 GB (48 GB VRAM, 62 GB RAM, 16 vCPU)",
   "measured": true,
   "memory_bytes": 48000000000.0,
   "mfu": 0.439,
   "name": "NVIDIA L40S (run 1 coefficients)",
   "peak_bandwidth": 864000000000.0,
   "peak_flops": 362050000000000.0,
   "provenance": "measured (run 1, 2026-08-18); survived runs 2-3"
  },
  "mi300x": {
   "achieved_bandwidth": 0.7,
   "hourly_rate": 1.99,
   "hourly_rate_provenance": "AMD Developer Cloud console, 2026-09-27, on-demand, 1x MI300X; paid from credits",
   "measured": false,
   "memory_bytes": 206158430208,
   "mfu": 0.45,
   "name": "AMD Instinct MI300X",
   "peak_bandwidth": 5300000000000.0,
   "peak_flops": 1307000000000000.0,
   "provenance": "prior, unvalidated; MI300X run 1 measured 0.46 / 0.166 against it"
  },
  "mi300x-run1": {
   "achieved_bandwidth": 0.46,
   "hourly_rate": 1.99,
   "hourly_rate_provenance": "AMD Developer Cloud console, 2026-09-27, on-demand, 1x MI300X; paid from credits",
   "measured": true,
   "memory_bytes": 206158430208,
   "mfu": 0.166,
   "name": "AMD Instinct MI300X (run 1 coefficients)",
   "peak_bandwidth": 5300000000000.0,
   "peak_flops": 1307000000000000.0,
   "provenance": "measured (MI300X run 1, 2026-09-27); MI300X run 2, on another droplet, ran 21 % faster decode and 33 % faster prefill at the same server line"
  },
  "mi300x-run2": {
   "achieved_bandwidth": 0.57,
   "hourly_rate": 1.99,
   "hourly_rate_provenance": "AMD Developer Cloud console, 2026-09-27, on-demand, 1x MI300X; paid from credits",
   "measured": true,
   "memory_bytes": 206158430208,
   "mfu": 0.247,
   "name": "AMD Instinct MI300X (run 2 coefficients)",
   "peak_bandwidth": 5300000000000.0,
   "peak_flops": 1307000000000000.0,
   "provenance": "measured (MI300X run 2, 2026-09-27); a second droplet, not yet faced"
  }
 },
 "constants": {
  "FLOPS_PER_MAC": 2
 },
 "defaults": {
  "accelerator": "l40s-run1",
  "gpu_memory_utilization": 0.9,
  "hit_rate": 0.0,
  "kv_dtype_bytes": 2,
  "model": "qwen3-8b",
  "output_tokens": 200,
  "prompt_tokens": 4000
 },
 "generated_by": "bench/export_site_data.py",
 "interference": {
  "l40s": {
   "intercept_s": -0.00636,
   "provenance": "measured (run 1, 2026-08-18): TPOT p50 minus median ITL at c = 13/23/32, 4 000-token prompts, max_num_batched_tokens 2 048; its (1 - h) scaling confirmed by run 3, slope ratio 0.194 against 0.200",
   "slope_s_per_seat": 0.00164
  },
  "l40s-run1": {
   "intercept_s": -0.00636,
   "provenance": "measured (run 1, 2026-08-18): TPOT p50 minus median ITL at c = 13/23/32, 4 000-token prompts, max_num_batched_tokens 2 048; its (1 - h) scaling confirmed by run 3, slope ratio 0.194 against 0.200",
   "slope_s_per_seat": 0.00164
  }
 },
 "measured": {
  "mi300x_run1_seats": {
   "accelerator": "mi300x",
   "decode_step_through": 32,
   "gpu_memory_utilization": 0.9,
   "hit_rate": 0.0,
   "kv_dtype_bytes": 2,
   "levels": [
    {
     "concurrency": 1,
     "max_running": 1.0,
     "median_itl_ms": 6.6131833333524055,
     "output_throughput": 127.27590778068505,
     "p50_tpot_ms": 6.6054507587939115,
     "p99_tpot_ms": 6.610969881658292
    },
    {
     "concurrency": 8,
     "max_running": 8.0,
     "median_itl_ms": 9.717632833314838,
     "output_throughput": 345.90209553346557,
     "p50_tpot_ms": 18.812358983249737,
     "p99_tpot_ms": 24.352873944271373
    },
    {
     "concurrency": 32,
     "max_running": 32.0,
     "median_itl_ms": 14.89725000000893,
     "output_throughput": 499.0702803604874,
     "p50_tpot_ms": 55.828281496649794,
     "p99_tpot_ms": 76.32667393127304
    },
    {
     "concurrency": 64,
     "max_running": 64.0,
     "median_itl_ms": 134.3990361666935,
     "output_throughput": 501.71371482709395,
     "p50_tpot_ms": 113.96066354103843,
     "p99_tpot_ms": 143.63443870284775
    },
    {
     "concurrency": 128,
     "max_running": 101.0,
     "median_itl_ms": 168.07005250007023,
     "output_throughput": 511.20168902439354,
     "p50_tpot_ms": 174.5736918065322,
     "p99_tpot_ms": 237.263182959874
    },
    {
     "concurrency": 192,
     "max_running": 101.0,
     "median_itl_ms": 169.12176849996285,
     "output_throughput": 524.6057752357567,
     "p50_tpot_ms": 174.5235888856783,
     "p99_tpot_ms": 236.91245765221032
    },
    {
     "concurrency": 224,
     "max_running": 101.0,
     "median_itl_ms": 169.71287775004384,
     "output_throughput": 529.1154715081127,
     "p50_tpot_ms": 174.8152450062817,
     "p99_tpot_ms": 236.84921779057714
    },
    {
     "concurrency": 240,
     "max_running": 100.0,
     "median_itl_ms": 169.80597824988308,
     "output_throughput": 530.4743833140576,
     "p50_tpot_ms": 174.96888210929689,
     "p99_tpot_ms": 237.3013777694979
    },
    {
     "concurrency": 256,
     "max_running": 101.0,
     "median_itl_ms": 170.06478250027612,
     "output_throughput": 531.7309293685132,
     "p50_tpot_ms": 175.1174055690953,
     "p99_tpot_ms": 237.53040764007395
    },
    {
     "concurrency": 288,
     "max_running": 100.0,
     "median_itl_ms": 170.17858824988252,
     "output_throughput": 534.1195098551223,
     "p50_tpot_ms": 175.1790417211044,
     "p99_tpot_ms": 237.63836793268786
    }
   ],
   "max_num_batched_tokens": 2048,
   "mode": "closed",
   "output_tokens": 200,
   "prompt_tokens": 4000,
   "run": 4
  },
  "model": "qwen3-8b",
  "run1_decode": {
   "accelerator": "l40s",
   "gpu_memory_utilization": 0.9,
   "hit_rate": 0.0,
   "kv_dtype_bytes": 2,
   "levels": [
    {
     "asked": 1,
     "decode_step_ms": 22.700132976751775,
     "output_tps": 41.119949338832164,
     "ran": 1,
     "tpot_p50_ms": 22.678319221168252,
     "tpot_p99_ms": 22.708936884218886,
     "ttft_p50_ms": 350.1494500087574
    },
    {
     "asked": 4,
     "decode_step_ms": 24.73115746397525,
     "output_tps": 144.94645308340657,
     "ran": 4,
     "tpot_p50_ms": 25.708177477067533,
     "tpot_p99_ms": 26.69623469693432,
     "ttft_p50_ms": 391.718806931749
    },
    {
     "asked": 4,
     "decode_step_ms": 26.32447355426848,
     "output_tps": 123.5016220736211,
     "ran": 4,
     "tpot_p50_ms": 29.028083329397866,
     "tpot_p99_ms": 30.52787265427944,
     "ttft_p50_ms": 674.8048434965312
    },
    {
     "asked": 4,
     "decode_step_ms": 29.52285303035751,
     "output_tps": 93.56734125149508,
     "ran": 4,
     "tpot_p50_ms": 36.10368417581805,
     "tpot_p99_ms": 38.67986823669658,
     "ttft_p50_ms": 1313.1662944797426
    },
    {
     "asked": 8,
     "decode_step_ms": 29.649760981556028,
     "output_tps": 192.67230435781798,
     "ran": 8,
     "tpot_p50_ms": 37.80383068071351,
     "tpot_p99_ms": 39.47517830304609,
     "ttft_p50_ms": 682.9211430158466
    },
    {
     "asked": 13,
     "decode_step_ms": 33.798288903199136,
     "output_tps": 244.2394355000919,
     "ran": 13,
     "tpot_p50_ms": 48.62432155773352,
     "tpot_p99_ms": 50.874759782276506,
     "ttft_p50_ms": 895.8767229923978
    },
    {
     "asked": 14,
     "decode_step_ms": 34.600860031787306,
     "output_tps": 251.00998761509712,
     "ran": 14,
     "tpot_p50_ms": 51.36181980672166,
     "tpot_p99_ms": 53.49323368770534,
     "ttft_p50_ms": 728.1322475173511
    },
    {
     "asked": 16,
     "decode_step_ms": 36.30061796866357,
     "output_tps": 264.7621172383324,
     "ran": 16,
     "tpot_p50_ms": 55.63711288213102,
     "tpot_p99_ms": 58.010175573258145,
     "ttft_p50_ms": 916.4518705219962
    },
    {
     "asked": 23,
     "decode_step_ms": 43.06510096648708,
     "output_tps": 293.5843293167354,
     "ran": 23,
     "tpot_p50_ms": 73.87099769633338,
     "tpot_p99_ms": 75.7564908450577,
     "ttft_p50_ms": 766.8764510308392
    },
    {
     "asked": 24,
     "decode_step_ms": 43.9562980318442,
     "output_tps": 296.47896714992595,
     "ran": 24,
     "tpot_p50_ms": 76.06817011302572,
     "tpot_p99_ms": 78.23350454090355,
     "ttft_p50_ms": 776.9883330329321
    },
    {
     "asked": 32,
     "decode_step_ms": 50.68951047724113,
     "output_tps": 314.93908848672277,
     "ran": 32,
     "tpot_p50_ms": 96.74075308774164,
     "tpot_p99_ms": 98.72904807329178,
     "ttft_p50_ms": 826.0087160160765
    },
    {
     "asked": 45,
     "decode_step_ms": 59.65597496833652,
     "output_tps": 311.61157422182345,
     "ran": 41,
     "tpot_p50_ms": 122.9878038415994,
     "tpot_p99_ms": 132.56605936757802,
     "ttft_p50_ms": 3078.4194045118056
    }
   ],
   "max_num_batched_tokens": 2048,
   "mode": "closed",
   "output_tokens": 200,
   "prompt_tokens": 4000,
   "run": 1
  },
  "run2_rates": {
   "accelerator": "l40s",
   "gpu_memory_utilization": 0.9,
   "hit_rate": 0.0,
   "kv_dtype_bytes": 2,
   "levels": [
    {
     "goodput": 0.72,
     "output_tps": 160.88,
     "peak_concurrency": 9,
     "rate": 1.0,
     "request_throughput": 0.8,
     "tpot_p50_ms": 26.87,
     "tpot_p99_ms": 28.18,
     "ttft_p99_ms": 401.7
    },
    {
     "goodput": 0.48,
     "output_tps": 96.03,
     "peak_concurrency": 8,
     "rate": 0.5,
     "request_throughput": 0.48,
     "tpot_p50_ms": 25.29,
     "tpot_p99_ms": 27.89,
     "ttft_p99_ms": 272.86
    },
    {
     "goodput": 0.95,
     "output_tps": 192.22,
     "peak_concurrency": 16,
     "rate": 1.0,
     "request_throughput": 0.96,
     "tpot_p50_ms": 27.34,
     "tpot_p99_ms": 33.37,
     "ttft_p99_ms": 314.49
    },
    {
     "goodput": 1.38,
     "output_tps": 287.25,
     "peak_concurrency": 22,
     "rate": 1.5,
     "request_throughput": 1.44,
     "tpot_p50_ms": 31.02,
     "tpot_p99_ms": 37.06,
     "ttft_p99_ms": 375.3
    },
    {
     "goodput": 1.71,
     "output_tps": 382.59,
     "peak_concurrency": 29,
     "rate": 2.0,
     "request_throughput": 1.91,
     "tpot_p50_ms": 35.29,
     "tpot_p99_ms": 43.13,
     "ttft_p99_ms": 437.77
    },
    {
     "goodput": 1.87,
     "output_tps": 471.88,
     "peak_concurrency": 37,
     "rate": 2.5,
     "request_throughput": 2.36,
     "tpot_p50_ms": 41.06,
     "tpot_p99_ms": 51.22,
     "ttft_p99_ms": 502.82
    },
    {
     "goodput": 0.9,
     "output_tps": 567.24,
     "peak_concurrency": 57,
     "rate": 3.0,
     "request_throughput": 2.84,
     "tpot_p50_ms": 54.17,
     "tpot_p99_ms": 67.02,
     "ttft_p99_ms": 626.94
    },
    {
     "goodput": 0.12,
     "output_tps": 722.77,
     "peak_concurrency": 111,
     "rate": 4.0,
     "request_throughput": 3.61,
     "tpot_p50_ms": 102.6,
     "tpot_p99_ms": 126.96,
     "ttft_p99_ms": 1723.72
    }
   ],
   "max_num_batched_tokens": 2048,
   "mode": "poisson",
   "output_tokens": 200,
   "prompt_tokens": 1500,
   "run": 2
  },
  "run3_rates": {
   "accelerator": "l40s",
   "gpu_memory_utilization": 0.9,
   "hit_rate": 0.8,
   "kv_dtype_bytes": 2,
   "levels": [
    {
     "max_running": 13.0,
     "max_waiting": 0.0,
     "output_throughput": 287.04242968289515,
     "rate": 1.5,
     "request_goodput": 1.4367020616636084,
     "request_throughput": 1.4367020616636084,
     "tpot_p50_ms": 25.020124063120416,
     "ttft_p99_ms": 109.40550420433281
    },
    {
     "max_running": 17.0,
     "max_waiting": 0.0,
     "output_throughput": 377.9861581256062,
     "rate": 2.0,
     "request_goodput": 1.8901408062731724,
     "request_throughput": 1.8901408062731724,
     "tpot_p50_ms": 25.542373698785077,
     "ttft_p99_ms": 133.34566250443467
    },
    {
     "max_running": 21.0,
     "max_waiting": 0.0,
     "output_throughput": 416.3289128997851,
     "rate": 2.5,
     "request_goodput": 2.0822923887976628,
     "request_throughput": 2.0822923887976628,
     "tpot_p50_ms": 25.5874848114935,
     "ttft_p99_ms": 125.79235970973967
    },
    {
     "max_running": 25.0,
     "max_waiting": 0.0,
     "output_throughput": 555.3068386362313,
     "rate": 3.0,
     "request_goodput": 2.777614376549815,
     "request_throughput": 2.777614376549815,
     "tpot_p50_ms": 26.61936840126712,
     "ttft_p99_ms": 140.5465550161898
    },
    {
     "max_running": 36.0,
     "max_waiting": 0.0,
     "output_throughput": 730.4619611553586,
     "rate": 4.0,
     "request_goodput": 3.6536291718666334,
     "request_throughput": 3.6536291718666334,
     "tpot_p50_ms": 29.011539665598367,
     "ttft_p99_ms": 143.25435269624003
    },
    {
     "max_running": 41.0,
     "max_waiting": 0.0,
     "output_throughput": 902.4903700046843,
     "rate": 5.0,
     "request_goodput": 4.514709204625735,
     "request_throughput": 4.514709204625735,
     "tpot_p50_ms": 31.41885002298121,
     "ttft_p99_ms": 153.8006310723722
    },
    {
     "max_running": 83.0,
     "max_waiting": 0.0,
     "output_throughput": 1161.7657621843325,
     "rate": 6.0,
     "request_goodput": 5.812434673172612,
     "request_throughput": 5.812434673172612,
     "tpot_p50_ms": 36.69499281101191,
     "ttft_p99_ms": 256.0456148907542
    },
    {
     "max_running": 69.0,
     "max_waiting": 0.0,
     "output_throughput": 1313.2495771394942,
     "rate": 7.0,
     "request_goodput": 6.573917456062877,
     "request_throughput": 6.573917456062877,
     "tpot_p50_ms": 40.47993703010543,
     "ttft_p99_ms": 238.6185875162484
    },
    {
     "max_running": 124.0,
     "max_waiting": 0.0,
     "output_throughput": 1662.9688046003937,
     "rate": 9.0,
     "request_goodput": 2.5880295502607145,
     "request_throughput": 8.351808508809476,
     "tpot_p50_ms": 59.26136657052753,
     "ttft_p99_ms": 325.17242746427655
    },
    {
     "max_running": 256.0,
     "max_waiting": 65.0,
     "output_throughput": 1814.8232964461572,
     "rate": 11.0,
     "request_goodput": 0.3029678451130552,
     "request_throughput": 9.089035353391656,
     "tpot_p50_ms": 118.27960232244664,
     "ttft_p99_ms": 7590.310124047101
    },
    {
     "max_running": 256.0,
     "max_waiting": 275.0,
     "output_throughput": 1828.9716373489102,
     "rate": 13.0,
     "request_goodput": 0.0,
     "request_throughput": 9.153268240897512,
     "tpot_p50_ms": 136.8247494803526,
     "ttft_p99_ms": 30611.66351197287
    }
   ],
   "max_num_batched_tokens": 2048,
   "mode": "poisson",
   "output_tokens": 200,
   "prompt_tokens": 1500,
   "run": 3
  },
  "run3_seats": {
   "accelerator": "l40s",
   "gpu_memory_utilization": 0.9,
   "kv_dtype_bytes": 2,
   "max_num_batched_tokens": 2048,
   "mode": "closed",
   "output_tokens": 200,
   "prompt_tokens": 4000,
   "run": 3,
   "series": [
    {
     "hit_rate": 0.0,
     "levels": [
      {
       "concurrency": 8.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 29.536533169448376,
       "output_throughput": 193.01988780165058,
       "p50_tpot_ms": 37.728461057846275,
       "p99_tpot_ms": 39.27890460910135,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 12.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 32.84799586981535,
       "output_throughput": 233.87204706630524,
       "p50_tpot_ms": 46.982302236834066,
       "p99_tpot_ms": 48.710224405399074,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 14.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 34.463612362742424,
       "output_throughput": 248.4080411725958,
       "p50_tpot_ms": 51.68155330378356,
       "p99_tpot_ms": 53.763266983044204,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 16.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 36.153769120574,
       "output_throughput": 260.411879583106,
       "p50_tpot_ms": 56.54570996930275,
       "p99_tpot_ms": 58.62848237390868,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 20.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 40.3556264936924,
       "output_throughput": 277.17419136316573,
       "p50_tpot_ms": 66.71629920040243,
       "p99_tpot_ms": 69.14818622724121,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 22.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 42.02086851000786,
       "output_throughput": 284.4972880259699,
       "p50_tpot_ms": 71.68690246253756,
       "p99_tpot_ms": 74.13793734401165,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 24.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 43.73655468225479,
       "output_throughput": 290.77468804388377,
       "p50_tpot_ms": 76.58539608064757,
       "p99_tpot_ms": 79.38680575364003,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 26.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 45.26264779269695,
       "output_throughput": 295.4438648316556,
       "p50_tpot_ms": 81.80237370519782,
       "p99_tpot_ms": 84.67145987266872,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 28.0,
       "measured_hit_rate": 0.0,
       "median_itl_ms": 47.205451875925064,
       "output_throughput": 301.8415436741826,
       "p50_tpot_ms": 86.22710907941084,
       "p99_tpot_ms": 89.70907165457339,
       "preemptions": 0.0,
       "request_goodput": 0.0
      }
     ]
    },
    {
     "hit_rate": 0.8,
     "levels": [
      {
       "concurrency": 8.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 25.424739345908165,
       "output_throughput": 284.2180906295321,
       "p50_tpot_ms": 26.28452434043271,
       "p99_tpot_ms": 27.20284555697037,
       "preemptions": 0.0,
       "request_goodput": 0.35534664383771464
      },
      {
       "concurrency": 12.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 27.364138513803482,
       "output_throughput": 382.1718114209024,
       "p50_tpot_ms": 28.86555926527959,
       "p99_tpot_ms": 30.790700949492123,
       "preemptions": 0.0,
       "request_goodput": 0.478046741179725
      },
      {
       "concurrency": 14.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 27.595202438533306,
       "output_throughput": 432.223605895129,
       "p50_tpot_ms": 29.718427628690574,
       "p99_tpot_ms": 31.649523998830634,
       "preemptions": 0.0,
       "request_goodput": 0.463096720601924
      },
      {
       "concurrency": 16.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 27.921663597226143,
       "output_throughput": 477.9141693264298,
       "p50_tpot_ms": 30.599287654923735,
       "p99_tpot_ms": 32.376641820386126,
       "preemptions": 0.0,
       "request_goodput": 0.29885200791647515
      },
      {
       "concurrency": 20.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 28.919425792992115,
       "output_throughput": 555.2294248626401,
       "p50_tpot_ms": 32.88359955496075,
       "p99_tpot_ms": 35.14225899985987,
       "preemptions": 0.0,
       "request_goodput": 0.3242080392090825
      },
      {
       "concurrency": 22.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 29.225319623947144,
       "output_throughput": 592.6166722926314,
       "p50_tpot_ms": 33.89173840177269,
       "p99_tpot_ms": 35.91250938349133,
       "preemptions": 0.0,
       "request_goodput": 0.26967766657230097
      },
      {
       "concurrency": 24.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 29.76326085627079,
       "output_throughput": 623.2814201870516,
       "p50_tpot_ms": 35.26527387319348,
       "p99_tpot_ms": 37.63436697170738,
       "preemptions": 0.0,
       "request_goodput": 0.12985029587230243
      },
      {
       "concurrency": 26.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 33.07456336915493,
       "output_throughput": 614.1255519380995,
       "p50_tpot_ms": 38.93416110501067,
       "p99_tpot_ms": 41.144251699311354,
       "preemptions": 0.0,
       "request_goodput": 0.2362172774939802
      },
      {
       "concurrency": 28.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 33.579638227820396,
       "output_throughput": 641.0984066229513,
       "p50_tpot_ms": 40.22599368744609,
       "p99_tpot_ms": 42.29941860402948,
       "preemptions": 0.0,
       "request_goodput": 0.2290318809013223
      },
      {
       "concurrency": 32.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 34.39267538487911,
       "output_throughput": 703.8654975458568,
       "p50_tpot_ms": 42.03953102099985,
       "p99_tpot_ms": 44.28606558188646,
       "preemptions": 0.0,
       "request_goodput": 0.21999234178648439
      },
      {
       "concurrency": 40.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 39.030191488564014,
       "output_throughput": 749.0516724336175,
       "p50_tpot_ms": 49.814388212097356,
       "p99_tpot_ms": 52.17015616109024,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 48.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 40.22933170199394,
       "output_throughput": 823.4478293736241,
       "p50_tpot_ms": 54.347815202292594,
       "p99_tpot_ms": 57.07884246015443,
       "preemptions": 0.0,
       "request_goodput": 0.0
      },
      {
       "concurrency": 56.0,
       "measured_hit_rate": 0.8,
       "median_itl_ms": 44.050367549061775,
       "output_throughput": 851.682703652274,
       "p50_tpot_ms": 61.61880795492898,
       "p99_tpot_ms": 64.05005710255637,
       "preemptions": 0.0,
       "request_goodput": 0.0
      }
     ]
    }
   ]
  },
  "seat_crossings": {
   "h0": 12.51049464744546,
   "h80": 37.79793888536536,
   "tpot_p99_target_ms": 50.0
  }
 },
 "models": {
  "qwen2.5-7b": {
   "head_dim": 128,
   "kv_bytes_per_token": 57344,
   "kv_dtype_bytes": 2,
   "measured": false,
   "name": "Qwen2.5-7B-Instruct",
   "num_kv_heads": 4,
   "num_layers": 28,
   "params_non_embedding": 6530000000.0,
   "params_total": 7610000000.0,
   "provenance": "config.json and the model card; never served or measured here, docs/audience.md",
   "served_by_this_stack": false,
   "weight_dtype_bytes": 2
  },
  "qwen3-8b": {
   "head_dim": 128,
   "kv_bytes_per_token": 147456,
   "kv_dtype_bytes": 2,
   "measured": true,
   "name": "Qwen3-8B",
   "num_kv_heads": 8,
   "num_layers": 36,
   "params_non_embedding": 6950000000.0,
   "params_total": 8200000000.0,
   "provenance": "config.json and the model card; docs/model-anatomy.md",
   "served_by_this_stack": true,
   "weight_dtype_bytes": 2
  }
 },
 "runs": {
  "1": {
   "card": "NVIDIA L40S",
   "date": "2026-08-18",
   "label": "run 1",
   "report": "docs/benchmarks/l40s-baseline.md"
  },
  "2": {
   "card": "NVIDIA L40S",
   "date": "2026-08-23",
   "label": "run 2",
   "report": "docs/benchmarks/l40s-run2.md"
  },
  "3": {
   "card": "NVIDIA L40S",
   "date": "2026-08-30",
   "label": "run 3",
   "report": "docs/benchmarks/l40s-run3.md"
  },
  "4": {
   "card": "AMD Instinct MI300X",
   "date": "2026-09-27",
   "label": "MI300X run 1",
   "report": "docs/benchmarks/mi300x-run1.md"
  },
  "5": {
   "card": "AMD Instinct MI300X",
   "date": "2026-09-27",
   "label": "MI300X run 2",
   "report": "docs/benchmarks/mi300x-run2.md"
  }
 },
 "slo_classes": {
  "batch": {
   "tpot_s": 0.2,
   "ttft_s": 3.0
  },
  "interactive": {
   "tpot_s": 0.05,
   "ttft_s": 0.3
  },
  "source": "docs/SLO.md#2-targets"
 }
};
