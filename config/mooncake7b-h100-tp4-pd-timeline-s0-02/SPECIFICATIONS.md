# Mooncake TP4 P/D timeline run

Run name: `mooncake7b-h100-tp4-pd-timeline-s0-02`.

llama-7b/FP16, H100 SXM, TP4, PP1, 4P+4D, 32 simulated GPUs.
Full shifted-v1 seed-0 input: 12,031 requests, 3,959 returning turns, 4,122,048 output tokens.
One Serial inf reference and Serial/PPD/MigrationTax at 450/128/50 GB/s: ten conditions.
Every native worker receives `--record-timeline`; per-request, worker-step and explicit MigrationTax transfer CSVs stay in its seed-0 output.
Independent bandwidth jobs share one resource-admitted queue; TP1, TP4 and TP8 execute in order.
The controller uses up to 12 logical CPUs, one CPU per job, an 8 GiB per-job cap and a 2 GiB host reserve.
Hardware, simulation capacity and input are copied from the corresponding previous TP experiment.
