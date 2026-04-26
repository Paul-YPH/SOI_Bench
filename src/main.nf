// Demo: main workflow for the SOI-Bench benchmark pipeline.
//
// Flow:
//   1. Load & preprocess each dataset (LoadDatasets)
//   2. Expand into (dataset × tool × param_combo) tuples
//   3. Route each tuple to the matching Run<Tool> process via branch
//   4. Evaluate all outputs (RunEvaluation)

include {
    RunBANKSY; RunCAST; RunCellCharter; RunCOSMOS; RunDeepST;
    RunGPSA; RunGraphST; RunINSPIRE; Runmoscot_NR; Runmoscot_R;
    RunNicheCompass_batch; RunNicheCompass_omics; RunPASTE; RunPASTE2;
    RunPRECAST; RunSANTO; RunscSLAT; RunSEDR; RunSMOPCA; RunSPACEL;
    RunSpaDo; RunspaMGCN; RunSpateo_R; RunSpateo_NR; RunSpatialGlue;
    RunspatiAlign; RunspaVAE_batch; RunspaVAE_omics; RunspCLUE; RunSPIRAL;
    RunSSGATE; RunSTADIA; RunSTAGATE; RunSTalign; RunSTAligner; RunSTAMP;
    RunstClinic; RunSTG3Net; RunSTitch3D; RunstMSA; RunTacos;
} from '../modules/run_methods.nf'
include { LoadDatasets  } from '../modules/load_datasets.nf'
include { RunEvaluation } from '../modules/run_evaluation.nf'

// ---------------------------------------------------------------------------
// Helper: serialise a Groovy map to a Python dict literal string so it can
// be embedded directly in an inline `python -c "..."` script.
// ---------------------------------------------------------------------------
class WorkflowHelpers {
    static String formatPythonDict(Map dict) {
        def items = dict.collect { key, value ->
            def formatted_value
            if (value instanceof String) {
                formatted_value = "'${value}'"
            } else if (value instanceof List) {
                formatted_value = "[${value.collect { it instanceof String ? "'$it'" : it }.join(', ')}]"
            } else {
                formatted_value = value
            }
            "'$key': $formatted_value"
        }.join(', ')
        return items
    }
}

// ---------------------------------------------------------------------------
// Optional: load pre-tuned best hyperparameters from a CSV
// (columns: tool, dataset, parameter, best_value).
// When present, a single-value param_space entry is overridden by the best value.
// ---------------------------------------------------------------------------
def best_params_map = [:]
def csvFile = file("${params.tools_py_dir}/best_params.csv")
if (csvFile.exists()) {
    csvFile.splitCsv(header: true).each { row ->
        if (!best_params_map.containsKey(row.tool))    best_params_map[row.tool] = [:]
        if (!best_params_map[row.tool].containsKey(row.dataset)) best_params_map[row.tool][row.dataset] = [:]
        best_params_map[row.tool][row.dataset][row.parameter] = row.best_value.isNumber() ? row.best_value.toInteger() : row.best_value
    }
}

// ---------------------------------------------------------------------------
// Main workflow
// ---------------------------------------------------------------------------
workflow {
    def effective_datasets = params.datasets ?: (params.selection_map ? params.selection_map.keySet().toList() : [])
    def effective_tools    = params.tools    ?: (params.selection_map ? params.selection_map.values().flatten().unique() : [])

    if (effective_datasets.isEmpty()) error("Missing required parameter: provide 'params.datasets' or 'params.selection_map'")
    if (effective_tools.isEmpty())    error("Missing required parameter: provide 'params.tools' or 'params.selection_map'")
    if (!params.param_space?.seed)    error("Missing required parameter: params.param_space.seed")
    if (!params.tool_config)          error("Missing required parameter: params.tool_config")

    // 1. Build (dataset, h5ad_file) channel
    input_ch = Channel
        .fromList(effective_datasets)
        .map { folder ->
            def f = file("${params.data_dir}/${folder}/adata.h5ad")
            if (!f.exists()) error("Data file not found: ${f}")
            tuple(folder, f)
        }

    // 2. Load & gene-filter datasets
    gene_ch   = Channel.fromList(params.param_space.gene)
    loaded_ch = LoadDatasets(input_ch.combine(gene_ch))

    // 3. Expand into benchmark tuples: (dataset, file, tool, py_dict_string, seed)
    tools_ch     = Channel.fromList(effective_tools)
    benchmark_ch = loaded_ch.combine(tools_ch)
        .flatMap { ds_obj, file, gene_mode, tool_obj ->
            def dataset = ds_obj.toString().trim()
            def tool    = tool_obj.toString().trim()

            // Skip blacklisted (dataset, tool) pairs
            if (params.failed_blacklist?.any { it.toString() == "${dataset}_${tool}" }) {
                log.info "Skipped: ${dataset}_${tool}"
                return []
            }

            // Respect optional per-dataset tool allow-list
            if (params.selection_map && params.selection_map[dataset] != null
                    && !params.selection_map[dataset].contains(tool)) {
                return []
            }

            // Build all (seed × knn × pcs) combinations for this tool
            def required_params = params.tool_config[tool] ?: []
            def seed_list = params.param_space.seed ?: [42]
            def tool_best = best_params_map.get(tool, [:]).get(dataset, [:])

            def resolve = { param ->
                required_params.contains(param)
                    ? (params.param_space.containsKey(param) && params.param_space[param].size() > 1
                        ? params.param_space[param]
                        : tool_best.containsKey(param) ? [tool_best[param]] : [null])
                    : [null]
            }

            def knn_list = resolve('knn')
            def pcs_list = resolve('pcs')

            return [seed_list, knn_list, pcs_list].combinations().collect { s, k, p ->
                def final_args = params.global_args ? params.global_args.clone() : [:]
                final_args['seed'] = s
                final_args['gene'] = gene_mode
                if (k != null) final_args['knn'] = k
                if (p != null) final_args['pcs'] = p
                tuple(dataset, file, tool, WorkflowHelpers.formatPythonDict(final_args), s)
            }
        }

    // 4. Route each tuple to the matching process
    //    Every tool needs its own branch label because each process has a
    //    different container image and resource label.
    def split_ch = benchmark_ch.branch {
            BANKSY:  it[2] == 'BANKSY'
            CAST:    it[2] == 'CAST'
            DeepST:  it[2] == 'DeepST'
            // ... one label per tool (42 total, same pattern)
            fallback: true
        }

    split_ch.fallback.subscribe { tuple ->
        log.error "CRITICAL ERROR: Tool '${tuple[2]}' not configured in the branch block."
        error("Unknown tool: ${tuple[2]}")
    }

    // 5. Dispatch and collect results
    def result_list = [
        RunBANKSY(split_ch.BANKSY),
        RunCAST(split_ch.CAST),
        RunDeepST(split_ch.DeepST),
        // ... RunXxx(split_ch.Xxx) for all remaining tools
    ]

    // 6. Evaluate
    RunEvaluation(Channel.empty().mix(*result_list))
}
