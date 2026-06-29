/**
 * qcs6490_classify.cpp
 *
 * Image classification on the QCS6490 NPU (Hexagon HTP v68) via the QNN C API.
 * Loads a pre-compiled Context-Binary (.bin), runs inference on a raw float32
 * input image, and prints the top-5 ImageNet class predictions.
 *
 * Target board : QCS6490 (Hexagon v68 HTP)
 * SDK          : QAIRT / QNN SDK 2.28+
 * Build        : See CMakeLists.txt
 *
 * Usage:
 *   ./qcs6490_classify \
 *       resnet50_qcs6490.bin \
 *       input_224x224.raw \
 *       imagenet_classes.txt
 */

#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>       // dlopen / dlsym
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// ── Helpers ──────────────────────────────────────────────────────────────────

#define QNN_CHECK(expr)                                                         \
    do {                                                                        \
        Qnn_ErrorHandle_t _e = (expr);                                          \
        if (_e != QNN_SUCCESS) {                                                \
            fprintf(stderr, "[QNN ERROR] %s:%d → 0x%x\n",                      \
                    __FILE__, __LINE__, (unsigned)_e);                          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Read a file into a byte vector
static std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); std::exit(1); }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

// Read ImageNet class labels (one per line)
static std::vector<std::string> read_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) labels.push_back(line);
    return labels;
}

// Top-K indices (descending)
static std::vector<int> top_k(const float* data, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b){ return data[a] > data[b]; });
    idx.resize(k);
    return idx;
}

// ── QNN backend loader ────────────────────────────────────────────────────────

struct QnnBackend {
    void*                 lib_handle  = nullptr;
    QNN_INTERFACE_VER_TYPE iface      = {};
    Qnn_BackendHandle_t   backend     = nullptr;
    Qnn_ContextHandle_t   context     = nullptr;
    Qnn_GraphHandle_t     graph       = nullptr;
    Qnn_ProfileHandle_t   profile     = nullptr;
};

static QnnInterface_t* get_qnn_interface(void* handle) {
    typedef Qnn_ErrorHandle_t (*GetInterfacesFn)(
        const QnnInterface_t***, uint32_t*);

    auto fn = reinterpret_cast<GetInterfacesFn>(dlsym(handle, "QnnInterface_getProviders"));
    if (!fn) { fprintf(stderr, "dlsym QnnInterface_getProviders failed\n"); std::exit(1); }

    const QnnInterface_t** providers = nullptr;
    uint32_t num = 0;
    fn(&providers, &num);

    for (uint32_t i = 0; i < num; ++i) {
        if (QNN_API_VERSION_MAJOR == providers[i]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
            return const_cast<QnnInterface_t*>(providers[i]);
        }
    }
    fprintf(stderr, "No compatible QNN provider found\n");
    std::exit(1);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model.bin> <input.raw> <labels.txt> [libQnnHtp.so]\n",
            argv[0]);
        return 1;
    }

    const std::string bin_path    = argv[1];
    const std::string input_path  = argv[2];
    const std::string labels_path = argv[3];
    // Default HTP backend library (adjust path if needed on-board)
    const std::string backend_lib = (argc > 4)
        ? argv[4]
        : "libQnnHtp.so";

    // ── 1. Load QNN HTP backend ───────────────────────────────────────────────
    printf("[1] Loading backend: %s\n", backend_lib.c_str());
    void* lib = dlopen(backend_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }

    QnnInterface_t* iface_ptr = get_qnn_interface(lib);
    auto& qnn = iface_ptr->QNN_INTERFACE_VER_NAME;  // version-specific accessor

    // ── 2. Initialise logging (optional, errors only) ─────────────────────────
    QnnLog_Callback_t log_cb = [](const char* fmt, QnnLog_Level_t lvl,
                                   uint64_t /*ts*/, va_list args) {
        if (lvl <= QNN_LOG_LEVEL_WARN) {
            vfprintf(stderr, fmt, args);
            fprintf(stderr, "\n");
        }
    };
    Qnn_LogHandle_t log_handle = nullptr;
    qnn.logCreate(log_cb, QNN_LOG_LEVEL_WARN, &log_handle);

    // ── 3. Create backend ─────────────────────────────────────────────────────
    printf("[2] Creating HTP backend ...\n");
    Qnn_BackendHandle_t backend_handle = nullptr;

    // HTP performance hint: BURST for lowest latency on QCS6490
    QnnHtpDevice_PerfInfrastructure_t perf_infra;
    memset(&perf_infra, 0, sizeof(perf_infra));

    const QnnBackend_Config_t* backend_configs[] = { nullptr };
    QNN_CHECK(qnn.backendCreate(log_handle, backend_configs, &backend_handle));

    // ── 4. Create device ──────────────────────────────────────────────────────
    Qnn_DeviceHandle_t device_handle = nullptr;
    QnnHtpDevice_CustomConfig_t htp_dev_cfg;
    htp_dev_cfg.option    = QNN_HTP_DEVICE_CONFIG_OPTION_SOC_INFO;
    htp_dev_cfg.socInfo.socId = 35;   // QCS6490

    QnnDevice_Config_t dev_cfg;
    dev_cfg.option          = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    dev_cfg.customConfig    = &htp_dev_cfg;

    const QnnDevice_Config_t* dev_cfgs[] = { &dev_cfg, nullptr };
    QNN_CHECK(qnn.deviceCreate(log_handle, dev_cfgs, &device_handle));

    // ── 5. Create context from pre-compiled Context-Binary ────────────────────
    printf("[3] Loading context binary: %s\n", bin_path.c_str());
    std::vector<char> bin_data = read_binary_file(bin_path);

    // Performance mode: BURST (sustained high clocks)
    QnnHtpContext_CustomConfig_t htp_ctx_perf;
    htp_ctx_perf.option = QNN_HTP_CONTEXT_CONFIG_OPTION_PERFORMANCE_MODE;
    htp_ctx_perf.perfMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_BURST_MODE;

    QnnContext_Config_t ctx_cfg;
    ctx_cfg.option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    ctx_cfg.customConfig = &htp_ctx_perf;

    const QnnContext_Config_t* ctx_cfgs[] = { &ctx_cfg, nullptr };

    Qnn_ContextHandle_t ctx_handle = nullptr;
    QNN_CHECK(qnn.contextCreateFromBinary(
        backend_handle,
        device_handle,
        ctx_cfgs,
        reinterpret_cast<const uint8_t*>(bin_data.data()),
        static_cast<uint64_t>(bin_data.size()),
        &ctx_handle,
        nullptr   // profile handle — pass nullptr to skip
    ));

    // ── 6. Retrieve the graph ─────────────────────────────────────────────────
    Qnn_GraphHandle_t graph_handle = nullptr;
    // Graph name = model filename without extension (as embedded by converter)
    QNN_CHECK(qnn.graphRetrieve(ctx_handle, "resnet50", &graph_handle));

    // ── 7. Inspect I/O tensors ────────────────────────────────────────────────
    uint32_t      num_inputs = 0, num_outputs = 0;
    Qnn_Tensor_t* input_tensors  = nullptr;
    Qnn_Tensor_t* output_tensors = nullptr;

    QNN_CHECK(qnn.graphGetInputs (graph_handle, &input_tensors,  &num_inputs));
    QNN_CHECK(qnn.graphGetOutputs(graph_handle, &output_tensors, &num_outputs));

    printf("[4] Graph I/O: %u input(s), %u output(s)\n", num_inputs, num_outputs);

    // Input: expect 1×3×224×224 float32 = 602112 bytes
    const size_t INPUT_BYTES  = 1 * 3 * 224 * 224 * sizeof(float);
    const size_t OUTPUT_ELEMS = 1000;   // ImageNet classes
    const size_t OUTPUT_BYTES = OUTPUT_ELEMS * sizeof(float);

    // ── 8. Allocate I/O buffers ───────────────────────────────────────────────
    std::vector<float> input_buf(1 * 3 * 224 * 224);
    std::vector<float> output_buf(OUTPUT_ELEMS);

    // Load raw input (float32 NCHW, ImageNet-preprocessed)
    printf("[5] Reading input: %s\n", input_path.c_str());
    {
        std::ifstream f(input_path, std::ios::binary);
        if (!f) { fprintf(stderr, "Cannot open input file\n"); return 1; }
        f.read(reinterpret_cast<char*>(input_buf.data()), INPUT_BYTES);
    }

    // Wire buffer pointers into the Qnn_Tensor_t structs
    // (QNN_TENSOR_SET_MEM_TYPE / clientBuf pattern for QAIRT 2.x)
    QNN_TENSOR_SET_MEM_TYPE(input_tensors[0],  QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_MEM_TYPE(output_tensors[0], QNN_TENSORMEMTYPE_RAW);

    Qnn_ClientBuffer_t in_cbuf  = { input_buf.data(),  (uint32_t)INPUT_BYTES  };
    Qnn_ClientBuffer_t out_cbuf = { output_buf.data(), (uint32_t)OUTPUT_BYTES };
    QNN_TENSOR_SET_CLIENT_BUF(input_tensors[0],  in_cbuf);
    QNN_TENSOR_SET_CLIENT_BUF(output_tensors[0], out_cbuf);

    // ── 9. Run inference ──────────────────────────────────────────────────────
    printf("[6] Running inference on HTP (NPU) ...\n");
    QNN_CHECK(qnn.graphExecute(
        graph_handle,
        input_tensors,  num_inputs,
        output_tensors, num_outputs,
        nullptr,   // profile
        nullptr    // signal
    ));

    // ── 10. Post-process: softmax + top-5 ────────────────────────────────────
    // Apply softmax
    float max_val = *std::max_element(output_buf.begin(), output_buf.end());
    float sum = 0.0f;
    for (auto& v : output_buf) { v = std::exp(v - max_val); sum += v; }
    for (auto& v : output_buf) { v /= sum; }

    auto labels = read_labels(labels_path);
    auto top5   = top_k(output_buf.data(), (int)OUTPUT_ELEMS, 5);

    printf("\n─── Top-5 Predictions ───────────────────────────────\n");
    for (int i = 0; i < 5; ++i) {
        int  idx  = top5[i];
        auto lbl  = (idx < (int)labels.size()) ? labels[idx] : "unknown";
        printf("  %d. [%4d] %-40s %.4f\n", i+1, idx, lbl.c_str(), output_buf[idx]);
    }

    // ── 11. Cleanup ───────────────────────────────────────────────────────────
    qnn.contextFree(ctx_handle, nullptr);
    qnn.deviceFree(device_handle);
    qnn.backendFree(backend_handle);
    qnn.logFree(log_handle);
    dlclose(lib);

    return 0;
}