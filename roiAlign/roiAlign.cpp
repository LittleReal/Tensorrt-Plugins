#include "roiAlign.h"
#include "NvInfer.h"
#include "roiAlignKernel.h"
#include "common.h"

#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace std;

// ROIAlign plugin specific constants
namespace {
    static const char* ROI_ALIGN_PLUGIN_VERSION{"1"};
    static const char* ROI_ALIGN_NAME{"ROIAlignPlugin"};
}

// Static class fields initialization
PluginFieldCollection ROIAlignPluginCreator::mFC{};
std::vector<PluginField> ROIAlignPluginCreator::mPluginAttributes;

// ROIAlignPlugin ...
ROIAlignPlugin::ROIAlignPlugin(const std::string name, ROIAlignParameters param)
    : mLayerName(name)
    , mParam(param)
{
}

ROIAlignPlugin::ROIAlignPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = static_cast<const char*>(data);
    mParam = readFromBuffer<ROIAlignParameters>(d);
}

int ROIAlignPlugin::getNbOutputs() const
{
    return 1;
}

Dims ROIAlignPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 2);
    assert(index == 0);
    
    int nb_dims0 = inputs[0].nbDims;
    int in_depth = inputs[0].d[nb_dims0 - 3];

    int n_rois=inputs[1].d[0];

    Dims output;
    output.nbDims = 4;
    output.d[0]=n_rois;
    output.d[1]=in_depth;
    output.d[2]=mParam.pooled_height;
    output.d[3]=mParam.pooled_width;
    return output;
}

int ROIAlignPlugin::initialize()
{
    return 0;
}

void ROIAlignPlugin::terminate() {}

int ROIAlignPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Launch CUDA kernel wrapper and save its return value
    status = roiAlignInference(
        stream, 
        outputs,
        inputs, 
        in_width_, 
        in_height_, 
        in_depth_, 
        mParam.spatial_scale, 
        mParam.sample_ratio,
        mParam.pooled_height, 
        mParam.pooled_width, 
        mParam.position_sensitive, 
        n_rois_);
    return status;
}

size_t ROIAlignPlugin::getSerializationSize() const
{
    return sizeof(ROIAlignParameters);
}

void ROIAlignPlugin::serialize(void* buffer) const 
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;

    writeToBuffer(d, mParam);
    assert(d == a + getSerializationSize());
}

void ROIAlignPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
    // Fetch volume for future enqueue() operations
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;

    int nb_dims0 = inputs[0].nbDims;
    in_depth_ = inputs[0].d[nb_dims0 - 3], in_height_ = inputs[0].d[nb_dims0 - 2], in_width_ = inputs[0].d[nb_dims0 - 1];

    int nb_dims1=inputs[1].nbDims;
    n_rois_=inputs[1].d[0];
}

bool ROIAlignPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // This plugin only supports ordinary floats, and NCHW input format
    if (type == DataType::kFLOAT)
        return true;
    else
        return false;
}

const char* ROIAlignPlugin::getPluginType() const
{
    return ROI_ALIGN_NAME;
}

const char* ROIAlignPlugin::getPluginVersion() const
{
    return ROI_ALIGN_PLUGIN_VERSION;
}

void ROIAlignPlugin::destroy() {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2* ROIAlignPlugin::clone() const
{
    return new ROIAlignPlugin(mLayerName, mParam);
}

void ROIAlignPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}

const char* ROIAlignPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

// ROIAlignPluginCreator ...
ROIAlignPluginCreator::ROIAlignPluginCreator()
{
    // Describe ROIAlignPlugin's required PluginField arguments
    mPluginAttributes.emplace_back(PluginField("output_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatial_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("sample_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("position_sensitive", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ROIAlignPluginCreator::getPluginName() const
{
    return ROI_ALIGN_NAME;
}

const char* ROIAlignPluginCreator::getPluginVersion() const
{
    return ROI_ALIGN_PLUGIN_VERSION;
}

const PluginFieldCollection* ROIAlignPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* ROIAlignPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    ROIAlignParameters params;

    // set default params values
    params.position_sensitive=false;
    params.sample_ratio=0;

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++){
        if (strcmp(fields[i].name, "output_height") == 0) {
            params.pooled_height = *(static_cast<const int*>(fields[i].data));
        } else if (strcmp(fields[i].name, "output_width") == 0) {
            params.pooled_width = *(static_cast<const int*>(fields[i].data));
        } else if (strcmp(fields[i].name, "spatial_scale") == 0) {
            params.spatial_scale = *(static_cast<const float*>(fields[i].data));
        } else if (strcmp(fields[i].name, "sample_ratio") == 0) {
            params.sample_ratio=*(static_cast<const int*>(fields[i].data));
        } else if (strcmp(fields[i].name, "position_sensitive") == 0) {
            params.position_sensitive=*(static_cast<const bool*>(fields[i].data));
        }else if(strcmp(fields[i].name, "mode")==0){
            params.mode = *(static_cast<const int*>(fields[i].data));
            if (params.mode!=0){
                throw "The params.mode only support 0 (avg)";
            }
        }
    }
    return new ROIAlignPlugin(name, params);
}

IPluginV2* ROIAlignPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ROIAlignPlugin::destroy()
    return new ROIAlignPlugin(name, serialData, serialLength);
}

void ROIAlignPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* ROIAlignPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(ROIAlignPluginCreator);