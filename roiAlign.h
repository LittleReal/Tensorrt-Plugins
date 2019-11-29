#ifndef ROI_ALIGN_H
#define ROI_ALIGN_H

#include "NvInferPlugin.h"
#include "NvInfer.h"
#include <string>
#include <vector>

using namespace nvinfer1;

struct ROIAlignParameters
{
    int pooled_width, pooled_height;
    float spatial_scale;

    // default=-1;
    int sample_ratio;

    // default=false
    bool position_sensitive;

    // mode=0 for "avg", mode=1 for "max". Now, the roiAlign only support avg.
    // default=0;
    int mode;
};


class ROIAlignPlugin : public IPluginV2
{
public:
    ROIAlignPlugin(const std::string name, ROIAlignParameters param);

    ROIAlignPlugin(const std::string name, const void* data, size_t length);

    ROIAlignPlugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int batchSize) const override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    const std::string mLayerName;
    ROIAlignParameters mParam;
    int in_depth_, in_height_, in_width_;
    int n_rois_;
    size_t mInputVolume;
    std::string mNamespace;
};

class ROIAlignPluginCreator : public IPluginCreator
{
public:
    ROIAlignPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

private:
    template <typename T>
    T* allocMemory(int size = 1)
    {
        mTmpAllocs.reserve(mTmpAllocs.size() + 1);
        T* tmpMem = static_cast<T*>(malloc(sizeof(T) * size));
        mTmpAllocs.push_back(tmpMem);
        return tmpMem;
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
    std::vector<void*> mTmpAllocs;
};

#endif