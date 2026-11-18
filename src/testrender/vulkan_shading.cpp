// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "vulkan_shading.h"

// Built without a Vulkan SDK?  Then this is a stub that reports failure, and
// testrender simply keeps shading on the CPU.
#ifndef OSL_USE_VULKAN_HOST
#    define OSL_USE_VULKAN_HOST 0
#endif

#if !OSL_USE_VULKAN_HOST

VulkanShading::VulkanShading() {}
VulkanShading::~VulkanShading() {}
bool VulkanShading::ready() const { return false; }
const std::string& VulkanShading::device_name() const
{
    static std::string none("no Vulkan SDK");
    return none;
}
bool VulkanShading::init(std::string& err)
{
    err = "testrender was built without a Vulkan SDK";
    return false;
}
int VulkanShading::add_shader(const uint32_t*, size_t, const char*,
                              std::string& err)
{
    err = "testrender was built without a Vulkan SDK";
    return -1;
}
bool VulkanShading::shade(int, void*, size_t, std::string& err)
{
    err = "testrender was built without a Vulkan SDK";
    return false;
}

#else

#include <cstring>
#include <mutex>

#include <vulkan/vulkan.h>

namespace {

#define VK_TRY(call, err)                                                     \
    do {                                                                      \
        VkResult _vkr = (call);                                               \
        if (_vkr != VK_SUCCESS) {                                             \
            err = std::string(#call) + " returned "                           \
                  + std::to_string((int)_vkr);                                \
            return false;                                                     \
        }                                                                     \
    } while (0)

}  // namespace


// Texture region layout.  These three numbers are an ABI contract with
// osl_texture() in liboslexec/vulkan_shadeops.cpp: the kernel reaches the
// table and the texels by pointer arithmetic from the ShadeBlock pointer,
// which sits at offset 0 of this buffer.  Change one side and the other breaks.
//
//   table: kMaxTextures entries of VkTextureSlot (24 bytes each)
//   data:  RGBA32F pixels, 16-byte aligned, addressed by slot.offset
static const size_t kTextureTableOffset = 64 * 1024;
static const size_t kMaxTextures        = 16;
static const size_t kTextureDataOffset  = kTextureTableOffset
    + kMaxTextures * 40;

struct VulkanShading::Impl {
    VkInstance instance    = VK_NULL_HANDLE;
    VkPhysicalDevice phys  = VK_NULL_HANDLE;
    VkDevice device        = VK_NULL_HANDLE;
    VkQueue queue          = VK_NULL_HANDLE;
    uint32_t queue_family  = (uint32_t)-1;
    VkCommandPool cmdpool  = VK_NULL_HANDLE;
    VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
    VkFence fence          = VK_NULL_HANDLE;

    VkBuffer buffer      = VK_NULL_HANDLE;
    VkDeviceMemory mem   = VK_NULL_HANDLE;
    void* mapped         = nullptr;
    size_t buffer_bytes  = 0;
    size_t texture_capacity = 0;  // bytes available after kTextureDataOffset
    size_t texture_used     = 0;  // bytes actually occupied by the texels

    VkDescriptorSetLayout dslayout = VK_NULL_HANDLE;
    VkDescriptorPool dspool        = VK_NULL_HANDLE;
    VkDescriptorSet dset           = VK_NULL_HANDLE;

    struct Pipeline {
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipe         = VK_NULL_HANDLE;
    };
    std::vector<Pipeline> pipelines;

    std::mutex submit_mutex;
    std::string device_name;
    bool inited = false;

    ~Impl() { destroy(); }

    void destroy()
    {
        if (device == VK_NULL_HANDLE)
            return;
        vkDeviceWaitIdle(device);
        for (auto& p : pipelines) {
            if (p.pipe != VK_NULL_HANDLE)
                vkDestroyPipeline(device, p.pipe, nullptr);
            if (p.layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device, p.layout, nullptr);
            if (p.module != VK_NULL_HANDLE)
                vkDestroyShaderModule(device, p.module, nullptr);
        }
        pipelines.clear();
        if (dspool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, dspool, nullptr);
        if (dslayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device, dslayout, nullptr);
        if (buffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, buffer, nullptr);
        if (mem != VK_NULL_HANDLE)
            vkFreeMemory(device, mem, nullptr);
        if (fence != VK_NULL_HANDLE)
            vkDestroyFence(device, fence, nullptr);
        if (cmdpool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device, cmdpool, nullptr);
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
};


VulkanShading::VulkanShading() : m_impl(new Impl) {}


VulkanShading::~VulkanShading() {}


bool VulkanShading::ready() const { return m_impl && m_impl->inited; }


const std::string& VulkanShading::device_name() const
{
    return m_impl->device_name;
}


bool VulkanShading::init(std::string& err)
{
    Impl& I = *m_impl;

    VkApplicationInfo app {};
    app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "testrender";
    app.apiVersion       = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ici {};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_TRY(vkCreateInstance(&ici, nullptr, &I.instance), err);

    uint32_t ndev = 0;
    VK_TRY(vkEnumeratePhysicalDevices(I.instance, &ndev, nullptr), err);
    if (ndev == 0) {
        err = "no Vulkan physical device";
        return false;
    }
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_TRY(vkEnumeratePhysicalDevices(I.instance, &ndev, devs.data()), err);

    // A compute queue is the only requirement: this backend never touches a
    // swapchain or a graphics stage.
    for (auto d : devs) {
        uint32_t nfam = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nfam, nullptr);
        std::vector<VkQueueFamilyProperties> fams(nfam);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nfam, fams.data());
        for (uint32_t f = 0; f < nfam; ++f) {
            if (fams[f].queueCount > 0
                && (fams[f].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                I.phys         = d;
                I.queue_family = f;
                break;
            }
        }
        if (I.phys != VK_NULL_HANDLE)
            break;
    }
    if (I.phys == VK_NULL_HANDLE) {
        err = "no Vulkan device with a compute queue";
        return false;
    }

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(I.phys, &props);
    I.device_name = props.deviceName;

    if (getenv("OSL_VULKAN_SHADE_TRACE"))
        fprintf(stderr, "[vulkan] api=%u.%u\n",
                props.apiVersion >> 22,
                (props.apiVersion >> 12) & 0x3ffu);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci {};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = I.queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    // The generated modules declare Int8/Int16/Int64 capabilities: OSL reads
    // and writes bytes and halves through the ShadeBlock (a `char` parameter
    // access is an i8 load/store in the IR, and clspv preserves that).  A
    // Vulkan device only accepts those if the matching features and storage
    // extensions are enabled at device creation -- without them the driver
    // rejects the module at pipeline bind time, which shows up as an access
    // violation a few dispatches in.  Enable whatever this device advertises.
    VkPhysicalDeviceFeatures avail {};
    vkGetPhysicalDeviceFeatures(I.phys, &avail);
    VkPhysicalDeviceFeatures want {};
    want.shaderInt64 = avail.shaderInt64;
    want.shaderInt16 = avail.shaderInt16;

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(I.phys, nullptr, &next, nullptr);
    std::vector<VkExtensionProperties> exts(next);
    vkEnumerateDeviceExtensionProperties(I.phys, nullptr, &next, exts.data());
    bool has_8bit  = false;
    bool has_16bit = false;
    bool has_varptr = false;
    for (auto& e : exts) {
        if (!strcmp(e.extensionName, VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
            has_8bit = true;
        if (!strcmp(e.extensionName, VK_KHR_16BIT_STORAGE_EXTENSION_NAME))
            has_16bit = true;
        if (!strcmp(e.extensionName, VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME))
            has_varptr = true;
    }

    std::vector<const char*> enabled_exts;
    if (has_8bit)
        enabled_exts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    if (has_16bit)
        enabled_exts.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    if (has_varptr)
        enabled_exts.push_back(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME);

    VkPhysicalDevice8BitStorageFeatures s8 {};
    s8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDevice16BitStorageFeatures s16 {};
    s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceVariablePointersFeatures varptr {};
    varptr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
    void* dev_pnext = nullptr;
    if (has_varptr) {
        // The LLVM SPIR-V backend's kernel takes its buffer through
        // address_space(11) pointers (clspv instead turns the kernel argument
        // into a descriptor), so the module declares
        // VariablePointersStorageBuffer.  That capability is legal once this
        // feature is enabled; without it the driver rejects the pipeline.
        varptr.variablePointersStorageBuffer = VK_TRUE;
        varptr.variablePointers              = VK_FALSE;
        varptr.pNext                         = dev_pnext;
        dev_pnext                            = &varptr;
    }
    if (has_16bit) {
        // Declaring VK_KHR_16bit_storage is not enough: its storage features
        // default to false, and the generated modules do use 16-bit storage
        // (the Int16 capability comes straight from OSL's half-sized fields).
        s16.storageBuffer16BitAccess           = VK_TRUE;
        s16.uniformAndStorageBuffer16BitAccess = VK_TRUE;
        s16.storagePushConstant16              = VK_TRUE;
        s16.storageInputOutput16               = VK_FALSE;
        s16.pNext                              = dev_pnext;
        dev_pnext                              = &s16;
    }
    if (has_8bit) {
        s8.storageBuffer8BitAccess           = VK_TRUE;
        s8.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        s8.pNext                             = dev_pnext;
        dev_pnext                            = &s8;
    }

    VkDeviceCreateInfo dci {};
    dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    dci.pNext                = dev_pnext;
    dci.pEnabledFeatures     = &want;
    dci.enabledExtensionCount   = (uint32_t)enabled_exts.size();
    dci.ppEnabledExtensionNames = enabled_exts.empty() ? nullptr
                                                       : enabled_exts.data();
    VK_TRY(vkCreateDevice(I.phys, &dci, nullptr, &I.device), err);
    vkGetDeviceQueue(I.device, I.queue_family, 0, &I.queue);

    if (getenv("OSL_VULKAN_SHADE_TRACE"))
        fprintf(stderr,
                "[vulkan] 8bit_storage=%d 16bit_storage=%d shaderInt64=%d "
                "shaderInt16=%d\n",
                (int)has_8bit, (int)has_16bit, (int)want.shaderInt64,
                (int)want.shaderInt16);

    // Descriptor: binding 0 is the one storage buffer holding the ShadeBlock.
    VkDescriptorSetLayoutBinding b {};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci {};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 1;
    dlci.pBindings    = &b;
    VK_TRY(vkCreateDescriptorSetLayout(I.device, &dlci, nullptr, &I.dslayout),
           err);

    VkDescriptorPoolSize psz {};
    psz.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci {};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psz;
    VK_TRY(vkCreateDescriptorPool(I.device, &dpci, nullptr, &I.dspool), err);

    VkDescriptorSetAllocateInfo dsai {};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = I.dspool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &I.dslayout;
    VK_TRY(vkAllocateDescriptorSets(I.device, &dsai, &I.dset), err);

    // One storage buffer, host visible + coherent: the ShadeBlock is written
    // and read back by the CPU around every dispatch, so a staging copy would
    // only add work.  Layout, from the start of the buffer:
    //
    //   [0, kBlockCapacity)                ShadeBlock (one per dispatch)
    //   [kTextureTableOffset, +16*32)      texture table, see VkTextureSlot
    //   [kTextureDataOffset, ...)          decoded RGBA32F pixels
    //
    // Keeping the texture in the same buffer means no second kernel argument
    // and no second descriptor binding: the kernel already gets a pointer to
    // the buffer, so texels are reachable as sg + kTextureDataOffset.
    // The whole-frame kernel keeps one 576-byte ShadeBlock per pixel sample,
    // so 320x240 at aa16 needs ~44 MB for the arena alone, on top of textures
    // and geometry: 64 MiB genuinely does not fit (measured 70.8 MB).  Raise
    // this only as far as the device's host-visible heap allows -- 72 MiB
    // already fails on Intel UHD ("no host-visible coherent memory type"), so
    // the real fix for full resolution is staging into DEVICE_LOCAL memory.
    I.buffer_bytes = 64 * 1024 * 1024;
    if (const char* mb = getenv("OSL_VULKAN_BUFFER_MB"))
        I.buffer_bytes = (size_t)atoi(mb) * 1024 * 1024;
    I.texture_capacity = I.buffer_bytes - kTextureDataOffset;
    VkBufferCreateInfo bci {};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = I.buffer_bytes;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_TRY(vkCreateBuffer(I.device, &bci, nullptr, &I.buffer), err);

    VkMemoryRequirements req {};
    vkGetBufferMemoryRequirements(I.device, I.buffer, &req);
    VkPhysicalDeviceMemoryProperties mp {};
    vkGetPhysicalDeviceMemoryProperties(I.phys, &mp);
    uint32_t mt = (uint32_t)-1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const auto flags = mp.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u << i))
            && (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                         | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                   == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                       | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            mt = i;
            break;
        }
    }
    if (mt == (uint32_t)-1) {
        // Say which heaps and types exist, so "does not fit" is actionable
        // instead of a bare message: the answer is usually that the only
        // mappable heap is smaller than the requested allocation.
        fprintf(stderr, "[vulkan] memory heaps (%u) / types (%u), request %llu bytes, typeBits 0x%x\n",
                mp.memoryHeapCount, mp.memoryTypeCount,
                (unsigned long long)req.size, (unsigned)req.memoryTypeBits);
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
            fprintf(stderr, "[vulkan]   heap %u: %llu MiB, flags 0x%x\n", h,
                    (unsigned long long)(mp.memoryHeaps[h].size >> 20),
                    (unsigned)mp.memoryHeaps[h].flags);
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t)
            fprintf(stderr, "[vulkan]   type %u: heap %u, flags 0x%x%s\n", t,
                    (unsigned)mp.memoryTypes[t].heapIndex,
                    (unsigned)mp.memoryTypes[t].propertyFlags,
                    (1u << t) & req.memoryTypeBits ? "" : "  (not for this buffer)");
        err = "no host-visible coherent memory type for the storage buffer";
        return false;
    }
    {
        const auto f = mp.memoryTypes[mt].propertyFlags;
        fprintf(stderr,
                "[vulkan] storage buffer: %llu MiB in memory type %u (heap %u, "
                "flags%s%s%s), heap size %llu MiB\n",
                (unsigned long long)(req.size >> 20), mt,
                (unsigned)mp.memoryTypes[mt].heapIndex,
                (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
                (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[mt].heapIndex].size >> 20));
    }
    VkMemoryAllocateInfo mai {};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = mt;
    VK_TRY(vkAllocateMemory(I.device, &mai, nullptr, &I.mem), err);
    VK_TRY(vkBindBufferMemory(I.device, I.buffer, I.mem, 0), err);
    VK_TRY(vkMapMemory(I.device, I.mem, 0, VK_WHOLE_SIZE, 0, &I.mapped), err);

    VkDescriptorBufferInfo dbi {};
    dbi.buffer = I.buffer;
    dbi.offset = 0;
    dbi.range  = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds {};
    wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet          = I.dset;
    wds.dstBinding      = 0;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo     = &dbi;
    vkUpdateDescriptorSets(I.device, 1, &wds, 0, nullptr);

    VkCommandPoolCreateInfo cpi {};
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = I.queue_family;
    VK_TRY(vkCreateCommandPool(I.device, &cpi, nullptr, &I.cmdpool), err);

    VkCommandBufferAllocateInfo cbai {};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = I.cmdpool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_TRY(vkAllocateCommandBuffers(I.device, &cbai, &I.cmdbuf), err);

    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_TRY(vkCreateFence(I.device, &fci, nullptr, &I.fence), err);

    I.inited = true;
    return true;
}


int VulkanShading::add_shader(const uint32_t* spirv, size_t word_count,
                              const char* entry, std::string& err)
{
    Impl& I = *m_impl;
    if (!I.inited) {
        err = "VulkanShading::add_shader before init";
        return -1;
    }
    if (word_count == 0) {
        err = "empty SPIR-V module";
        return -1;
    }

    Impl::Pipeline p;

    VkShaderModuleCreateInfo smci {};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = word_count * sizeof(uint32_t);
    smci.pCode    = spirv;
    if (vkCreateShaderModule(I.device, &smci, nullptr, &p.module)
        != VK_SUCCESS) {
        err = "vkCreateShaderModule failed";
        return -1;
    }

    VkPipelineLayoutCreateInfo plci {};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // The module declares a storage-buffer variable, so the layout must carry
    // the matching set layout.  Leaving it empty is a validation error
    // (VUID-VkComputePipelineCreateInfo-layout-07988) and, on the Intel
    // driver, a process-level access violation at the first dispatch.
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &I.dslayout;
    if (vkCreatePipelineLayout(I.device, &plci, nullptr, &p.layout)
        != VK_SUCCESS) {
        err = "vkCreatePipelineLayout failed";
        return -1;
    }

    VkPipelineShaderStageCreateInfo stage {};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.module;
    stage.pName  = (entry && *entry) ? entry : "osl_kernel";

    VkComputePipelineCreateInfo cpci {};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = p.layout;

    VkResult r = vkCreateComputePipelines(I.device, VK_NULL_HANDLE, 1, &cpci,
                                          nullptr, &p.pipe);
    if (r != VK_SUCCESS) {
        err = "vkCreateComputePipelines returned " + std::to_string((int)r);
        return -1;
    }

    I.pipelines.push_back(p);
    return (int)I.pipelines.size() - 1;
}


bool VulkanShading::shade(int id, void* block, size_t block_bytes,
                          std::string& err)
{
    Impl& I = *m_impl;
    // One queue, one command buffer, one fence, shared by every rendering
    // thread: the submit/wait pair has to be serialized.
    std::lock_guard<std::mutex> lock(I.submit_mutex);
    if (!I.inited) {
        err = "VulkanShading::shade before init";
        return false;
    }
    if (id < 0 || id >= (int)I.pipelines.size()) {
        err = "VulkanShading::shade with an unknown pipeline id";
        return false;
    }
    if (block_bytes > I.buffer_bytes) {
        err = "ShadeBlock does not fit in the storage buffer";
        return false;
    }

    memcpy(I.mapped, block, block_bytes);

    VK_TRY(vkResetFences(I.device, 1, &I.fence), err);
    VK_TRY(vkResetCommandBuffer(I.cmdbuf, 0), err);

    VkCommandBufferBeginInfo cbbi {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY(vkBeginCommandBuffer(I.cmdbuf, &cbbi), err);
    vkCmdBindPipeline(I.cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      I.pipelines[id].pipe);
    vkCmdBindDescriptorSets(I.cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            I.pipelines[id].layout, 0, 1, &I.dset, 0, nullptr);
    // One workgroup of one invocation: the module's gl_WorkGroupSize is
    // (1,1,1) and the kernel takes the address of a single ShadeBlock.
    vkCmdDispatch(I.cmdbuf, 1, 1, 1);
    VK_TRY(vkEndCommandBuffer(I.cmdbuf), err);

    VkSubmitInfo si {};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &I.cmdbuf;
    VK_TRY(vkQueueSubmit(I.queue, 1, &si, I.fence), err);
    VK_TRY(vkWaitForFences(I.device, 1, &I.fence, VK_TRUE, UINT64_MAX), err);

    memcpy(block, I.mapped, block_bytes);
    return true;
}


bool VulkanShading::set_textures(const std::vector<Texture>& textures,
                                 std::string& err)
{
    Impl& I = *m_impl;
    std::lock_guard<std::mutex> lock(I.submit_mutex);
    if (!I.inited) {
        err = "VulkanShading::set_textures before init";
        return false;
    }
    if (textures.size() > kMaxTextures) {
        err = "set_textures: more textures than the table holds";
        return false;
    }

    // slot layout, matching TexSlot in vulkan_shadeops.cpp byte for byte:
    //   u64 hash | i32 w | i32 h | i64 offset | i32 channels | i32 nlevels |
    //   i32 pad0 | i32 pad1
    struct Slot {
        unsigned long long hash;
        int width, height;
        long long offset;  // bytes from kTextureDataOffset
        int channels, nlevels, pad0, pad1;
    };
    static_assert(sizeof(Slot) == 40, "texture slot must stay 40 bytes");
    auto* table = (Slot*)((char*)I.mapped + kTextureTableOffset);
    memset(table, 0, kMaxTextures * sizeof(Slot));

    size_t cursor = 0;
    for (size_t i = 0; i < textures.size(); ++i) {
        const Texture& t = textures[i];
        size_t bytes     = t.rgba.size() * sizeof(float);
        if (t.width <= 0 || t.height <= 0 || t.level_width.empty()) {
            err = "set_textures: malformed image";
            return false;
        }
        // The device walks the chain itself, assuming each level is half the
        // previous one, so refuse anything that does not have that shape.
        size_t expect = 0;
        for (size_t k = 0; k < t.level_width.size(); ++k) {
            expect += (size_t)t.level_width[k] * t.level_height[k] * 16;
        }
        if (bytes != expect) {
            err = "set_textures: mip chain does not match its level sizes";
            return false;
        }
        if (cursor + bytes > I.texture_capacity) {
            err = "set_textures: texture data does not fit in the buffer";
            return false;
        }
        table[i].hash     = t.hash;
        table[i].width    = t.width;
        table[i].height   = t.height;
        table[i].offset   = (long long)cursor;
        table[i].channels = 4;
        table[i].nlevels  = (int)t.level_width.size();
        table[i].pad0     = 0;
        table[i].pad1     = 0;
        memcpy((char*)I.mapped + kTextureDataOffset + cursor, t.rgba.data(),
               bytes);
        cursor += bytes;
    }
    I.texture_used = cursor;
    return true;
}


size_t VulkanShading::capacity() const
{
    return m_impl ? m_impl->buffer_bytes : 0;
}


size_t VulkanShading::texture_end() const
{
    return m_impl ? kTextureDataOffset + m_impl->texture_used : 0;
}


bool VulkanShading::upload(size_t offset, const void* src, size_t bytes,
                           std::string& err)
{
    Impl& I = *m_impl;
    if (!I.inited) {
        err = "VulkanShading::upload before init";
        return false;
    }
    if (offset > I.buffer_bytes || bytes > I.buffer_bytes - offset) {
        err = "upload out of range (offset " + std::to_string(offset)
              + ", bytes " + std::to_string(bytes) + ", buffer "
              + std::to_string(I.buffer_bytes) + ")";
        return false;
    }
    memcpy((char*)I.mapped + offset, src, bytes);
    return true;
}


bool VulkanShading::download(size_t offset, void* dst, size_t bytes,
                             std::string& err)
{
    Impl& I = *m_impl;
    if (!I.inited) {
        err = "VulkanShading::download before init";
        return false;
    }
    if (offset > I.buffer_bytes || bytes > I.buffer_bytes - offset) {
        err = "download out of range";
        return false;
    }
    memcpy(dst, (char*)I.mapped + offset, bytes);
    return true;
}


bool VulkanShading::dispatch(int id, unsigned groups, std::string& err)
{
    Impl& I = *m_impl;
    std::lock_guard<std::mutex> lock(I.submit_mutex);
    if (!I.inited) {
        err = "VulkanShading::dispatch before init";
        return false;
    }
    if (id < 0 || id >= (int)I.pipelines.size()) {
        err = "VulkanShading::dispatch with an unknown pipeline id";
        return false;
    }

    VK_TRY(vkResetFences(I.device, 1, &I.fence), err);
    VK_TRY(vkResetCommandBuffer(I.cmdbuf, 0), err);

    VkCommandBufferBeginInfo cbbi {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY(vkBeginCommandBuffer(I.cmdbuf, &cbbi), err);
    vkCmdBindPipeline(I.cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      I.pipelines[id].pipe);
    vkCmdBindDescriptorSets(I.cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            I.pipelines[id].layout, 0, 1, &I.dset, 0, nullptr);
    vkCmdDispatch(I.cmdbuf, groups ? groups : 1, 1, 1);
    VK_TRY(vkEndCommandBuffer(I.cmdbuf), err);

    VkSubmitInfo si {};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &I.cmdbuf;
    VK_TRY(vkQueueSubmit(I.queue, 1, &si, I.fence), err);
    VK_TRY(vkWaitForFences(I.device, 1, &I.fence, VK_TRUE, UINT64_MAX), err);
    return true;
}

#endif  // OSL_USE_VULKAN_HOST
