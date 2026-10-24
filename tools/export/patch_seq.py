# 多帧超分的序列导出（s193），打在"LearnDX12 + 沙盒钩子"的临时副本上（钩子：SHADOW_CAM、WIN_POS、NO_UI、TAA_DT60 等）
#   python patch_seq.py <副本目录>
#
# 环境变量：
#   SEQ_CAMS=机位表（tools/gen_seq.py 生成）：每行 seq,frame,export,balls,ex,ey,ez,tx,ty,tz
#            frame < 0 是预滚帧（只为让第 0 帧的"上一帧"有意义，不导出）；balls = 1 时测试球左右动
#   SEQ_DIR=输出目录   SEQ_MODE=lr | gt   SEQ_SUB=GT 每帧抖几次（默认 64）   SEQ_WARMUP=预热帧数（默认 30）
#   RENDER_SIZE=W,H：渲染分辨率（lr 用 1280,720，gt 用 2560,1440）
#
# 一行 = 一个"序列帧"。时间（波浪、测试球）只在每个序列帧的第一次绘制时推进一步（1/60 秒），
# 所以 lr 和 gt 两次运行在同一个序列帧上场景完全一样，和各自画得快慢无关。
#   lr：每行画 1 次。抖动 = 32 相位（mTaaUpscale = 2）里的第 (frame mod 32) + 1 个；
#       读回 ③ 透明之后、TAA / Bloom 之前的 HDR，光照 Pass 刚结束时的深度，帧末的速度缓冲
#   gt：导出的行画 1 + SEQ_SUB 次：第 0 次不抖（读回 G-Buffer、深度；测试集再读一份不抗锯齿的 HDR），
#       第 1..SEQ_SUB 次用 64 相位 Halton 抖动，HDR 用加法混合累加进一张 R32G32B32A32_FLOAT，最后除以次数
# 文件（{seq:02d}_{frame:02d}_*.npy）：
#   lr：color (H,W,3) f16 | velocity (H,W,2) f16（uv 空间，这一帧 − 上一帧）| depthraw (H,W) u4 | 另有 jitter_lr.csv
#   gt：gt (H,W,3) f16 | gb0 (H,W,4) u1 | gb1 (H,W,4) f2 | depthraw (H,W) u4 | noaa (H,W,3) f16（只有 split = test）
import os
import sys

D = sys.argv[1]
s = ""
crlf = False


def load(p):
    global s, crlf
    raw = open(os.path.join(D, p), "rb").read().decode("utf-8-sig")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")


def save(p):
    open(os.path.join(D, p), "w", encoding="utf-8", newline="\r\n" if crlf else "\n").write(s)


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


ACCUM_HLSL = '''// 导出用（s193）：把这一帧的 HDR 原样输出，PSO 开加法混合 → 累加进 R32G32B32A32_FLOAT
Texture2D<float4> gHdr : register(t0);

struct VertexOut
{
    float4 PosH : SV_Position;
};

VertexOut VS(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    VertexOut vout;
    vout.PosH = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    return float4(gHdr.Load(int3(pin.PosH.xy, 0)).rgb, 1.0f);
}
'''
open(os.path.join(D, "HLSL", "SeqAccum.hlsl"), "w", encoding="utf-8", newline="\n").write(ACCUM_HLSL)

load("main.cpp")
rep('''#include <random>
''', '''#include <random>
#include <array>
#include <fstream>
#include <string>
#include <DirectXPackedVector.h>
''')
rep('''static std::wstring gSnapPath, gDumpPath;
''', '''static std::wstring gSnapPath, gDumpPath;
// ---- 序列导出（s193）----
struct SeqRow { int seq, frame, exportIt, balls; float cam[6]; };
static std::vector<SeqRow> gSeq;
static std::string gSeqDir, gSeqSplit;
static int gSeqMode = 0; // 0 = 不导出，1 = lr，2 = gt
static int gSeqSub = 64, gSeqWarmup = 30;
static int gSeqWarm = 0, gSeqRow = 0, gSeqSubIdx = -1; // 调度：预热了几帧、画到第几行、这一行第几次
static bool gSeqAdvance = true; // 这一次绘制要不要推进时间
static bool gSeqDone = false;
static int SeqRowsDone() { return gSeqRow; }

// 读回缓冲 → .npy。shape (h, w, c)（c = 1 写 (h, w)）；每个纹素只拷前 copyBytes 个字节
static void WriteNpy(const std::string& path, ID3D12Resource* rb, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,
                     const char* descr, UINT channels, UINT bytesPerTexel, UINT copyBytes)
{
    const UINT w = fp.Footprint.Width, h = fp.Footprint.Height, pitch = fp.Footprint.RowPitch;
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" +
                         std::to_string(h) + ", " + std::to_string(w) + (channels > 1 ? ", " + std::to_string(channels) : "") + "), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\\x93NUMPY\\x01\\x00", 8);
    const uint16_t hl = (uint16_t)header.size();
    f.write((const char*)&hl, 2);
    f.write(header.data(), header.size());
    BYTE* data = nullptr;
    ThrowIfFailed(rb->Map(0, nullptr, (void**)&data));
    std::vector<BYTE> row((size_t)w * copyBytes);
    for (UINT y = 0; y < h; ++y)
    {
        const BYTE* src = data + (size_t)y * pitch;
        if (copyBytes == bytesPerTexel) memcpy(row.data(), src, row.size());
        else for (UINT x = 0; x < w; ++x) memcpy(&row[(size_t)x * copyBytes], src + (size_t)x * bytesPerTexel, copyBytes);
        f.write((const char*)row.data(), (std::streamsize)row.size());
    }
    D3D12_RANGE noWrite = {0, 0};
    rb->Unmap(0, &noWrite);
}

// R32G32B32A32_FLOAT 的累加结果 → 除以 n → float16 RGB .npy
static void WriteNpyAccum(const std::string& path, ID3D12Resource* rb, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, int n)
{
    const UINT w = fp.Footprint.Width, h = fp.Footprint.Height, pitch = fp.Footprint.RowPitch;
    std::string header = "{'descr': '<f2', 'fortran_order': False, 'shape': (" + std::to_string(h) + ", " + std::to_string(w) + ", 3), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\\x93NUMPY\\x01\\x00", 8);
    const uint16_t hl = (uint16_t)header.size();
    f.write((const char*)&hl, 2);
    f.write(header.data(), header.size());
    BYTE* data = nullptr;
    ThrowIfFailed(rb->Map(0, nullptr, (void**)&data));
    std::vector<uint16_t> row((size_t)w * 3);
    const float inv = 1.0f / n;
    for (UINT y = 0; y < h; ++y)
    {
        const float* src = (const float*)(data + (size_t)y * pitch);
        for (UINT x = 0; x < w; ++x)
            for (UINT c = 0; c < 3; ++c) row[(size_t)x * 3 + c] = DirectX::PackedVector::XMConvertFloatToHalf(src[x * 4 + c] * inv);
        f.write((const char*)row.data(), (std::streamsize)row.size() * 2);
    }
    D3D12_RANGE noWrite = {0, 0};
    rb->Unmap(0, &noWrite);
}
static ComPtr<ID3D12Resource> gSeqRb[5]; // 0 HDR / GT 累加 1 深度 2 速度 / G-Buffer0 3 G-Buffer1 4 不抗锯齿的 HDR
static D3D12_PLACED_SUBRESOURCE_FOOTPRINT gSeqFp[5];
static bool gSeqUsed[5];
''')

# 读机位表
rep('''    if (GetEnvironmentVariableA("SHADOW_CAM", env, sizeof(env)) > 0)
''', '''    if (GetEnvironmentVariableA("SEQ_CAMS", env, sizeof(env)) > 0)
    {
        std::ifstream cams(env);
        std::string line;
        while (std::getline(cams, line))
        {
            SeqRow r{};
            if (sscanf_s(line.c_str(), "%d,%d,%d,%d,%f,%f,%f,%f,%f,%f", &r.seq, &r.frame, &r.exportIt, &r.balls,
                         &r.cam[0], &r.cam[1], &r.cam[2], &r.cam[3], &r.cam[4], &r.cam[5]) == 10) gSeq.push_back(r);
        }
        char buf[MAX_PATH] = {};
        if (GetEnvironmentVariableA("SEQ_DIR", buf, sizeof(buf)) > 0) gSeqDir = buf;
        if (GetEnvironmentVariableA("SEQ_SPLIT", buf, sizeof(buf)) > 0) gSeqSplit = buf;
        if (GetEnvironmentVariableA("SEQ_MODE", buf, sizeof(buf)) > 0) gSeqMode = std::string(buf) == "gt" ? 2 : 1;
        gSeqSub = EnvInt("SEQ_SUB", 64);
        gSeqWarmup = EnvInt("SEQ_WARMUP", 30);
        char msg[160];
        sprintf_s(msg, "[SEQ] %zu rows, mode %d, sub %d, warmup %d\\n", gSeq.size(), gSeqMode, gSeqSub, gSeqWarmup);
        OutputDebugStringA(msg);
    }
    if (GetEnvironmentVariableA("SHADOW_CAM", env, sizeof(env)) > 0)
''')

# 每帧开头：排这一次画哪一行、第几次；换机位、抖动、测试球
rep('''    void Update(const GameTimer& gt) override
    {
''', '''    void Update(const GameTimer& gt) override
    {
        if (gSeqMode != 0) SeqSchedule();
''')
rep('''    static void WriteBmp(''', '''    // 序列导出：这一次绘制画哪一行、第几次（预热时 gSeqSubIdx = -1，用第 0 行的机位）
    void SeqSchedule()
    {
        if (gSeq.empty() || gSeqDone) return;
        int rowIdx = 0;
        if (gSeqWarm < gSeqWarmup)
        {
            ++gSeqWarm;
            gSeqSubIdx = -1;
            gSeqAdvance = true;
        }
        else
        {
            const SeqRow& cur = gSeq[gSeqRow];
            const int last = (gSeqMode == 2 && cur.exportIt) ? gSeqSub : 0; // 这一行最后一次的编号
            if (gSeqSubIdx < 0) gSeqSubIdx = 0; // 预热刚结束：第 0 行第 0 次
            else if (gSeqSubIdx < last) ++gSeqSubIdx;
            else { ++gSeqRow; gSeqSubIdx = 0; }
            if (gSeqRow >= (int)gSeq.size()) { gSeqDone = true; PostQuitMessage(0); return; }
            gSeqAdvance = gSeqSubIdx == 0;
            rowIdx = gSeqRow;
        }
        const SeqRow& r = gSeq[rowIdx];
        mCamera.LookAt(XMFLOAT3(r.cam[0], r.cam[1], r.cam[2]), XMFLOAT3(r.cam[3], r.cam[4], r.cam[5]), XMFLOAT3(0.0f, 1.0f, 0.0f));
        mTaaMoveSpheres = r.balls != 0;
        if (gSeqMode == 1)
        {
            mTaa = true;
            mTaaUpscale = 2.0f; // 32 相位
            mTaaFrame = (UINT)((r.frame % 32 + 32) % 32);
        }
        else
        {
            mTaa = gSeqSubIdx > 0; // 第 0 次不抖
            mTaaUpscale = sqrtf(8.0f); // TaaPhaseCount = 8 × 8 = 64 相位
            mTaaFrame = gSeqSubIdx > 0 ? (UINT)(gSeqSubIdx - 1) : 0;
        }
    }
    // 这一次绘制要读回哪些
    bool SeqExportNow() const { return gSeqMode != 0 && !gSeqDone && gSeqSubIdx >= 0 && gSeq[gSeqRow].exportIt; }

    // src（当前在 state）→ 读回缓冲 gSeqRb[i]，录完转回 state
    void SeqCopy(int i, ID3D12Resource* src, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC desc = src->GetDesc();
        UINT64 total = 0;
        mDevice->GetCopyableFootprints(&desc, 0, 1, 0, &gSeqFp[i], nullptr, nullptr, &total);
        if (!gSeqRb[i])
        {
            CD3DX12_HEAP_PROPERTIES rbHeap(D3D12_HEAP_TYPE_READBACK);
            CD3DX12_RESOURCE_DESC rbDesc = CD3DX12_RESOURCE_DESC::Buffer(total);
            ThrowIfFailed(mDevice->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gSeqRb[i])));
        }
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(src, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mCommandList->ResourceBarrier(1, &toCopy);
        CD3DX12_TEXTURE_COPY_LOCATION dst(gSeqRb[i].Get(), gSeqFp[i]), from(src, 0);
        mCommandList->CopyTextureRegion(&dst, 0, 0, 0, &from, nullptr);
        auto back = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        mCommandList->ResourceBarrier(1, &back);
        gSeqUsed[i] = true;
    }

    // GT 累加：HDR（PIXEL_SHADER_RESOURCE）加进 mSeqAccum（一直停在 RENDER_TARGET）
    ComPtr<ID3D12Resource> mSeqAccum;
    ComPtr<ID3D12DescriptorHeap> mSeqAccumRtvHeap;
    ComPtr<ID3D12PipelineState> mSeqAccumPSO;
    void SeqAccumulate(bool clearFirst)
    {
        if (!mSeqAccum)
        {
            D3D12_RESOURCE_DESC desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT, mClientWidth, mClientHeight, 1, 1);
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            D3D12_CLEAR_VALUE clear = {.Format = DXGI_FORMAT_R32G32B32A32_FLOAT, .Color = {0, 0, 0, 0}};
            CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
            ThrowIfFailed(mDevice->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                           &clear, IID_PPV_ARGS(&mSeqAccum)));
            D3D12_DESCRIPTOR_HEAP_DESC hd = {.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, .NumDescriptors = 1};
            ThrowIfFailed(mDevice->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&mSeqAccumRtvHeap)));
            mDevice->CreateRenderTargetView(mSeqAccum.Get(), nullptr, mSeqAccumRtvHeap->GetCPUDescriptorHandleForHeapStart());
            ComPtr<ID3DBlob> vs = CompileShader(L"HLSL\\\\SeqAccum.hlsl", nullptr, "VS", "vs_6_6");
            ComPtr<ID3DBlob> ps = CompileShader(L"HLSL\\\\SeqAccum.hlsl", nullptr, "PS", "ps_6_6");
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = mTonemapRootSignature.Get(); // 根参数 0 = t0 表
            pd.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
            pd.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
            pd.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pd.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            auto& b = pd.BlendState.RenderTarget[0];
            b.BlendEnable = TRUE;
            b.SrcBlend = b.DestBlend = D3D12_BLEND_ONE;
            b.BlendOp = D3D12_BLEND_OP_ADD;
            b.SrcBlendAlpha = b.DestBlendAlpha = D3D12_BLEND_ONE;
            b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            pd.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
            pd.DepthStencilState.DepthEnable = false;
            pd.SampleMask = UINT_MAX;
            pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pd.NumRenderTargets = 1;
            pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
            pd.SampleDesc.Count = 1;
            ThrowIfFailed(mDevice->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&mSeqAccumPSO)));
        }
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = mSeqAccumRtvHeap->GetCPUDescriptorHandleForHeapStart();
        if (clearFirst)
        {
            const float zero[4] = {0, 0, 0, 0};
            mCommandList->ClearRenderTargetView(rtv, zero, 0, nullptr);
        }
        mCommandList->OMSetRenderTargets(1, &rtv, true, nullptr);
        mCommandList->SetGraphicsRootSignature(mTonemapRootSignature.Get());
        mCommandList->SetGraphicsRootDescriptorTable(0, HeapGpuHandle(HeapSlot::Hdr));
        mCommandList->SetPipelineState(mSeqAccumPSO.Get());
        mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        mCommandList->DrawInstanced(3, 1, 0, 0);
    }

    // 帧末（已 Flush）：写这一次读回的文件
    void SeqWrite()
    {
        const SeqRow& r = gSeq[gSeqRow];
        char base[MAX_PATH];
        sprintf_s(base, "%s\\\\%02d_%02d_", gSeqDir.c_str(), r.seq, r.frame);
        const std::string b = base;
        if (gSeqMode == 1)
        {
            WriteNpy(b + "color.npy", gSeqRb[0].Get(), gSeqFp[0], "<f2", 3, 8, 6);
            WriteNpy(b + "depthraw.npy", gSeqRb[1].Get(), gSeqFp[1], "<u4", 1, 4, 4);
            WriteNpy(b + "velocity.npy", gSeqRb[2].Get(), gSeqFp[2], "<f2", 2, 4, 4);
            std::ofstream j(gSeqDir + "\\\\jitter_lr.csv", std::ios::app);
            j << r.seq << "," << r.frame << "," << mTaaJitter.x << "," << mTaaJitter.y << "\\n";
        }
        else if (gSeqSubIdx == 0)
        {
            WriteNpy(b + "gb0.npy", gSeqRb[2].Get(), gSeqFp[2], "|u1", 4, 4, 4);
            WriteNpy(b + "gb1.npy", gSeqRb[3].Get(), gSeqFp[3], "<f2", 4, 8, 8);
            WriteNpy(b + "depthraw.npy", gSeqRb[1].Get(), gSeqFp[1], "<u4", 1, 4, 4);
            if (gSeqUsed[4]) WriteNpy(b + "noaa.npy", gSeqRb[4].Get(), gSeqFp[4], "<f2", 3, 8, 6);
        }
        else if (gSeqSubIdx == gSeqSub)
        {
            WriteNpyAccum(b + "gt.npy", gSeqRb[0].Get(), gSeqFp[0], gSeqSub);
        }
        for (bool& u : gSeqUsed) u = false;
        if (r.frame % 8 == 0 && (gSeqMode == 1 || gSeqSubIdx == gSeqSub))
        {
            char msg[160];
            sprintf_s(msg, "[SEQ] seq %d frame %d  %ux%u\\n", r.seq, r.frame, mClientWidth, mClientHeight);
            OutputDebugStringA(msg);
        }
    }

    static void WriteBmp(''')

# 深度：光照 Pass 刚结束（只有不透明物体，和 G-Buffer 对得上）
rep('''        mCommandList->ResourceBarrier(1, &depthBack);
''', '''        mCommandList->ResourceBarrier(1, &depthBack);
        if (SeqExportNow() && (gSeqMode == 1 || gSeqSubIdx == 0))
            SeqCopy(1, mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
''')
# HDR：③ 透明之后、TAA / Bloom 之前（刚切成 PIXEL_SHADER_RESOURCE）
rep('''        mCommandList->ResourceBarrier(1, &hdrToSrv);
''', '''        mCommandList->ResourceBarrier(1, &hdrToSrv);
        if (SeqExportNow())
        {
            if (gSeqMode == 1)
                SeqCopy(0, mHdrMap.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            else if (gSeqSubIdx == 0 && gSeqSplit == "test")
                SeqCopy(4, mHdrMap.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            else if (gSeqSubIdx > 0)
            {
                SeqAccumulate(gSeqSubIdx == 1);
                if (gSeqSubIdx == gSeqSub) SeqCopy(0, mSeqAccum.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET);
            }
        }
''')
# 帧末：速度缓冲（lr）、G-Buffer（gt 第 0 次）
rep('''        // 后台缓冲：RENDER_TARGET → PRESENT
''', '''        const bool seqWrite = SeqExportNow() && (gSeqMode == 1 || gSeqSubIdx == 0 || gSeqSubIdx == gSeqSub);
        if (SeqExportNow() && gSeqMode == 1)
            SeqCopy(2, mVelocity.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        if (SeqExportNow() && gSeqMode == 2 && gSeqSubIdx == 0)
        {
            SeqCopy(2, mGBuffer[0].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            SeqCopy(3, mGBuffer[1].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // 后台缓冲：RENDER_TARGET → PRESENT
''')
rep('''        if (snap)
        {
            FlushCommandQueue();
            WriteBmp(''', '''        if (seqWrite)
        {
            FlushCommandQueue();
            SeqWrite();
        }
        if (snap)
        {
            FlushCommandQueue();
            WriteBmp(''')
# 时间只在每个序列帧的第一次绘制时推进
rep('''        sWaveT += 1.0f / 60.0f;
''', '''        if (gSeqAdvance) sWaveT += 1.0f / 60.0f;
''')
rep('''        mWaves->Update(1.0f / 60.0f);
''', '''        if (gSeqAdvance) mWaves->Update(1.0f / 60.0f);
''')
rep('''        mSphereAnimTime += EnvInt("TAA_DT60", 0) ? 1.0f / 60.0f : gt.DeltaTime();''',
    '''        if (gSeqAdvance) mSphereAnimTime += EnvInt("TAA_DT60", 0) ? 1.0f / 60.0f : gt.DeltaTime();''')
save("main.cpp")

# 渲染分辨率：窗口按 RENDER_SIZE 的客户区建（交换链跟着客户区走）
load(r"Common\D3DApp.cpp")
rep('''    const int winW = wr.right - wr.left, winH = wr.bottom - wr.top;
''', '''    char renderSize[64] = {};
    if (GetEnvironmentVariableA("RENDER_SIZE", renderSize, sizeof(renderSize)) > 0)
    {
        int w = 0, h = 0;
        sscanf_s(renderSize, "%d,%d", &w, &h);
        client = {w, h};
        wr = {0, 0, w, h};
        AdjustWindowRectExForDpi(&wr, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);
    }
    const int winW = wr.right - wr.left, winH = wr.bottom - wr.top;
''')
save(r"Common\D3DApp.cpp")
print("seq patch ok", D)
