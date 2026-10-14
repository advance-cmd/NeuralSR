# 神经超分的数据导出（在 ssm 副本的基础上）：
#   DUMP_CAMS=机位表（每行 ex,ey,ez,tx,ty,tz）  DUMP_DIR=输出目录  DUMP_WARMUP=预热帧数（默认 30）
#   RENDER_SIZE=W,H：渲染分辨率（交换链、G-Buffer、深度都按它建）
#   第 warmup+k 帧用第 k 个机位，读回 后台缓冲 / G-Buffer0 / G-Buffer1 / 深度，写成 .npy；最后一个机位写完就退出
#   导数据时不画 ImGui
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, sys.argv[1])
s = ""


def load(p):
    global s
    s = open(os.path.join(D, p), encoding="utf-8-sig").read().replace("\r\n", "\n")


def save(p):
    open(os.path.join(D, p), "w", encoding="utf-8", newline="\n").write(s)


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


load("main.cpp")
rep('''#include <random>
''', '''#include <random>
#include <array>
#include <fstream>
#include <string>
''')
rep('''static std::wstring gSnapPath, gDumpPath;
''', '''static std::wstring gSnapPath, gDumpPath;
static std::vector<std::array<float, 6>> gDataCams; // 导数据：机位表
static std::string gDataDir;
static int gDataWarmup = 30;
static int gFrameNo = 0;
static int DataCamIndex() { return gDataCams.empty() ? -1 : gFrameNo - 1 - gDataWarmup; } // 这一帧用第几个机位（<0 = 预热）

// 读回缓冲 → .npy（去掉行距）。shape 按 (h, w, c) 写；c = 1 时写 (h, w)
static void WriteNpy(const std::string& path, ID3D12Resource* rb, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,
                     const char* descr, UINT channels, UINT bytesPerTexel)
{
    const UINT w = fp.Footprint.Width, h = fp.Footprint.Height, pitch = fp.Footprint.RowPitch;
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" +
                         std::to_string(h) + ", " + std::to_string(w) + (channels > 1 ? ", " + std::to_string(channels) : "") + "), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\\n';
    std::ofstream f(path, std::ios::binary);
    const char magic[] = "\\x93NUMPY\\x01\\x00";
    f.write(magic, 8);
    const uint16_t hl = (uint16_t)header.size();
    f.write((const char*)&hl, 2);
    f.write(header.data(), header.size());
    BYTE* data = nullptr;
    ThrowIfFailed(rb->Map(0, nullptr, (void**)&data));
    for (UINT y = 0; y < h; ++y) f.write((const char*)(data + (size_t)y * pitch), (std::streamsize)w * bytesPerTexel);
    D3D12_RANGE noWrite = {0, 0};
    rb->Unmap(0, &noWrite);
}
''')
# 机位、输出目录、预热帧数
rep('''    if (GetEnvironmentVariableA("SHADOW_CAM", env, sizeof(env)) > 0)
''', '''    if (GetEnvironmentVariableA("DUMP_CAMS", env, sizeof(env)) > 0)
    {
        std::ifstream cams(env);
        std::string line;
        while (std::getline(cams, line))
        {
            std::array<float, 6> c{};
            if (sscanf_s(line.c_str(), "%f,%f,%f,%f,%f,%f", &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) == 6) gDataCams.push_back(c);
        }
        char dir[MAX_PATH] = {};
        if (GetEnvironmentVariableA("DUMP_DIR", dir, sizeof(dir)) > 0) gDataDir = dir;
        gDataWarmup = EnvInt("DUMP_WARMUP", 30);
        char msg[128];
        sprintf_s(msg, "[DATA] %zu cams, warmup %d\\n", gDataCams.size(), gDataWarmup);
        OutputDebugStringA(msg);
    }
    if (GetEnvironmentVariableA("SHADOW_CAM", env, sizeof(env)) > 0)
''')
# 每帧开头：帧号 +1，到了就换机位
rep('''    void Update(const GameTimer& gt) override
    {
''', '''    void Update(const GameTimer& gt) override
    {
        ++gFrameNo;
        if (const int k = DataCamIndex(); k >= 0 && k < (int)gDataCams.size())
        {
            const auto& c = gDataCams[k];
            mCamera.LookAt(XMFLOAT3(c[0], c[1], c[2]), XMFLOAT3(c[3], c[4], c[5]), XMFLOAT3(0.0f, 1.0f, 0.0f));
        }
''')
# 不画 UI
rep('''        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), mCommandList.Get());
''', '''        if (gDataCams.empty()) ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), mCommandList.Get());
''')
# 读回：后台缓冲（帧末，RT）、G-Buffer（PSR）；深度在光照 Pass 刚结束时读（只有不透明物体，和 G-Buffer 对得上）
rep('''static void WriteNpy(''', '''static ComPtr<ID3D12Resource> gDataRb[4]; // 0 成品 1 G-Buffer0 2 G-Buffer1 3 深度
static D3D12_PLACED_SUBRESOURCE_FOOTPRINT gDataFp[4];

static void WriteNpy(''')
rep('''    static void WriteBmp(''', '''    // 导数据：src（当前在 state）→ 读回缓冲 gDataRb[i]，录完转回 state
    void CopyForData(int i, ID3D12Resource* src, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC desc = src->GetDesc();
        UINT64 total = 0;
        mDevice->GetCopyableFootprints(&desc, 0, 1, 0, &gDataFp[i], nullptr, nullptr, &total);
        if (!gDataRb[i])
        {
            CD3DX12_HEAP_PROPERTIES rbHeap(D3D12_HEAP_TYPE_READBACK);
            CD3DX12_RESOURCE_DESC rbDesc = CD3DX12_RESOURCE_DESC::Buffer(total);
            ThrowIfFailed(mDevice->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
                                                           D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gDataRb[i])));
        }
        auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(src, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mCommandList->ResourceBarrier(1, &toCopy);
        CD3DX12_TEXTURE_COPY_LOCATION dst(gDataRb[i].Get(), gDataFp[i]), from(src, 0);
        mCommandList->CopyTextureRegion(&dst, 0, 0, 0, &from, nullptr);
        auto back = CD3DX12_RESOURCE_BARRIER::Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        mCommandList->ResourceBarrier(1, &back);
    }

    static void WriteBmp(''')
rep('''        mCommandList->ResourceBarrier(1, &depthBack);
''', '''        mCommandList->ResourceBarrier(1, &depthBack);
        if (const int k = DataCamIndex(); k >= 0 && k < (int)gDataCams.size())
            CopyForData(3, mDepthStencilBuffer.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
''')
rep('''        // 后台缓冲：RENDER_TARGET → PRESENT
''', '''        const int dataCam = DataCamIndex();
        const bool dataFrame = dataCam >= 0 && dataCam < (int)gDataCams.size();
        if (dataFrame)
        {
            CopyForData(0, CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET);
            CopyForData(1, mGBuffer[0].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            CopyForData(2, mGBuffer[1].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // 后台缓冲：RENDER_TARGET → PRESENT
''')
rep('''        if (snap)
        {
            FlushCommandQueue();
''', '''        if (dataFrame)
        {
            FlushCommandQueue();
            char base[MAX_PATH];
            sprintf_s(base, "%s\\\\%04d", gDataDir.c_str(), dataCam);
            WriteNpy(std::string(base) + "_color.npy", gDataRb[0].Get(), gDataFp[0], "|u1", 4, 4);
            WriteNpy(std::string(base) + "_gb0.npy", gDataRb[1].Get(), gDataFp[1], "|u1", 4, 4);
            WriteNpy(std::string(base) + "_gb1.npy", gDataRb[2].Get(), gDataFp[2], "<f2", 4, 8);
            WriteNpy(std::string(base) + "_depth.npy", gDataRb[3].Get(), gDataFp[3], "<u4", 1, 4);
            if (dataCam % 20 == 0 || dataCam + 1 == (int)gDataCams.size())
            {
                char msg[128];
                sprintf_s(msg, "[DATA] cam %d / %zu  %ux%u  depth fmt %d\\n", dataCam + 1, gDataCams.size(),
                          gDataFp[0].Footprint.Width, gDataFp[0].Footprint.Height, (int)gDataFp[3].Footprint.Format);
                OutputDebugStringA(msg);
            }
            if (dataCam + 1 == (int)gDataCams.size()) PostQuitMessage(0);
        }
        if (snap)
        {
            FlushCommandQueue();
''')
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
print("dump ok", D)
