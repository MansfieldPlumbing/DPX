cbuffer Params : register(b0) { uint M; uint N; uint K; uint BlockSize; uint HasZp; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<uint> Bq : register(t1);
StructuredBuffer<float> Scales : register(t2);
StructuredBuffer<uint> ZPs : register(t3);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID) {
    uint row = DTid.y;
    if (row >= N) return;
    uint lane = GTid.x;
    float acc = 0.0f;
    uint numBlocks = K / BlockSize;
    for (uint b = 0; b < numBlocks; b++) {
        float scale = Scales[row * numBlocks + b];
        uint zp = HasZp ? ((ZPs[(row * numBlocks + b) / 8] >> ((b % 8) * 4)) & 0xF) : 8;
        uint k0 = b * BlockSize + lane * 2;
        uint wordIdx = (row * K + k0) / 8;
        uint word = Bq[wordIdx];
        uint shift = (k0 % 8) * 4;
        acc += scale * ((((word >> shift) & 0xF) - zp) * A[k0] + (((word >> (shift + 4)) & 0xF) - zp) * A[k0 + 1]);
    }
    float row_sum = WaveActiveSum(acc);
    if (WaveIsFirstLane()) C[row] = row_sum;
}
