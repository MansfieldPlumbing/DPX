cbuffer Params : register(b0) { uint M; uint N; uint K; uint BlockSize; uint HasZp; };
StructuredBuffer<float> A : register(t0);
StructuredBuffer<uint> Bq : register(t1);
StructuredBuffer<float> Scales : register(t2);
StructuredBuffer<uint> ZPs : register(t3);
RWStructuredBuffer<float> C : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint rowM = DTid.y, rowN = DTid.x;
    if (rowM >= M || rowN >= N) return;
    float acc = 0.0f;
    uint numBlocks = K / BlockSize;
    for (uint b = 0; b < numBlocks; b++) {
        float scale = Scales[rowN * numBlocks + b];
        uint zp = HasZp ? ((ZPs[(rowN * numBlocks + b) / 8] >> ((b % 8) * 4)) & 0xF) : 8;
        uint k0 = b * BlockSize;
        for (uint i=0; i<BlockSize; i+=2) {
            uint k = k0 + i;
            uint word = Bq[(rowN * K + k) / 8];
            uint shift = (k % 8) * 4;
            acc += scale * ((((word >> shift) & 0xF) - zp) * A[rowM * K + k] + 
                            (((word >> (shift + 4)) & 0xF) - zp) * A[rowM * K + k + 1]);
        }
    }
    C[rowM * N + rowN] = acc;
}
