#include "blake3/blake3.cuh"

namespace pearl {

// ─── cert-v3 domain-separation salts ────────────────────────────────────────
// blake3("pearl/cert-v3/noise-seed/A") / blake3("pearl/cert-v3/noise-seed/B").
// Consensus constants (zk-pow api/seed.rs); u32 little-endian words.
// Verified against the pinned test vectors (host-side derive.rs).
// NOTE: returned as CUTLASS_DEVICE functions (not namespace-scope arrays)
// so nvcc constant-folds them inside unrolled device loops.
CUTLASS_DEVICE static uint32_t seed_salt_a(int i) {
  switch (i) {
    case 0: return 0x6c404982u;
    case 1: return 0x1615eda0u;
    case 2: return 0x92f61696u;
    case 3: return 0xf876f0fcu;
    case 4: return 0x2adbdb92u;
    case 5: return 0x52b82370u;
    case 6: return 0x1977d4f0u;
    default: return 0x7b0190c3u;
  }
}

CUTLASS_DEVICE static uint32_t seed_salt_b(int i) {
  switch (i) {
    case 0: return 0x32063011u;
    case 1: return 0xca0163ecu;
    case 2: return 0x71afe22bu;
    case 3: return 0x4f4d3f8bu;
    case 4: return 0x39c6e91au;
    case 5: return 0x04cce888u;
    case 6: return 0x1d304448u;
    default: return 0xa99ab871u;
  }
}

class CommitmentHashFromMerkleRootsKernel {
 public:
  using Element = uint8_t;
  static constexpr uint32_t MaxThreadsPerBlock = 1;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SharedStorageSize = 0;

  using RmemChainingValueLayout =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemBlockLayout = Layout<Shape<Int<blake3::MSG_BLOCK_SIZE_U32>>>;

  // One keyed BLAKE3 block: msg is exactly 64 bytes, CV init = key words,
  // flags = single-chunk root. This is blake3::keyed_hash(key, msg) for a
  // 64-byte message (cert-v3 bind_root_a / bind_root_b).
  static constexpr blake3::CompressParams keyed_single_block_params = {
      .counter = 0,
      .block_len = blake3::MSG_BLOCK_SIZE,
      .flags = blake3::KEYED_HASH | blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT,
  };

  // Device side arguments
  struct Arguments {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
    // cert-v3 salting: 0 = legacy pre-V3 chain, 1 = salted (bind roots to
    // (m, n) with SEED_SALT_A/B before the unchanged seed chain).
    uint32_t m;
    uint32_t n;
    uint32_t salted;
  };

  // Kernel entry point API
  struct Params {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
    uint32_t m;
    uint32_t n;
    uint32_t salted;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.ptr_A_merkle_root, args.ptr_B_merkle_root, args.ptr_key,
            args.ptr_A_commitment_hash, args.ptr_B_commitment_hash,
            args.m, args.n, args.salted};
  }

  static dim3 get_grid_shape(Params const& params) { return dim3(1); }

  static dim3 get_block_shape() { return dim3(1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {

    uint32_t const* key_u32 = (uint32_t const*)params.ptr_key;
    uint32_t const* A_merkle_root_u32 =
        (uint32_t const*)params.ptr_A_merkle_root;
    uint32_t const* B_merkle_root_u32 =
        (uint32_t const*)params.ptr_B_merkle_root;

    // Optional cert-v3 root binding (salted derivation, cert_version == 3):
    //   bind_root_b = blake3(SEED_SALT_B, B_root || u32le(n) || 28 zeros)
    //   bind_root_a = blake3(SEED_SALT_A, A_root || u32le(m) || 28 zeros)
    // 32 + 4 + 28 = 64 bytes = exactly one BLAKE3 message block.
    Tensor rSaltedA = make_tensor<uint32_t>(RmemChainingValueLayout{});
    Tensor rSaltedB = make_tensor<uint32_t>(RmemChainingValueLayout{});

    if (params.salted) {
      Tensor rBindB = make_tensor<uint32_t>(RmemBlockLayout{});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rBindB(i) = B_merkle_root_u32[i];
        rBindB(i + blake3::CHAINING_VALUE_SIZE_U32) = 0;
      }
      rBindB(8) = params.n;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rSaltedB(i) = seed_salt_b(i);
      }
      blake3::compress_msg_block_u32(rBindB, rSaltedB,
                                      keyed_single_block_params);

      Tensor rBindA = make_tensor<uint32_t>(RmemBlockLayout{});
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rBindA(i) = A_merkle_root_u32[i];
        rBindA(i + blake3::CHAINING_VALUE_SIZE_U32) = 0;
      }
      rBindA(8) = params.m;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
        rSaltedA(i) = seed_salt_a(i);
      }
      blake3::compress_msg_block_u32(rBindA, rSaltedA,
                                      keyed_single_block_params);
    }

    // First calculate B commitment hash: BLAKE3(key || effective_B_root)
    Tensor rBlockB = make_tensor<uint32_t>(RmemBlockLayout{});
    Tensor rChainingValueB = make_tensor<uint32_t>(RmemChainingValueLayout{});

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlockB(i) = key_u32[i];
      rBlockB(i + blake3::CHAINING_VALUE_SIZE_U32) =
          params.salted ? rSaltedB(i) : B_merkle_root_u32[i];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValueB(i) = blake3::IV[i];
    }

    // We do one BLAKE3 block as we have exactly blake3::MSG_BLOCK_SIZE data
    static constexpr blake3::CompressParams single_block_params = {
        .counter = 0,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT,
    };
    blake3::compress_msg_block_u32(rBlockB, rChainingValueB,
                                   single_block_params);

    // Now calculate A commitment hash: BLAKE3(B_commitment_hash || effective_A_root)
    Tensor rBlockA = make_tensor<uint32_t>(RmemBlockLayout{});
    Tensor rChainingValueA = make_tensor<uint32_t>(RmemChainingValueLayout{});

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlockA(i) = rChainingValueB(i);  // Use B's result, not the key!
      rBlockA(i + blake3::CHAINING_VALUE_SIZE_U32) =
          params.salted ? rSaltedA(i) : A_merkle_root_u32[i];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValueA(i) = blake3::IV[i];
    }

    blake3::compress_msg_block_u32(rBlockA, rChainingValueA,
                                   single_block_params);

    // Copy the result back to global memory
    uint32_t* A_commitment_hash_u32 = (uint32_t*)params.ptr_A_commitment_hash;
    uint32_t* B_commitment_hash_u32 = (uint32_t*)params.ptr_B_commitment_hash;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      A_commitment_hash_u32[i] = rChainingValueA(i);
      B_commitment_hash_u32[i] = rChainingValueB(i);
    }
  }
};  // class CommitmentHashFromMerkleRootsKernel
}  // namespace pearl
