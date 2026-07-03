using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace UMT
{
    /// <summary>
    /// Drives full skinning for one generated <see cref="SkinnedMeshRenderer"/> whose mesh contains SDEF vertices, in either GPU or CPU mode. In GPU mode a compute shader reads the bind-pose source mesh buffer and writes the renderer's output vertex buffer. In CPU mode a Burst job (<see cref="MMDSDEFSkinJob"/>) computes the skinned vertices, which are uploaded into an all-root-bone-weighted runtime pass-through mesh. Both modes apply vertex-morph blend shapes and bone skinning (linear blend skinning for BDEF vertices, the SDEF formula for SDEF vertices) and share one set of native buffers. The owning <see cref="MMDTransformManager"/> selects the mode, calls <see cref="Dispatch"/> (GPU) or <see cref="CpuScheduleJob"/>/<see cref="CpuApply"/> (CPU) after the bone solve, and completes the CPU jobs.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class MMDSDEFSkinner : MonoBehaviour, IDisposable
    {
        /// <summary>The skinned renderer whose deformation this skinner owns.</summary>
        public SkinnedMeshRenderer targetRenderer;
        /// <summary>Source PMX model providing the bone count.</summary>
        public PMXModel model;
        /// <summary>The original real-bone-weighted mesh, set at import; the single source of truth for restoring GPU mode and cloning the CPU runtime mesh.</summary>
        public Mesh sharedMesh;
        /// <summary>Mesh bindposes, one per bone, used to form the per-bone skinning matrices.</summary>
        [HideInInspector] public Matrix4x4[] bindposes = Array.Empty<Matrix4x4>();
        /// <summary>Per-vertex SDEF skinning data, in mesh-vertex order.</summary>
        [HideInInspector] public SDEFVertexData[] sdefVertexData = Array.Empty<SDEFVertexData>();
        /// <summary>Vertex-morph table feeding the per-frame morph weights.</summary>
        [HideInInspector] public MMDMorphTable morphTable;
        /// <summary>Whether the fixed vertex layout includes a tangent attribute (reserved; currently false).</summary>
        public bool hasTangent;

        private const int k_ThreadGroupSize = 64;
        private const int k_BatchSize = 64;

        private static readonly int s_SourceVerticesProperty = Shader.PropertyToID("_SourceVertices");
        private static readonly int s_DestVerticesProperty = Shader.PropertyToID("_DestVertices");
        private static readonly int s_VertexDataProperty = Shader.PropertyToID("_VertexData");
        private static readonly int s_BoneMatricesProperty = Shader.PropertyToID("_BoneMatrices");
        private static readonly int s_BoneQuaternionsProperty = Shader.PropertyToID("_BoneQuaternions");
        private static readonly int s_MorphWeightsProperty = Shader.PropertyToID("_MorphWeights");
        private static readonly int s_MorphOffsetsProperty = Shader.PropertyToID("_MorphOffsets");
        private static readonly int s_MorphRangesProperty = Shader.PropertyToID("_MorphRanges");
        private static readonly int s_VertexCountProperty = Shader.PropertyToID("_VertexCount");
        private static readonly int s_SourceStrideProperty = Shader.PropertyToID("_SourceStride");
        private static readonly int s_DestStrideProperty = Shader.PropertyToID("_DestStride");
        private static readonly int s_PositionOffsetProperty = Shader.PropertyToID("_PositionOffset");
        private static readonly int s_NormalOffsetProperty = Shader.PropertyToID("_NormalOffset");
        private static readonly int s_HasTangentProperty = Shader.PropertyToID("_HasTangent");
        private static readonly int s_TangentOffsetProperty = Shader.PropertyToID("_TangentOffset");

        private ComputeShader m_ComputeShader;
        private int m_KernelIndex;
        private int m_VertexCount;
        private int m_SourceStride;
        private int m_DestStride;
        private int m_PositionOffset;
        private int m_NormalOffset;
        private int m_TangentOffset;
        private int m_ThreadGroups;

        private GraphicsBuffer m_SourceBuffer;
        private GraphicsBuffer m_DestBuffer;
        private ComputeBuffer m_VertexDataBuffer;
        private ComputeBuffer m_BoneMatrixBuffer;
        private ComputeBuffer m_BoneQuatBuffer;
        private ComputeBuffer m_MorphWeightBuffer;
        private ComputeBuffer m_MorphOffsetsBuffer;
        private ComputeBuffer m_MorphRangeBuffer;
        private CommandBuffer m_CommandBuffer;
        private bool m_GPUInitialized;

        private NativeArray<float4x4> m_Bindposes;
        private NativeArray<SDEFVertexData> m_VertexData;
        private NativeArray<uint4> m_MorphOffsets;
        private NativeArray<int2> m_MorphRanges;
        private NativeArray<int> m_BlendShapeIndices;
        private NativeArray<float3> m_SourcePositions;
        private NativeArray<float3> m_SourceNormals;
        private NativeArray<float4x4> m_BoneMatrices;
        private NativeArray<float4> m_BoneQuaternions;
        private NativeArray<float> m_MorphWeights;
        private NativeArray<float3> m_OutPositions;
        private NativeArray<float3> m_OutNormals;

        private Mesh m_RuntimeMesh;
        private Transform[] m_RealBones;
        private Transform m_RootBone;
        private bool m_CPUReady;

        private void OnEnable()
        {
            m_GPUInitialized = false;
            m_CPUReady = false;
        }

        /// <summary>
        /// (Re)allocates and fills the common native buffers from the serialized skinning data whenever their length no longer matches the serialized source. Static inputs are copied on resize; per-frame buffers are only sized here and filled each frame.
        /// </summary>
        private void EnsureNativeBuffers()
        {
            if (sharedMesh == null)
            {
                sharedMesh = targetRenderer.sharedMesh;
            }
            m_VertexCount = sharedMesh.vertexCount;

            if (!m_Bindposes.IsCreated || m_Bindposes.Length != bindposes.Length)
            {
                PMXUtilities.ResizePersistent(ref m_Bindposes, bindposes.Length);
                for (int i = 0; i < bindposes.Length; ++i)
                {
                    m_Bindposes[i] = bindposes[i];
                }
            }

            if (!m_VertexData.IsCreated || m_VertexData.Length != sdefVertexData.Length)
            {
                PMXUtilities.ResizePersistent(ref m_VertexData, sdefVertexData.Length);
                m_VertexData.CopyFrom(sdefVertexData);
            }

            int morphOffsetsCount = math.max(morphTable != null ? morphTable.flatOffsets.Length : 0, 1);
            if (!m_MorphOffsets.IsCreated || m_MorphOffsets.Length != morphOffsetsCount)
            {
                PMXUtilities.ResizePersistent(ref m_MorphOffsets, morphOffsetsCount);
                if (morphTable != null && morphTable.flatOffsets.Length > 0)
                {
                    NativeArray<uint4>.Copy(morphTable.flatOffsets, m_MorphOffsets, morphTable.flatOffsets.Length);
                }
            }

            if (!m_MorphRanges.IsCreated || m_MorphRanges.Length != m_VertexCount)
            {
                PMXUtilities.ResizePersistent(ref m_MorphRanges, m_VertexCount);
                if (morphTable != null && morphTable.perVertexRanges.Length == m_VertexCount)
                {
                    m_MorphRanges.CopyFrom(morphTable.perVertexRanges);
                }
            }

            int morphSlotCount = math.max(morphTable != null ? morphTable.blendShapeNames.Length : 0, 1);
            if (!m_BlendShapeIndices.IsCreated || m_BlendShapeIndices.Length != morphSlotCount)
            {
                PMXUtilities.ResizePersistent(ref m_BlendShapeIndices, morphSlotCount);
                for (int slot = 0; slot < m_BlendShapeIndices.Length; ++slot)
                {
                    m_BlendShapeIndices[slot] = morphTable != null && slot < morphTable.blendShapeNames.Length ? sharedMesh.GetBlendShapeIndex(morphTable.blendShapeNames[slot]) : -1;
                }
            }

            if (!m_MorphWeights.IsCreated || m_MorphWeights.Length != morphSlotCount)
            {
                PMXUtilities.ResizePersistent(ref m_MorphWeights, morphSlotCount);
            }

            if (!m_SourcePositions.IsCreated || m_SourcePositions.Length != m_VertexCount)
            {
                PMXUtilities.ResizePersistent(ref m_SourcePositions, m_VertexCount);
                PMXUtilities.ResizePersistent(ref m_SourceNormals, m_VertexCount);
                FillSourceGeometry();
            }

            int boneCount = math.max(model.bones.Length, 1);
            if (!m_BoneMatrices.IsCreated || m_BoneMatrices.Length != boneCount)
            {
                PMXUtilities.ResizePersistent(ref m_BoneMatrices, boneCount);
                PMXUtilities.ResizePersistent(ref m_BoneQuaternions, boneCount);
            }

            if (!m_OutPositions.IsCreated || m_OutPositions.Length != m_VertexCount)
            {
                PMXUtilities.ResizePersistent(ref m_OutPositions, m_VertexCount);
                PMXUtilities.ResizePersistent(ref m_OutNormals, m_VertexCount);
            }
        }

        /// <summary>
        /// Copies the bind-pose source positions and normals from the original mesh into the native source buffers the CPU job reads.
        /// </summary>
        private void FillSourceGeometry()
        {
            Vector3[] positions = sharedMesh.vertices;
            Vector3[] normals = sharedMesh.normals;
            for (int i = 0; i < m_SourcePositions.Length; ++i)
            {
                m_SourcePositions[i] = positions[i];
                m_SourceNormals[i] = i < normals.Length ? (float3)normals[i] : float3.zero;
            }
        }

        /// <summary>
        /// Allocates the GPU buffers and command buffer for this renderer's mesh. Must be called once with the SDEF compute shader before the first dispatch; the mesh must be GPU-resident at this point.
        /// </summary>
        /// <param name="shader">The SDEF skinning compute shader.</param>
        public void Initialize(ComputeShader shader)
        {
            if (shader == null)
            {
                Debug.LogError("Compute shader is null; cannot initialize GPU SDEF skinner.");
                return;
            }
            if (targetRenderer == null || targetRenderer.sharedMesh == null)
            {
                Debug.LogWarning("Target renderer or its shared mesh is null.");
                return;
            }

            Mesh mesh = targetRenderer.sharedMesh;
            mesh.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
            m_SourceBuffer = mesh.GetVertexBuffer(0);
            if (m_SourceBuffer == null)
            {
                Debug.LogWarning("Mesh source vertex buffer is null; the mesh may not be GPU-resident yet.");
                return;
            }

            targetRenderer.vertexBufferTarget |= GraphicsBuffer.Target.Raw | GraphicsBuffer.Target.Vertex;
            m_DestBuffer = targetRenderer.GetVertexBuffer();
            if (m_DestBuffer == null)
            {
                return; // The renderer may not have allocated its output buffer yet; it will be acquired on the first dispatch.
            }
            m_VertexCount = mesh.vertexCount;
            if (sdefVertexData.Length != m_VertexCount)
            {
                Debug.LogWarning($"SDEF vertex data count ({sdefVertexData.Length}) does not match mesh vertex count ({m_VertexCount}).");
                return;
            }

            EnsureNativeBuffers();

            m_ComputeShader = shader;
            m_KernelIndex = shader.FindKernel("CSSkin");
            m_ThreadGroups = (m_VertexCount + k_ThreadGroupSize - 1) / k_ThreadGroupSize;

            // The authored source buffer (interleaved) and the skinned destination buffer (Unity's deformed stream) usually differ in stride, but both place position and normal at the same offsets. Query the real strides/offsets instead of assuming a fixed layout.
            m_SourceStride = m_SourceBuffer.stride;
            m_DestStride = m_DestBuffer.stride;
            m_PositionOffset = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
            m_NormalOffset = mesh.GetVertexAttributeOffset(VertexAttribute.Normal);
            m_TangentOffset = hasTangent ? mesh.GetVertexAttributeOffset(VertexAttribute.Tangent) : 0;

            m_VertexDataBuffer = new ComputeBuffer(m_VertexCount, UnsafeStrideOf<SDEFVertexData>(), ComputeBufferType.Structured);
            m_VertexDataBuffer.SetData(m_VertexData);

            int boneCount = math.max(model.bones.Length, 1);
            m_BoneMatrixBuffer = new ComputeBuffer(boneCount, UnsafeStrideOf<float4x4>(), ComputeBufferType.Structured);
            m_BoneQuatBuffer = new ComputeBuffer(boneCount, UnsafeStrideOf<float4>(), ComputeBufferType.Structured);

            m_MorphWeightBuffer = new ComputeBuffer(m_MorphWeights.Length, sizeof(float), ComputeBufferType.Structured);

            m_MorphOffsetsBuffer = new ComputeBuffer(m_MorphOffsets.Length * 4, 4, ComputeBufferType.Raw);
            m_MorphOffsetsBuffer.SetData(m_MorphOffsets);

            m_MorphRangeBuffer = new ComputeBuffer(m_VertexCount, UnsafeStrideOf<int2>(), ComputeBufferType.Structured);
            m_MorphRangeBuffer.SetData(m_MorphRanges);

            m_CommandBuffer = new CommandBuffer { name = "MMD SDEF Skinning" };
            m_GPUInitialized = true;

            Debug.Log($"[UMT SDEF] {targetRenderer.name}: verts={m_VertexCount} sourceStride={m_SourceStride} destStride={m_DestStride} (destCount={m_DestBuffer.count}) posOffset={m_PositionOffset} normalOffset={m_NormalOffset} bones={boneCount} morphSlots={m_BlendShapeIndices.Length}", targetRenderer);
        }

        /// <summary>
        /// Updates the per-frame bone matrices, bone rotation quaternions, and morph weights, then encodes and executes the GPU compute dispatch that overwrites the renderer's output vertex buffer. Tears down any active CPU runtime mesh first so a GPU-mode frame runs cleanly.
        /// </summary>
        /// <param name="shader">The SDEF compute shader, used to lazily initialize on the first dispatch.</param>
        public void Dispatch(ComputeShader shader)
        {
            if (m_CPUReady)
            {
                TeardownCPU();
            }

            if (!m_GPUInitialized)
            {
                Initialize(shader);
            }

            if (!m_GPUInitialized || m_BoneMatrixBuffer == null || m_BoneQuatBuffer == null || m_MorphWeightBuffer == null || m_MorphOffsetsBuffer == null || m_MorphRangeBuffer == null)
            {
                return;
            }

            // Source the skinning matrices from the live bone transforms rather than the transform solver so SDEF deformation tracks whatever poses the bones (the MMD solver, an Animator, or manual posing) and is independent of the solver's transformEnabled state.
            Transform[] bones = targetRenderer.bones;
            int boneCount = math.min(bones.Length, m_BoneMatrices.Length);
            // The skinned buffer is expressed in the renderer's rootBone space (Unity skins relative to the rootBone, not the SMR's own transform), matching the bindposes Unity uses for the non-SDEF meshes. Prefix each skinning matrix with the rootBone's inverse world matrix so the deformed vertices land in that space and respect the model root transform.
            Transform rootBone = targetRenderer.rootBone != null ? targetRenderer.rootBone : targetRenderer.transform;
            float4x4 worldToObject = (float4x4)rootBone.worldToLocalMatrix;
            for (int b = 0; b < boneCount; ++b)
            {
                Matrix4x4 boneLocalToWorld = bones[b] != null ? bones[b].localToWorldMatrix : Matrix4x4.identity;
                float4x4 skinMatrix = math.mul(worldToObject, math.mul((float4x4)boneLocalToWorld, m_Bindposes[b]));
                m_BoneMatrices[b] = skinMatrix;
                m_BoneQuaternions[b] = ExtractRotation(skinMatrix).value;
            }
            m_BoneMatrixBuffer.SetData(m_BoneMatrices, 0, 0, boneCount);
            m_BoneQuatBuffer.SetData(m_BoneQuaternions, 0, 0, boneCount);

            if (morphTable != null && morphTable.blendShapeNames.Length > 0)
            {
                for (int slot = 0; slot < m_BlendShapeIndices.Length; ++slot)
                {
                    int blendShapeIndex = m_BlendShapeIndices[slot];
                    m_MorphWeights[slot] = blendShapeIndex >= 0 ? targetRenderer.GetBlendShapeWeight(blendShapeIndex) / 100.0f : 0.0f;
                }
                m_MorphWeightBuffer.SetData(m_MorphWeights);
            }

            // Re-acquire the destination buffer each frame; the renderer may reallocate it (e.g. on enable or motion-vector buffer swaps).
            m_DestBuffer?.Dispose();
            m_DestBuffer = targetRenderer.GetVertexBuffer();
            if (m_DestBuffer == null)
            {
                Debug.LogWarning("Skinned renderer vertex buffer became unavailable during dispatch.");
                return;
            }

            m_CommandBuffer.Clear();
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_SourceVerticesProperty, m_SourceBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_DestVerticesProperty, m_DestBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_VertexDataProperty, m_VertexDataBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_BoneMatricesProperty, m_BoneMatrixBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_BoneQuaternionsProperty, m_BoneQuatBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_MorphWeightsProperty, m_MorphWeightBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_MorphOffsetsProperty, m_MorphOffsetsBuffer);
            m_CommandBuffer.SetComputeBufferParam(m_ComputeShader, m_KernelIndex, s_MorphRangesProperty, m_MorphRangeBuffer);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_VertexCountProperty, m_VertexCount);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_SourceStrideProperty, m_SourceStride);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_DestStrideProperty, m_DestStride);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_PositionOffsetProperty, m_PositionOffset);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_NormalOffsetProperty, m_NormalOffset);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_HasTangentProperty, hasTangent ? 1 : 0);
            m_CommandBuffer.SetComputeIntParam(m_ComputeShader, s_TangentOffsetProperty, m_TangentOffset);
            m_CommandBuffer.DispatchCompute(m_ComputeShader, m_KernelIndex, m_ThreadGroups, 1, 1);
            Graphics.ExecuteCommandBuffer(m_CommandBuffer);
        }

        /// <summary>
        /// Gathers the per-frame bone matrices and morph weights into the common buffers, then schedules the CPU SDEF skinning job. Tears down any active GPU buffers and builds the runtime pass-through mesh on the first CPU call. The returned handle is completed by the owning <see cref="MMDTransformManager"/>; do not complete it here.
        /// </summary>
        /// <returns>The scheduled job handle, or the default handle when the skinner is not usable.</returns>
        public JobHandle CpuScheduleJob()
        {
            if (targetRenderer == null || sharedMesh == null)
            {
                return default;
            }

            if (m_GPUInitialized)
            {
                TeardownGPU();
            }

            EnsureNativeBuffers();

            if (!m_CPUReady)
            {
                InitializeCPU();
            }

            int boneCount = math.min(m_RealBones.Length, m_BoneMatrices.Length);
            float4x4 worldToObject = (float4x4)m_RootBone.worldToLocalMatrix;
            for (int b = 0; b < boneCount; ++b)
            {
                Matrix4x4 boneLocalToWorld = m_RealBones[b] != null ? m_RealBones[b].localToWorldMatrix : Matrix4x4.identity;
                float4x4 skinMatrix = math.mul(worldToObject, math.mul((float4x4)boneLocalToWorld, m_Bindposes[b]));
                m_BoneMatrices[b] = skinMatrix;
                m_BoneQuaternions[b] = ExtractRotation(skinMatrix).value;
            }

            for (int slot = 0; slot < m_BlendShapeIndices.Length; ++slot)
            {
                int blendShapeIndex = m_BlendShapeIndices[slot];
                m_MorphWeights[slot] = blendShapeIndex >= 0 ? targetRenderer.GetBlendShapeWeight(blendShapeIndex) / 100.0f : 0.0f;
            }

            MMDSDEFSkinJob job = new MMDSDEFSkinJob
            {
                sourcePositions = m_SourcePositions,
                sourceNormals = m_SourceNormals,
                vertexData = m_VertexData,
                boneMatrices = m_BoneMatrices,
                boneQuaternions = m_BoneQuaternions,
                morphWeights = m_MorphWeights,
                morphOffsets = m_MorphOffsets,
                morphRanges = m_MorphRanges,
                outPositions = m_OutPositions,
                outNormals = m_OutNormals,
            };
            return job.Schedule(m_OutPositions.Length, k_BatchSize);
        }

        /// <summary>
        /// Uploads the completed CPU job output into the runtime pass-through mesh. Call only after the scheduled job handle has completed.
        /// </summary>
        public void CpuApply()
        {
            if (m_RuntimeMesh == null || !m_OutPositions.IsCreated)
            {
                return;
            }

            m_RuntimeMesh.SetVertices(m_OutPositions.Reinterpret<Vector3>());
            m_RuntimeMesh.SetNormals(m_OutNormals.Reinterpret<Vector3>());
        }

        /// <summary>
        /// Builds the all-root-bone-weighted runtime pass-through mesh, neutralizes its blend shapes, and swaps it onto the renderer so Unity's skinning passes the CPU-computed rootBone-space vertices through unchanged. Captures the real bones and root bone first so they can be restored.
        /// </summary>
        private void InitializeCPU()
        {
            m_RealBones = targetRenderer.bones;
            m_RootBone = targetRenderer.rootBone != null ? targetRenderer.rootBone : targetRenderer.transform;

            m_RuntimeMesh = Instantiate(sharedMesh);
            m_RuntimeMesh.hideFlags = HideFlags.DontSave;
            m_RuntimeMesh.MarkDynamic();

            int vertexCount = m_RuntimeMesh.vertexCount;
            int shapeCount = sharedMesh.blendShapeCount;
            string[] shapeNames = new string[shapeCount];
            for (int s = 0; s < shapeCount; ++s)
            {
                shapeNames[s] = sharedMesh.GetBlendShapeName(s);
            }

            // Keep the blend-shape slots (same names/order) so animation still drives GetBlendShapeWeight, but zero every channel so Unity's own blend-shape deformation contributes nothing; the job applies the real morph deltas.
            Vector3[] zeroDeltas = new Vector3[vertexCount];
            m_RuntimeMesh.ClearBlendShapes();
            for (int s = 0; s < shapeCount; ++s)
            {
                m_RuntimeMesh.AddBlendShapeFrame(shapeNames[s], 100.0f, zeroDeltas, zeroDeltas, zeroDeltas);
            }

            // Weight every vertex 1.0 to a single bone (the root bone) with an identity bindpose so Unity's SMR skinning becomes a pass-through of the CPU-computed rootBone-space vertices.
            NativeArray<byte> bonesPerVertex = new NativeArray<byte>(vertexCount, Allocator.Temp);
            NativeArray<BoneWeight1> weights = new NativeArray<BoneWeight1>(vertexCount, Allocator.Temp);
            for (int i = 0; i < vertexCount; ++i)
            {
                bonesPerVertex[i] = 1;
                weights[i] = new BoneWeight1 { boneIndex = 0, weight = 1.0f };
            }
            m_RuntimeMesh.SetBoneWeights(bonesPerVertex, weights);
            m_RuntimeMesh.bindposes = new Matrix4x4[] { Matrix4x4.identity };
            bonesPerVertex.Dispose();
            weights.Dispose();

            targetRenderer.sharedMesh = m_RuntimeMesh;
            targetRenderer.bones = new Transform[] { m_RootBone };
            targetRenderer.rootBone = m_RootBone;

            m_CPUReady = true;
        }

        /// <summary>
        /// Restores the renderer to its original mesh, bones, and root bone and destroys the CPU runtime pass-through mesh, if one is active. No-op in GPU mode or when CPU skinning was never started. Called when CPU SDEF skinning is turned off or the mode leaves CPU at runtime.
        /// </summary>
        public void RestoreOriginalMesh()
        {
            if (m_CPUReady)
            {
                TeardownCPU();
            }
        }

        /// <summary>
        /// Extracts the rotation quaternion from a skinning matrix, normalizing the basis columns so skew or scale in the matrix does not corrupt the quaternion (PMX skinning is rigid, so columns are near orthonormal).
        /// </summary>
        private static quaternion ExtractRotation(float4x4 matrix)
        {
            float3 column0 = math.normalizesafe(matrix.c0.xyz, new float3(1.0f, 0.0f, 0.0f));
            float3 column1 = math.normalizesafe(matrix.c1.xyz, new float3(0.0f, 1.0f, 0.0f));
            float3 column2 = math.normalizesafe(matrix.c2.xyz, new float3(0.0f, 0.0f, 1.0f));
            return math.normalize(new quaternion(new float3x3(column0, column1, column2)));
        }

        private static int UnsafeStrideOf<T>() where T : struct
        {
            return UnsafeUtility.SizeOf<T>();
        }

        private void OnDestroy()
        {
            Dispose();
        }

        private void OnDisable()
        {
            Dispose();
        }

        /// <summary>
        /// Releases the GPU buffers, restores the renderer to its original mesh/bones, destroys the CPU runtime mesh, and disposes the common native buffers.
        /// </summary>
        public void Dispose()
        {
            TeardownGPU();
            TeardownCPU();
            DisposeNativeBuffers();
        }

        /// <summary>
        /// Releases the GPU buffers and command buffer.
        /// </summary>
        private void TeardownGPU()
        {
            m_SourceBuffer?.Dispose();
            m_DestBuffer?.Dispose();
            m_VertexDataBuffer?.Dispose();
            m_BoneMatrixBuffer?.Dispose();
            m_BoneQuatBuffer?.Dispose();
            m_MorphWeightBuffer?.Dispose();
            m_MorphOffsetsBuffer?.Dispose();
            m_MorphRangeBuffer?.Dispose();
            m_SourceBuffer = null;
            m_DestBuffer = null;
            m_VertexDataBuffer = null;
            m_BoneMatrixBuffer = null;
            m_BoneQuatBuffer = null;
            m_MorphWeightBuffer = null;
            m_MorphOffsetsBuffer = null;
            m_MorphRangeBuffer = null;

            m_CommandBuffer?.Dispose();
            m_CommandBuffer = null;

            m_GPUInitialized = false;
        }

        /// <summary>
        /// Restores the renderer's original mesh, bones, and root bone, then destroys the runtime pass-through mesh.
        /// </summary>
        private void TeardownCPU()
        {
            if (targetRenderer != null)
            {
                if (sharedMesh != null)
                {
                    targetRenderer.sharedMesh = sharedMesh;
                }
                if (m_RealBones != null)
                {
                    targetRenderer.bones = m_RealBones;
                }
                if (m_RootBone != null)
                {
                    targetRenderer.rootBone = m_RootBone;
                }
            }

            if (m_RuntimeMesh != null)
            {
                PMXUtilities.DestroyRuntimeObject(m_RuntimeMesh);
                m_RuntimeMesh = null;
            }

            m_CPUReady = false;
        }

        /// <summary>
        /// Disposes every common native buffer shared by the GPU and CPU paths.
        /// </summary>
        private void DisposeNativeBuffers()
        {
            DisposeNativeArray(ref m_Bindposes);
            DisposeNativeArray(ref m_VertexData);
            DisposeNativeArray(ref m_MorphOffsets);
            DisposeNativeArray(ref m_MorphRanges);
            DisposeNativeArray(ref m_BlendShapeIndices);
            DisposeNativeArray(ref m_SourcePositions);
            DisposeNativeArray(ref m_SourceNormals);
            DisposeNativeArray(ref m_BoneMatrices);
            DisposeNativeArray(ref m_BoneQuaternions);
            DisposeNativeArray(ref m_MorphWeights);
            DisposeNativeArray(ref m_OutPositions);
            DisposeNativeArray(ref m_OutNormals);
        }

        private static void DisposeNativeArray<T>(ref NativeArray<T> array) where T : struct
        {
            if (array.IsCreated)
            {
                array.Dispose();
            }
        }
    }
}
